# Hovering / Waypoint-Following Environment — Specification

A manager-based IsaacLab task for training a policy that flies a quadcopter
through a sequence of 3D waypoints, intended for **sim-to-real deployment**
against a motion-capture setup that exposes only position + orientation.

Sibling package to `racing_quadcopter`. No cross-coupling beyond reusing the
rates-PID `ActionTerm`.

---

## 1. Task

The drone must visit waypoints in order. Each waypoint has a 3D position only
(no orientation). A waypoint is **collected** once the drone has remained
within proximity of it for a configurable **collection time** `T_c` (seconds).
On collection the target advances to the next waypoint; the episode ends when
all waypoints have been collected or `episode_length_s` elapses.

### Two-radius proximity hysteresis

| state              | condition                            | action                                |
|--------------------|--------------------------------------|---------------------------------------|
| **idle**           | `d ≥ r_capture`                      | hold idle, timer = 0                  |
| idle → **holding** | `d < r_capture`                      | enter holding, timer = 0              |
| holding            | `d < r_hold` (looser)                | timer += dt                           |
| holding → idle     | `d > r_hold`                         | exit holding, timer = 0               |
| holding → **collected** | `timer ≥ T_c`                  | advance waypoint, reset state         |

- `r_capture` (strict, e.g. **0.20 m**): required to *enter* holding.
- `r_hold` (lax, e.g. **0.50 m**): required to *stay* in holding.
- `T_c = 0`: collection on first entry, no holding phase.
- The looser hold radius tolerates drift from pushes/disturbances while still
  forcing a re-approach if the drone wanders off.

Both radii and `T_c` are env-cfg fields.

---

## 2. Action space (mirrors racing)

Collective thrust + body rates (4-dim) → inner-loop rate PID → body moments.
Implemented by **re-using** `mdp.actions.QuadcopterRatesActionCfg` (shared `tasks/manager_based/mdp`)
unchanged. Drone profile (mass, PID gains) comes from hovering's **own**
`HOVERING_DRONE_PROFILES` dict in `hovering_quadcopter_env_cfg.py` (a separate copy from racing's
`DRONE_PROFILES`, so retuning one does not touch the other), defaulting to **`anemos_urdf`**,
switchable to `anemos_usd` via the `HOVERING_DRONE_MODEL` env var — deliberately *not* the racing
`DRONE_MODEL` env var, so the two tasks can run side-by-side with different drone choices. The
crazyflie profile is intentionally not wired in.

### Profile overrides for hovering

Two racing-tuned profile values are overridden in `ActionsCfg`:

- **`thrust_to_weight = 1.9`** (was ≈4.1 from the anemos profile). At T2W≈4.1
  the hover-equilibrium thrust action is ≈ −0.51, far from the policy's
  initial mean of 0. With `initial_log_std=1.0` the early actor centred at
  action≈0 climbs at ~1g until it hits the ceiling, never settling. T2W=1.9
  puts hover at action ≈ 0 (matches the old direct hovering env).
- **`rate_command_scale = 6.5`** rad/s (was 30 from the racing profile).
  For a hovering task a tight commanded body-rate envelope is plenty; the
  racing slew rate produces poor low-speed precision and lets the
  under-authoritative yaw axis (`pid_p` for yaw is far weaker than
  roll/pitch on anemos) drift into a persistent spin.

### Action noise / lag

`ActionNoiseCfg` is mirrored from racing (per-step Gaussian, per-episode
bias, first-order LPF for actuator lag). Defaults are off; flip
`enable=True` to activate. The racing `QuadcopterRatesAction` reads these
fields via `getattr` on the env cfg, so the same action term is used
unchanged.

---

## 3. Observation space (asymmetric actor / critic)

Deployment constraint: real-world mocap yields only position + orientation.
Everything the **actor** sees must be derivable from those plus on-board IMU
(gyro is cheap; linear velocity needs state estimation that isn't wired up
yet). The **critic** sees a strict superset including the privileged
`base_lin_vel_b` straight from sim — value-baseline quality improves without
the actor inheriting a deployment-impossible signal.

History window: **1** step for both groups (`_HOVER_OBS_WINDOW = 1` in
`hovering_quadcopter_env_cfg.py`; a GRU agent config owns temporal memory instead, so history is
not duplicated in both the observation tensor and the recurrent hidden state).

### Actor (`policy` ObsGroup)

| term                       | dim | notes                                                       |
|----------------------------|-----|-------------------------------------------------------------|
| `projected_gravity_b`      | 3   | derived from quat (orientation-only)                        |
| `base_ang_vel_b`           | 3   | gyro (IMU)                                                  |
| `next_waypoints_rel_b`     | 6   | next + next-next waypoints in body frame (Δpos rotated)     |
| `mocap_lin_vel_b`          | 3   | filtered finite-difference velocity from mocap poses (`mocap_velocity_obs_enabled`) |
| `collection_time_const`    | 1   | constant `T_c` for the cell (lets policy modulate behaviour — race through if 0, dwell if large) |
| `collection_progress`      | 1   | `timer / T_c` ∈ [0, 1] (0 if `T_c == 0`)                    |
| `last_action`              | 4   | last **applied** action (post noise/lag; `mdp.observations.last_applied_action`) |
| `speed_limits`             | 2   | episode-conditioned `[linear, angular]` speed limits (always enabled for hovering) |

Actor total per step: **23**.

### Critic (`critic` ObsGroup, inherits from `policy`)

All terms above **plus**:

| term                       | dim | notes                                                       |
|----------------------------|-----|-------------------------------------------------------------|
| `base_lin_vel_b`           | 3   | privileged — true body-frame linear velocity from sim       |

Critic total per step: **26**.

### Routing through SKRL

The IsaacLab→SKRL wrapper surfaces `obs["policy"]` to the actor and
`obs["critic"]` to the value head via the `STATES` token. In
`agents/skrl_ppo_cfg.yaml`:

- Policy network input: `OBSERVATIONS`
- Value network input: `STATES`
- A matching `state_preprocessor: RunningStandardScaler` is configured so
  the privileged obs are normalised independently of the actor obs.

`enable_corruption` is forced `False` on the critic group — any future
observation-noise toggle on the actor (deployment robustness) must not bleed
into the value-baseline path.

---

## 4. Reward terms

Reward functions live in the shared `mdp/rewards.py` (the waypoint-following section) and
`mdp/shared_features.py`. Each callable returns the **unweighted factor** `[N]`; the
`RewardManager` multiplies it by the `weight` configured in `WaypointRewardWeights`
(`hovering_quadcopter_env_cfg.py`) and by `step_dt` — uniformly, for every term. There is no
per-term manual `× env.step_dt` inside the reward functions themselves, and no
`scale_dataclass_attributes` decorator on `WaypointRewardWeights`: the weights below are the
direct `RewTerm` weights (SKRL's `rewards_shaper_scale` is not set for hovering, so no extra
scaling is applied).

14 terms are registered on `RewardsCfg`:

| term                     | weight  | function                                    | shape                                                              |
|--------------------------|---------|----------------------------------------------|---------------------------------------------------------------------|
| `progress_to_waypoint`   | +0.4    | `reward_progress_to_waypoint`                | `tanh(Δdist / progress_to_waypoint_scale_m)` (scale = 0.10 m)       |
| `waypoint_proximity`     | +0.2    | `reward_waypoint_proximity`                  | `1 - tanh(dist / waypoint_proximity_scale_m)` (scale = 1.25 m)      |
| `waypoint_proximity_fine`| +0.14   | `reward_waypoint_proximity_fine`             | `1 - tanh(dist / waypoint_proximity_fine_scale_m)` (scale = 0.30 m) |
| `lin_vel`                | −0.04   | `penalty_lin_vel`                            | `‖v_b‖²`                                                            |
| `ang_vel`                | −0.1    | `penalty_ang_vel`                            | `‖ω_b‖²` (without this the drone learns to circle)                 |
| `near_waypoint_lin_vel`  | −0.12   | `penalty_near_waypoint_lin_vel`              | `‖v_b‖² · exp(-(dist/0.75)²)` — extra braking cost near target      |
| `near_waypoint_ang_vel`  | −0.20   | `penalty_near_waypoint_ang_vel`              | `‖ω_b‖² · exp(-(dist/0.75)²)`                                       |
| `action_magnitude`       | −1.0    | `shared_features.penalty_action_magnitude`   | `‖a_raw‖²`, ×100 multiplier when `max(|a_raw|) > 1` (shared with racing) |
| `action_smoothness`      | −5.0    | `penalty_action_smoothness` (`skip_first_step=False`) | `‖a_t − a_{t-1}‖²`                                         |
| `crash`                  | −2000.0 | `penalty_out_of_bounds_or_upside_down`       | indicator: `z < min_height ∨ z > max_height ∨ upside_down`          |
| `waypoint_collected`     | +300.0  | `reward_waypoint_collected`                  | sparse +1 on the step the active target advances                   |
| `holding`                | +2.0    | `reward_holding`                             | +1 while inside the hold radius (`r_capture`/`r_hold` hysteresis; optional, safe to comment out) |
| `velocity_limit`         | −0.12   | `shared_features.penalty_velocity_limit`     | quadratic excess above the episode's linear speed limit             |
| `angular_velocity_limit` | −0.04   | `shared_features.penalty_angular_velocity_limit` | quadratic excess above the episode's angular speed limit        |

There is no `distance_to_waypoint` or `oversaturation` term (an earlier cubic-distance-penalty /
`_unclamped_actions_buf` design has been replaced by the tanh-shaped progress/proximity terms
above and the shared `penalty_action_magnitude`).

### Notes on weights

- The crash weight is large on purpose: with `episode_length_s = 15`, bumping crash to −2000
  makes "crash to end suffering" strictly worse than chasing the target.
- `WAYPOINT_COLLECTED = +300` and `HOLDING = +2` shape the collection mechanic without
  overwhelming the dense progress/proximity signal.
- `action_magnitude`'s ×100 excess multiplier (via the shared `clip_excess_multiplier` param)
  mirrors the AI-GP experimental / racing `current`-preset action regularization.

---

## 5. Terminations

| term          | time_out | condition                                            |
|---------------|----------|------------------------------------------------------|
| `time_out`    | True     | `episode_length_buf ≥ max_episode_length - 1`        |
| `crashed`     | False    | z < min_height OR z > max_height OR upside_down      |
| `all_collected` | True   | all waypoints visited in order (success terminal)    |

No collision termination (no obstacles in the hovering env).

---

## 6. Events / randomization

### Active terms

| event                    | mode  | params                                                   |
|--------------------------|-------|----------------------------------------------------------|
| `reset_scene_to_default` | reset | —                                                        |
| `reset_waypoint_spawn`   | reset | spawn near waypoint 0 with ±0.3 m XY / ±0.1 m Z jitter   |
| `reset_action_noise`     | reset | shared (`mdp/shared_features.py`) — samples the per-episode action-noise bias/LPF state |
| `randomize_speed_limits` | reset | shared — samples the per-episode `[linear, angular]` speed limits (hovering has `speed_limits.enabled=True` by default) |

### Commented-out DR (kept in source as a starting point)

`randomize_robot_mass`, `randomize_base_com`, `randomize_actuator_gains`,
and `base_external_force_torque` are written in `EventsCfg` but commented
out while the base reward / action setup is being stabilised. The matching
flag fields on `HoveringQuadcopterEnvCfg`
(`randomize_mass`, `randomize_com`, `randomize_actuator_gains`,
`apply_external_disturbances`) and the `_apply_dr_flags()` helper are kept
intact so DR can be re-enabled by uncommenting the block and flipping the
flags.

`push_robot` (velocity-set kick) is intentionally **not** wired in — a
deployed mocap drone is never physically pushed; aerodynamic / mass
mismatch is better modelled by the force/torque interval term and by
action noise/lag.

---

## 7. Waypoint tracks

### Default: per-terrain random tracks with per-terrain T_c

Tracks are generated per **sub-terrain cell**, not per env. A `TerrainGeneratorCfg`
produces an **8×8** grid of flat sub-terrains (`num_rows=8, num_cols=8 → 64
cells`); each cell has:

- its own randomly-generated waypoint sequence, seeded deterministically from
  `(env_cfg.seed, row, col)`;
- its own **per-cell collection time `T_c`**, also fixed per seed.

The 4096 envs wrap modulo 64 onto the cells (mirroring the racing terrain
pattern). On average 64 envs share each cell's track + `T_c` but spawn at
different terrain origins.

Per-cell `T_c` distribution:
- `num_zero_tc_cells: int = 16` cells get `T_c = 0` exactly (explicit
  fly-through behaviour). These are placed first in scan order so they're
  always present regardless of grid size.
- Remaining cells sample `T_c ~ Uniform(collection_time_range_s)` (default
  `(0.2, 2.0)`).
- The drone observes its current cell's `T_c` via the `collection_time_const`
  obs term, so the policy can condition on it (race through vs. hover).

Generator parameters (cfg fields, all tunable):
- `num_waypoints_range: tuple[int, int] = (4, 8)`
- `waypoint_xy_range: float = 3.0` (half-width of the per-cell sampling box)
- `waypoint_z_range: tuple[float, float] = (0.5, 2.5)`
- `min_waypoint_spacing: float = 0.8` (rejection-sampled to avoid clustering)
- `num_zero_tc_cells: int = 16`
- `collection_time_range_s: tuple[float, float] = (0.2, 2.0)`

Random tracks are the **default for both training and the play configs.**

### YAML override

Setting the `WAYPOINTS` env variable to a path replaces the per-terrain
random tracks with a single fixed YAML-loaded track shared across all cells.
The YAML may optionally specify a `collection_time_s` (otherwise the cfg
default `collection_time_s` applies uniformly):

```bash
WAYPOINTS=configs/waypoint_tracks/example.yaml \
    python scripts/rsl_rl/train.py --task Isaac-Hovering-Quadcopter-Manager-v0
```

Format under `configs/waypoint_tracks/`:

```yaml
# configs/waypoint_tracks/example.yaml
collection_time_s: 0.0   # optional; falls back to cfg default
waypoint_track:
  - position: [7.5, 6.0, 2.0]
  - position: [9.5, 7.5, 2.5]
  - position: [7.5, 9.0, 2.0]
  - position: [5.5, 7.5, 2.5]
```

A separate loader `mdp/waypoint_io.py` (NOT `track_io.py`) parses these files
into a `(waypoints, collection_time_s)` pair. The loader rejects any file
containing a `gate_track` key to prevent accidental mixing.

One example YAML (`example.yaml`) is provided as a template; otherwise users
write their own under `configs/waypoint_tracks/`.

---

## 8. Visualization

Red sphere markers at every waypoint of every cell, published **once** at
env init via `VisualizationMarkers` with a custom `SphereCfg`
(radius = 0.1 m, red `PreviewSurfaceCfg`). Total: ~64 cells × ~6 waypoints
≈ 400 markers. Positions are static (tracks don't move during training) so
no per-step update is needed. Active-waypoint highlighting / collected-fade
were dropped to keep `vis.py` minimal — visual inspection of which
waypoint is currently active happens via the drone's own pose against the
fixed red dots, which is enough for the user's validation workflow.

Toggle: `HoveringQuadcopterEnvCfg.draw_waypoints: bool = True`.

Implementation lives in `vis.py` next to the env (mirroring the racing
`trajectory_vis.py` placement but minimal).

---

## 9. Registered env IDs

In `hovering_quadcopter/__init__.py`:

- `Isaac-Hovering-Quadcopter-Manager-v0` → `HoveringQuadcopterEnvCfg`
- `Isaac-Hovering-Quadcopter-Manager-Play-v0` → `HoveringQuadcopterEnvCfg_PLAY`
- `Isaac-Hovering-Quadcopter-Manager-RNN-v0` → `HoveringQuadcopterEnvCfg` (recurrent agent config)
- `Isaac-Hovering-Quadcopter-Manager-RNN-Play-v0` → `HoveringQuadcopterEnvCfg_PLAY` (recurrent agent config)

Each env points at a **hovering-specific `agents/` folder** inside the
package, with its own `rsl_rl_ppo_cfg.py` (`HoveringQuadcopterPPORunnerCfg` /
`HoveringQuadcopterRecurrentPPORunnerCfg` for the RNN ids, experiment names
`hovering_quadcopter_manager` / `hovering_quadcopter_manager_gru`) and `skrl_ppo_cfg.yaml`.
Initial hyperparameters are copied from the racing agent configs; the
experiment name change is what makes logs land in a separate directory.

---

## 10. Metrics (minimal)

Logged in `extras["log"]` on episode reset:

- `Metrics/waypoints_collected` — count of waypoints collected this episode
  (also the primary success indicator)
- `Metrics/episode_length` — useful as a denominator for the above

Per-reward-term episode sums are emitted automatically by the
`RewardManager`, so no extra book-keeping is needed there.

---

## 11. Defaults / hyperparameters at a glance

| field                                | value                                     |
|--------------------------------------|-------------------------------------------|
| `decimation`                         | 2                                         |
| `sim.dt`                             | 1/50 s                                    |
| policy rate                          | 25 Hz                                     |
| `episode_length_s` (train)           | 15.0                                      |
| `episode_length_s` (play)            | 30.0                                      |
| `num_envs`                           | 4096 / 1024 (play)                        |
| `env_spacing`                        | 5.0 m                                     |
| terrain grid                         | 8 × 8 (64 cells), 15 × 15 m each          |
| `r_capture`                          | 0.20 m                                    |
| `r_hold`                             | 0.50 m                                    |
| `collection_time_s` (YAML fallback)  | 1.0                                       |
| `collection_time_range_s`            | (0.2, 2.0)                                |
| `num_zero_tc_cells`                  | 16                                        |
| `min_height`                         | 0.10 m                                    |
| `max_height`                         | 5.0 m                                     |
| `upside_down_threshold`              | 0.7                                       |
| action `thrust_to_weight` (override) | 1.9                                       |
| action `rate_command_scale` (override)| 6.5 rad/s                                |
| drone profile (default)              | `anemos_urdf` (`HOVERING_DRONE_MODEL` env) |

# `hovering_quadcopter/` — Waypoint-Following Env

Manager-based IsaacLab task: fly a quadcopter through a sequence of 3D waypoints,
collecting each by dwelling near it for a per-waypoint time `T_c`. Sibling package
to `racing_quadcopter`, built for **sim-to-real deployment against a motion-capture
setup** that exposes only position + orientation — hence the asymmetric actor/critic
observation split (see below). Full design spec: `docs/hovering_quadcopter/SPEC.md`.

## File map

| File | What it is |
|---|---|
| `hovering_quadcopter_env_cfg.py` | Main config: scene, actions (reuses racing's `QuadcopterRatesActionCfg` + anemos `DRONE_PROFILES`), observations, rewards, events, terminations. **Start here.** |
| `hovering_quadcopter_env.py` | `HoveringQuadcopterRLEnv` — allocates per-env waypoint/hysteresis buffers, drives `update_waypoint_state()` each step. |
| `terrain.py` | 8×8 grid of flat sub-terrains; per-cell random waypoint track generation (`generate_random_tracks`) and YAML-track broadcasting (`broadcast_yaml_track`). |
| `vis.py` | Static red-sphere markers at every waypoint, published once at init. |
| `mdp/waypoint_utils.py` | The two-radius proximity hysteresis state machine (`update_waypoint_state`) — idle → holding → collected. |
| `mdp/waypoint_io.py` | YAML loader for fixed waypoint tracks (`configs/waypoint_tracks/*.yaml`). **Different schema** from both racing track formats — top-level key is `waypoint_track`, not `gate_track`; rejects files that look like a racing track. |
| `mdp/observations.py` | Actor obs (mocap+IMU-derivable only) vs critic obs (adds privileged `base_lin_vel`). |
| `mdp/rewards.py` | Mirrors the old direct-env reward set + two waypoint-task extensions (`reward_waypoint_collected`, `reward_holding`). |
| `mdp/terminations.py` | `time_out`, `crashed` (out of flight envelope), `all_collected`. |
| `mdp/events.py` | Reset/spawn + episode-metric snapshotting. DR terms (mass/CoM/actuator-gain/force-torque) live in `EventsCfg` directly, gated by flags — see below. |
| `mdp/__init__.py` | Re-exports + `scale_dataclass_attributes` decorator (×100 reward weights, matches racing's SKRL scaling convention). |
| `agents/skrl_ppo_cfg.yaml` | SKRL PPO — the actively-used trainer. Asymmetric actor/critic: value net reads the privileged `STATES` (critic) obs group. |
| `agents/rsl_rl_ppo_cfg.py` | RSL-RL PPO — registered but **see caveat below**, reward scale doesn't line up. |
| `__init__.py` | `gym.register()` for both task IDs. |

## Registered tasks

| Task ID | Config | Use |
|---|---|---|
| `Isaac-Hovering-Quadcopter-Manager-v0` | `HoveringQuadcopterEnvCfg` | Training |
| `Isaac-Hovering-Quadcopter-Manager-Play-v0` | `HoveringQuadcopterEnvCfg_PLAY` | Eval/play — `num_envs=1024`, `episode_length_s=30`, `eval_mode=True` |

Registered from `tasks/manager_based/__init__.py` (`from . import hovering_quadcopter`) alongside `racing_quadcopter` — if `Isaac-Hovering-Quadcopter-Manager-v0` isn't found by `gym.make`, that import is the first thing to check.

## Training

```bash
python scripts/skrl/train.py \
    --task Isaac-Hovering-Quadcopter-Manager-v0 \
    --num_envs 4096 --headless
```

Logs land in `logs/skrl/hovering_quadcopter/<timestamp>_ppo_torch/` (directory name
set by `agents/skrl_ppo_cfg.yaml`'s `agent.experiment.directory`). Watch training with:

```bash
./tensorboard_hovering.sh   # wraps: tensorboard --logdir=logs/skrl/hovering_quadcopter
```

**Domain randomization** (mass ±10%, CoM ±15mm, actuator gains ±10%, external
force/torque) is defined in `EventsCfg` but **off by default** — flip the flags
per-run:

```bash
python scripts/skrl/train.py --task Isaac-Hovering-Quadcopter-Manager-v0 --headless \
    --override randomize_mass=True --override apply_external_disturbances=True
```

**Fixed waypoint track** instead of the default per-cell random generation:

```bash
WAYPOINTS=configs/waypoint_tracks/example.yaml \
python scripts/skrl/train.py --task Isaac-Hovering-Quadcopter-Manager-v0 --headless
```

**Drone model** (default `anemos_usd`; deliberately a separate env var from
racing's `DRONE_MODEL` so the two tasks can run side-by-side with different
drones):

```bash
HOVERING_DRONE_MODEL=anemos_urdf python scripts/skrl/train.py --task Isaac-Hovering-Quadcopter-Manager-v0 --headless
```

### RSL-RL — works, but check the reward scale first

`Isaac-Hovering-Quadcopter-Manager-v0` also registers an `rsl_rl_cfg_entry_point`
(`agents/rsl_rl_ppo_cfg.py`), so this runs without error:

```bash
python scripts/rsl_rl/train.py --task Isaac-Hovering-Quadcopter-Manager-v0 --num_envs 4096 --headless
```

but `WaypointRewardWeights` is scaled ×100 in `hovering_quadcopter_env_cfg.py`
specifically so SKRL's `rewards_shaper_scale: 0.01` (set in `agents/skrl_ppo_cfg.yaml`)
divides it back down to the documented magnitudes. RSL-RL's `RslRlPpoAlgorithmCfg`
has no equivalent knob, so training via RSL-RL sees rewards ~100× the intended size
(e.g. crash ≈ −200,000 instead of −2,000). Not RSL-RL-specific to this package —
same pre-existing gap on the racing side — but worth knowing before trusting a
RSL-RL run's absolute reward numbers.

## Eval / Play

```bash
python scripts/skrl/play.py \
    --task Isaac-Hovering-Quadcopter-Manager-Play-v0 \
    --num_envs 64 \
    --checkpoint logs/skrl/hovering_quadcopter/<run>/checkpoints/best_agent.pt
```

Omit `--checkpoint` to auto-discover the most recent run. Use `--real-time` to
throttle playback to wall-clock speed, `--video` to record. The `-Play-v0` config
sets `eval_mode=True` (affects any eval-gated logic), 1024 envs, and a 30 s episode
so longer tracks have time to complete.

To eval on the same fixed track used in training, pass the same `WAYPOINTS=...`
value; the default is per-cell random generation, so omitting it evaluates on a
*different* random track than whatever the policy trained on (by design — it's
testing generalization, not memorization).

## Coordinate / track conventions

- **Terrain**: 8×8 grid of 15×15 m flat cells (`terrain_num_rows` / `terrain_num_cols` /
  `cell_size` on the env cfg). Waypoints are generated **by the env**, not by the
  `TerrainGenerator` sub-terrain callback — `TerrainGenerator` differentiates the
  seed per cell but doesn't expose row/col to the callback, so per-cell control
  (e.g. forcing `num_zero_tc_cells` cells to `T_c=0`) happens in
  `HoveringQuadcopterRLEnv._setup_waypoint_tracks()` after terrain generation.
- **Random tracks** (default): `terrain.generate_random_tracks()` rejection-samples
  4-8 waypoints per cell inside a cell-centred box, with independent per-cell RNGs
  seeded from `env_cfg.seed` so the whole layout is reproducible. The first
  `num_zero_tc_cells` cells get `T_c=0` (fly-through) so that behaviour is always
  represented in training; the rest sample `T_c` uniformly from
  `collection_time_range_s`.
- **Fixed tracks** (`WAYPOINTS=path.yaml`): same track broadcast to every cell.
  Format is `waypoint_track: [{position: [x,y,z]}, ...]` plus an optional
  `collection_time_s:` — **not** the racing `gate_track:` format; see
  `mdp/waypoint_io.py`.
- **Two-radius hysteresis**: `r_capture` (0.20 m, strict — required to *enter*
  holding) and `r_hold` (0.50 m, lax — required to *stay*). A waypoint collects
  once the drone has spent `T_c` seconds continuously inside `r_hold` after first
  entering `r_capture`. Full state machine in `mdp/waypoint_utils.py`.

## Common tasks

| I want to... | Look at |
|---|---|
| Change reward weights | `WaypointRewardWeights` in `hovering_quadcopter_env_cfg.py` (values are pre-scale; `@scale_dataclass_attributes(100.0)` multiplies them for the SKRL convention) |
| Change waypoint generation (count, spacing, box size) | `HoveringQuadcopterEnvCfg` fields `num_waypoints_range` / `waypoint_xy_range` / `waypoint_z_range` / `min_waypoint_spacing`, consumed by `terrain.generate_random_tracks()` |
| Change hysteresis radii / dwell time | `r_capture` / `r_hold` / `collection_time_s` / `collection_time_range_s` on the env cfg |
| Add/author a fixed waypoint track | `configs/waypoint_tracks/*.yaml`, loaded via `mdp/waypoint_io.load_waypoints_from_yaml()` — see `example.yaml` for the format |
| Enable domain randomization | Flip `randomize_mass` / `randomize_com` / `randomize_actuator_gains` / `apply_external_disturbances` on the env cfg (all default `False`) |
| Change what the actor vs critic can see | `ObservationsCfg.PolicyCfg` / `CriticCfg` in `hovering_quadcopter_env_cfg.py` — keep in mind the actor must stay mocap+IMU-derivable for deployment |
| Understand the collection/hysteresis logic | `mdp/waypoint_utils.py::update_waypoint_state()` |

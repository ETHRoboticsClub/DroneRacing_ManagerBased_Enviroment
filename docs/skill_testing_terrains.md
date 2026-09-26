# Skill-Testing Terrains

Evaluation-only setup that lays out **N independent terrains** side by side, each
running its own fixed circuit and its own per-terrain configuration (track YAML,
spawn regime, …). One rollout measures performance across N qualitatively
different scenarios in parallel.

---

## 1. Entry points

**Gym ID:** `Isaac-Racing-Quadcopter-Manager-SkillEval-v0`
(registered in [tasks/manager_based/racing_quadcopter/__init__.py](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/racing_quadcopter/__init__.py#L67))

Use any standard play / eval script, e.g.
`rsl_rl/play.py --task Isaac-Racing-Quadcopter-Manager-SkillEval-v0`.
Training is **not** supported — the cfg sets `eval_mode=True` and disables
observation/action noise and domain randomization.

| File | Purpose |
|---|---|
| [racing_quadcopter_env_cfg.py](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/racing_quadcopter/racing_quadcopter_env_cfg.py) | `SkillTerrainCfg`, `RacingQuadcopterSkillEvalEnvCfg` — per-terrain settings and the eval cfg |
| [racing_quadcopter_env.py](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/racing_quadcopter/racing_quadcopter_env.py) | env classes (`RacingQuadcopterMultiTrackRLEnv`, `RacingQuadcopterSkillEvalRLEnv`) |
| [mdp/events.py](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/mdp/events.py) | `reset_skill_eval_spawn` and the `_apply_*_override` helpers |
| [terrains/racing_terrain.py](../source/Racing_Quadcopter/Racing_Quadcopter/terrains/racing_terrain.py) | `create_multi_track_terrain` |
| [configs/skill_tracks/](../configs/skill_tracks/) | track YAMLs |

`skill_terrains: list[SkillTerrainCfg]` on the env cfg is the **single source of
truth**: its length sets the number of terrains, and each entry controls both
the circuit built in that terrain and how drones behave there.

---

## 2. Current setup

Four terrains, ordered left → right in the world:

| # | YAML | Spawn flag / other override | Description |
|---|---|---|---|
| 0 | [upside_down_challenge.yaml](../configs/skill_tracks/upside_down_challenge.yaml) | `spawn_upside_down` | 4-gate diamond at 4.5 m. Drones spawn behind gate 0 then get a 180° flip around X. Tests self-righting. |
| 1 | [default_difficult.yaml](../configs/skill_tracks/default_difficult.yaml) | — | 10-gate circuit with split-S and chicane. Standard deterministic spawn. Reference baseline. |
| 2 | [recovery_challenge.yaml](../configs/skill_tracks/recovery_challenge.yaml) | `spawn_recovery` | 8-gate circuit. Drones spawn with random attitude, 2–6 m/s linear velocity, ±3 rad/s angular velocity. Tests robustness to non-zero initial conditions. |
| 3 | [sensor_delay_challenge.yaml](../configs/skill_tracks/sensor_delay_challenge.yaml) | `obs_delay_steps=5` | 7-gate open, fast circuit with altitude changes. The policy observation is delayed by 5 policy steps (5 × `step_dt` = 200 ms at the default 1/50 s `sim.dt` × decimation 2). Tests robustness to sensor/communication latency. |

Other eval defaults set in `__post_init__`: `num_envs=256`, `eval_mode=True`,
`force_collision_termination=True` (collisions terminate from step 0, bypassing
the training-curriculum gate), no observation/action corruption, mass and
actuator-gain randomization off, `stats_warmup_s=30.0`.

---

## 3. Adding a new terrain

**1) Drop a YAML in `configs/skill_tracks/`** with positions in the cell's
**corner-origin** local frame — for the default 15×15 m cell, valid x/y range is
`[0, 15]` (centre at 7.5, 7.5). See [§4](#4-coordinate-convention).

```yaml
gate_track:
  - position: [7.5, 2.5, 1.5]
    rotation: [0, 0, 0]            # [roll, pitch, yaw], degrees
  - position: [11.5, 5.5, 3.0]
    rotation: [0, 0, 60]
  ...
```

**2) (Only if you need new per-terrain behaviour)** add a field to
`SkillTerrainCfg` and read it where it applies. For a new spawn regime: add a
flag, write an `_apply_my_thing_override(env, env_ids)` helper in
`mdp/events.py`, and add a branch in `reset_skill_eval_spawn`. Other settings
(gate aperture, episode length, gate colour, …) are read from
`env.cfg.skill_terrains[idx]` wherever you need them.

```python
# racing_quadcopter_env_cfg.py
@configclass
class SkillTerrainCfg:
    track_yaml: str = ...
    spawn_upside_down: bool = False
    spawn_recovery: bool = False
    spawn_my_new_thing: bool = False           # new field
```

```python
# mdp/events.py — reset_skill_eval_spawn
_apply_fixed_initialization(env, col_ids)
if   getattr(terrain_cfg, "spawn_upside_down", False):
    _apply_upside_down_override(env, col_ids)
elif getattr(terrain_cfg, "spawn_recovery", False):
    _apply_recovery_override(env, col_ids)
elif getattr(terrain_cfg, "spawn_my_new_thing", False):
    _apply_my_new_override(env, col_ids)       # new branch
```

**3) Add a `SkillTerrainCfg(...)` entry to `skill_terrains`** in
`RacingQuadcopterSkillEvalEnvCfg`. Order is positional — the i-th entry becomes
the i-th terrain (left → right in the world). Nothing else needs to change; the
env class, terrain generator, and gym registration all consume the cfg.

---

## 4. Coordinate convention

YAML positions are in the **sub-terrain's corner-origin local frame** — for the
default 15×15 m cell, x/y in `[0, 15]` and the centre at (7.5, 7.5).
The runtime applies a `centre_offset = (-size_x/2, -size_y/2, 0)` to convert to
the centre-origin frame `TerrainGenerator` uses internally.

> ⚠️ **Don't use negative or terrain-centre-relative coordinates.** Values like
> (±2.5, ±2.5) put gates outside the terrain footprint once the corner-origin
> conversion is applied. Stay in `[0, 15]` — see `default_difficult.yaml` for a
> reference layout.

`rotation` is `[roll, pitch, yaw]` in degrees. **Yaw is CW-from-+Y**: 0° → +Y,
90° → +X, 180° → −Y, 270° → −X. The conversion to scipy's CCW-around-Z internal
form happens in `gate_specs_to_gate_data` (it negates yaw).

---

## 5. Things to know about the implementation

- **Per-terrain dispatch is a single event, not N events.**
  `reset_skill_eval_spawn` iterates `skill_terrains`, masks `env_ids` by terrain
  index, applies the base spawn (`_apply_fixed_initialization`), then optionally
  applies a per-flag override. Adding a new behaviour = one new field + one new
  branch.

- **Overrides compose on top of the base spawn.** They run *after*
  `_apply_fixed_initialization` and read `_prev_drone_pos` for the position; they
  only change attitude and velocity. If you write a new override that needs a
  different position, read it from `_prev_drone_pos` and offset — don't replace
  the base spawn call.

- **env → terrain mapping uses `scene.env_origins`, not the row-major formula.**
  The single-circuit env's parent uses `(env_id // num_cols) % num_rows` for
  env→cell. That is **wrong** under `curriculum=True` (which the multi-track
  layout uses). The multi-track env's `_load_cell_gate_data` hook resolves
  env→terrain by closest-cell match against `scene.env_origins`, which is
  correct in both modes. The result is stored on the env as `_env_col_idx` so
  the spawn dispatcher can use it.

- **The terrain layout uses `curriculum=True, num_rows=1, num_cols=N`.** This
  guarantees the j-th terrain renders the j-th YAML deterministically and that
  envs are evenly partitioned across terrains. `difficulty_range` is collapsed
  to `(0.5, 0.5)` — `fixed_gates_terrain` ignores difficulty anyway.

- **Gates are read from `sub_terrains.values()`, not the `gate_registry`.** The
  registry's iteration order is not guaranteed to match terrain order. Reading
  from the sub_terrain cfg dict directly preserves insertion order (= terrain
  order, by construction in `create_multi_track_terrain`).

- **Episode metrics are logged per terrain.** `collect_episode_metrics_before_reset`
  emits the global `Episode_Metrics/{gates_passed, mean_velocity_kmh,
  lap_completion_pct, mean_lap_time_s}` and additionally `terrain_{i}/...` for
  each terrain that reset this step (gated on `env._env_col_idx` being present).
  Each terrain becomes its own top-level tensorboard panel; the cross-terrain
  aggregate stays under `Episode_Metrics/`. The console-printed sliding-window
  stats (Speed / Gates / Lap%) remain global.

- **The skill-eval env writes its own tensorboard log.** `RacingQuadcopterRLEnv`
  exposes a hook `_on_episode_metrics_collected(gate_log)` (default no-op) called
  every reset after `gate_log` is merged into `extras["log"]`.
  `RacingQuadcopterSkillEvalRLEnv` overrides it to write every metric scalar to
  its own `SummaryWriter`. This is necessary because `scripts/rsl_rl/play.py`
  initialises the runner with `log_dir=None`, so the runner does not consume
  `extras["log"]` — without this writer, no metrics would reach disk during
  skill eval. The writer log directory is printed at startup; see [§6](#6-viewing-per-terrain-metrics).

---

## 6. Viewing per-terrain metrics

`RacingQuadcopterSkillEvalRLEnv` opens its own `SummaryWriter` and writes every
metric key (global aggregate and per-terrain) to it on each reset. The log
directory is announced at startup, e.g.:

```
==================================================================
[Skill Eval] Per-terrain metrics → tensorboard
  Log dir : /abs/path/to/logs/rsl_rl/<exp>/<run>/skill_eval/2026-05-08_14-32-10
  View    : tensorboard --logdir /abs/path/to/.../skill_eval/2026-05-08_14-32-10
  Look for one panel per terrain (`terrain_{i}/...`) plus the cross-terrain
  aggregate under `Episode_Metrics/...`.
==================================================================
```

Path resolution:
- If `cfg.log_dir` is set (the play scripts set this from the resumed checkpoint
  directory), logs go to `<cfg.log_dir>/skill_eval/<timestamp>/`.
- Otherwise, logs fall back to `<cwd>/logs/skill_eval/<timestamp>/`.

In tensorboard you'll see:
- `terrain_{i}/{gates_passed, mean_velocity_kmh, lap_completion_pct, mean_lap_time_s}`
  — metrics restricted to envs assigned to terrain `i`. Each terrain is its own
  top-level panel, so per-terrain plots live side-by-side.
- `Episode_Metrics/{gates_passed, mean_velocity_kmh, lap_completion_pct, mean_lap_time_s}`
  — the same metrics aggregated across all envs.

Notes on the metrics:
- `gates_passed` and `mean_velocity_kmh` are emitted every reset.
- `lap_completion_pct` mirrors the console "Lap%" formula: each completed lap
  counts as 1.0, each terminated (not timed-out) episode adds a partial run
  value of `(gates_passed % gate_count) / gate_count`, then averaged. Skipped
  when no env in the subset has either a finished lap or a termination this step.
- `mean_lap_time_s` is emitted only when at least one env completed a full lap.

The x-axis is `env.common_step_counter`. Sparse points are normal: a per-terrain
key is only written when at least one env in that terrain reset that step.

Requires `tensorboard` (i.e. `torch.utils.tensorboard`) to be importable. If it
isn't, the env logs a warning at startup and continues without a writer.

---

## 7. Debugging

Set `DEBUG_TERRAIN=1` to print `num_rows`, `num_cols`, and the per-terrain env
count distribution at startup.

| Symptom | Likely cause |
|---|---|
| Wrong track in the j-th terrain | `skill_terrains` order doesn't match expected order, or `curriculum=True` was disabled |
| Spawn override not applied | Flag name on `SkillTerrainCfg` doesn't match the `getattr(...)` lookup in `reset_skill_eval_spawn` |
| `AttributeError: ... '_env_col_idx'` | Using `RacingQuadcopterMultiTrackRLEnv` (no terrain-index tracking) instead of `RacingQuadcopterSkillEvalRLEnv` |
| No tensorboard scalars under `Episode_Metrics/...` | `tensorboard` not installed (check the startup warning), or you ran `RacingQuadcopterMultiTrackRLEnv` instead of `RacingQuadcopterSkillEvalRLEnv` |

# Drone Racing Environment — Complete Codebase Reference

This document covers every file, parameter, and design decision in the codebase.
It is intended for readers familiar with IsaacLab and reinforcement learning.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Running the Scripts](#2-running-the-scripts)
   - 2.1 [train.py](#21-trainpy)
   - 2.2 [play.py](#22-playpy)
   - 2.3 [sweep.py](#23-sweeppy)
   - 2.4 [Common examples](#24-common-examples)
   - 2.5 [RSL-RL entrypoints](#25-rsl-rl-entrypoints)
3. [Task Variants](#3-task-variants)
4. [Environment Architecture](#4-environment-architecture)
5. [Physics Model](#5-physics-model)
6. [MDP Specification](#6-mdp-specification)
   - 6.1 [Observation Space](#61-observation-space)
   - 6.2 [Action Space](#62-action-space)
   - 6.3 [Reward Terms](#63-reward-terms)
   - 6.4 [Termination Conditions](#64-termination-conditions)
   - 6.5 [Episode Reset (Events)](#65-episode-reset-events)
   - 6.6 [Curriculum](#66-curriculum)
7. [Terrain System](#7-terrain-system)
   - 7.1 [Random Track (Procedural)](#71-random-track-procedural)
   - 7.2 [Fixed Track (Deterministic)](#72-fixed-track-deterministic)
   - 7.3 [Gate Coordinate Convention](#73-gate-coordinate-convention)
   - 7.4 [Gate Registry Pattern](#74-gate-registry-pattern)
   - 7.5 [Gate Passage Detection](#75-gate-passage-detection)
8. [Observation Corruption](#8-observation-corruption)
9. [Action Corruption](#9-action-corruption)
10. [Statistics and Logging](#10-statistics-and-logging)
11. [Training Setup](#11-training-setup)
    - 11.1 [SKRL PPO Configuration](#111-skrl-ppo-configuration)
    - 11.2 [RSL-RL PPO Configuration](#112-rsl-rl-ppo-configuration)
    - 11.3 [Reward Scaling Chain](#113-reward-scaling-chain)
12. [Evaluation Setup](#12-evaluation-setup)
13. [Sweep System](#13-sweep-system)
    - 13.1 [How It Works](#131-how-it-works)
    - 13.2 [Configuration via YAML](#132-configuration-via-yaml)
    - 13.3 [Sweep Modes](#133-sweep-modes)
    - 13.4 [PARAM_GRID Reference](#134-param_grid-reference)
    - 13.5 [DIRECT_ Shorthand](#135-direct_-shorthand)
    - 13.6 [Output and Metrics](#136-output-and-metrics)
14. [Override Utility](#14-override-utility)
15. [Environment Configuration Reference](#15-environment-configuration-reference)
16. [Visualization](#16-visualization)
17. [Demo Mode](#17-demo-mode)
18. [Debug Flags](#18-debug-flags)
19. [Known Limitations and Design Notes](#19-known-limitations-and-design-notes)

---

## 1. Repository Structure

```
DroneRacing_ManagerBased_Enviroment/
├── scripts/
│   ├── skrl/
│   │   ├── train.py             # SKRL training entrypoint
│   │   ├── play.py              # SKRL evaluation / playback entrypoint
│   │   └── override_utils.py    # Shared --override CLI helper
│   ├── rsl_rl/
│   │   ├── train.py / play.py   # RSL-RL entrypoints (Hydra overrides, no --override; see 2.5)
│   │   └── cli_args.py          # RSL-RL flags (--experiment_name, --resume, --load_run, --logger, ...)
│   ├── sweep.py                 # Hyperparameter grid-search orchestrator
│   ├── list_envs.py / random_agent.py / zero_agent.py
│   ├── custom_track/            # Qt track designer, play a checkpoint on a custom track
│   ├── deployment/              # Replay recorded real-flight actions in sim
│   └── tools/                   # generate_anemos_usd.py, view_drone.py, ...
│
├── source/
│   └── Racing_Quadcopter/
│       └── Racing_Quadcopter/
│           ├── tasks/
│           │   └── manager_based/
│           │       ├── mdp/                             # ONE shared MDP package (racing + hovering; AI-GP snapshots have their own copies)
│           │       │   ├── __init__.py                  # Re-exports isaaclab.envs.mdp + all modules below
│           │       │   ├── actions.py                   # Rates / moments / SITL action terms
│           │       │   ├── observations.py              # Gate-racing + waypoint observations
│           │       │   ├── rewards.py                   # Gate-racing + waypoint reward/penalty functions
│           │       │   ├── terminations.py              # Gate-racing + waypoint termination conditions
│           │       │   ├── events.py                    # Spawns, reset metrics, mass/COM/inertia/actuator randomization
│           │       │   ├── curriculums.py               # Staged curriculum + collision gating
│           │       │   ├── adaptive_curriculums.py      # Route-progress reward/randomization curricula
│           │       │   ├── shared_features.py           # Action noise + episode speed limits
│           │       │   ├── noise.py                     # Observation noise models (per-term, opt-in)
│           │       │   ├── route_progress.py            # Normalized route progress
│           │       │   ├── gate_utils.py / stats.py     # Gate-racing state update + statistics
│           │       │   └── waypoint_utils.py / waypoint_io.py  # Waypoint state + YAML loader
│           │       ├── drone_profile.py                 # DroneProfile schema (values live per task)
│           │       ├── action_diagnostics.py            # Action statistics logging
│           │       ├── ai_gp_quadcopter/                # AI-GP task: FROZEN june11/ + experimental/ snapshots (self-contained; see its README)
│           │       ├── racing_quadcopter/
│           │       │   ├── __init__.py                  # gym.register (racing)
│           │       │   ├── racing_quadcopter_env.py     # RacingQuadcopterRLEnv, _MultiTrackRLEnv, _SkillEvalRLEnv
│           │       │   ├── racing_quadcopter_env_cfg.py # All racing config: scene/sim/managers, drone profiles, reward presets, SkillEval cfg
│           │       │   ├── trajectory_vis.py            # USD BasisCurves visualization
│           │       │   └── agents/                      # skrl YAML, rsl_rl runners
│           │       └── hovering_quadcopter/             # Hovering env, cfg, agents, terrain (imports nothing from racing)
│           └── terrains/
│               ├── __init__.py
│               ├── gate_base.py          # GateData, gate_registry, create_gate_frame
│               ├── racing_gates.py       # Procedural track generator
│               ├── racing_gates_cfg.py   # MeshRacingGatesTerrainCfg, MeshFixedGatesTerrainCfg
│               ├── racing_terrain.py     # create_racing_gates_terrain, create_fixed_track_terrain
│               ├── fixed_gates.py        # Fixed circuit terrain function
│               └── track_io.py           # load_gates_from_yaml (track YAML → GateSpec)
│
└── docs/
    ├── CODEBASE.md                  # This document
    └── README_randomization.md      # Noise/randomization parameter guide
```

---

## 2. Running the Scripts

### 2.1 train.py

Trains a new policy (or resumes from a checkpoint).

```bash
python scripts/skrl/train.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-v0 \
    --num_envs 4096 \
    --headless
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--task TASK` | required | Gymnasium task ID (see [Section 3](#3-task-variants)) |
| `--num_envs N` | cfg default | Number of parallel environments |
| `--headless` | off | Disable the GUI window |
| `--max_iterations N` | cfg default (timesteps) | Total PPO update iterations; overrides `trainer.timesteps` as `N × rollouts` |
| `--checkpoint PATH` | None | Resume training from this checkpoint file |
| `--seed N` | agent cfg default | RNG seed; `-1` picks a random seed |
| `--distributed` | off | Multi-GPU training (requires CUDA) |
| `--algorithm ALG` | `PPO` | RL algorithm: `PPO`, `AMP`, `IPPO`, `MAPPO` |
| `--ml_framework FW` | `torch` | Backend: `torch`, `jax`, `jax-numpy` |
| `--exp-name NAME` | auto timestamp | Override the run's experiment directory name (used by sweep.py) |
| `--video` | off | Record videos every `--video_interval` steps |
| `--video_length N` | 200 | Steps per recorded video |
| `--video_interval N` | 2000 | Steps between video recordings |
| `--override KEY=VAL` | none | Override any `env_cfg` attribute; repeatable (see [Section 14](#14-override-utility)) |
| `--agent NAME` | None | Agent config entry point to use; if unset, derived from `--algorithm` |
| `--export_io_descriptors` | off | Export the env's observation/action IO descriptors |
| `--ray-proc-id`, `-rid N` | None | Set automatically by the Isaac Lab Ray integration |

Logs land in `logs/skrl/racing_quadcopter/YYYY-MM-DD_HH-MM-SS_ppo_torch/` (the directory name comes from `agent.experiment.directory` in `skrl_ppo_cfg.yaml`, not the gym task id).

### 2.2 play.py

Evaluates a saved checkpoint.

```bash
python scripts/skrl/play.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0 \
    --num_envs 64 \
    --checkpoint logs/skrl/racing_quadcopter/<run>/checkpoints/best_agent.pt
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--task TASK` | required | Gymnasium task ID; use a `*-Play-*` variant for eval configs |
| `--num_envs N` | cfg default | Number of parallel environments |
| `--headless` | off | Disable the GUI window |
| `--checkpoint PATH` | auto-discover | Path to `.pt` checkpoint; if omitted, finds the most recent run |
| `--use_pretrained_checkpoint` | off | Download a pre-trained checkpoint from Nucleus (if available) |
| `--seed N` | agent cfg default | RNG seed; `-1` picks a random seed |
| `--algorithm ALG` | `PPO` | Must match the algorithm used during training |
| `--ml_framework FW` | `torch` | Must match the ML framework used during training |
| `--real-time` | off | Throttle steps to match `step_dt` (0.04 s per step) in wall time |
| `--eval-steps N` | 0 | Stop after N environment steps and print a `[SWEEP_RESULT]` line; 0 = run until window close |
| `--video` | off | Record a single video of `--video_length` steps |
| `--video_length N` | 200 | Steps in the recorded video |
| `--override KEY=VAL` | none | Override any `env_cfg` attribute; repeatable |
| `--disable_fabric` | off | Force USD I/O instead of Fabric for debugging |
| `--agent NAME` | None | Agent config entry point to use; if unset, derived from `--algorithm` |

**Checkpoint auto-discovery:** if `--checkpoint` is not given, play.py searches `logs/skrl/racing_quadcopter/` for the most recent directory matching `*_ppo_torch` and loads `best_agent.pt` from its `checkpoints/` subdirectory.

### 2.3 sweep.py

Runs a hyperparameter sweep: trains one policy per parameter combination, evaluates each, and writes results to CSV.

```bash
# Recommended: write a YAML config and pass it with --config
python scripts/sweep.py --config path/to/my_sweep.yaml

# Fallback: run with defaults baked into sweep.py (single-combination smoke test)
python scripts/sweep.py
```

Results are written to `logs/sweep/results_<timestamp>.csv`.
See [Section 13](#13-sweep-system) for the full sweep reference including YAML format and sweep modes.

### 2.4 Common examples

```bash
# Train with a custom episode length (observation noise is opt-in per term via
# mdp/noise.py and is off by default; see docs/README_randomization.md)
python scripts/skrl/train.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-v0 \
    --num_envs 4096 --headless \
    --override episode_length_s=30.0

# Resume training from a checkpoint
python scripts/skrl/train.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-v0 \
    --num_envs 4096 --headless \
    --checkpoint logs/skrl/racing_quadcopter/<run>/checkpoints/best_agent.pt

# Evaluate with real-time pacing and demo visuals
python scripts/skrl/play.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0 \
    --num_envs 8 \
    --checkpoint <path> \
    --real-time \
    --override demo_mode=True

# Debug: 1 env, full terrain and reward logging
DEBUG_TERRAIN=1 DEBUG_REWARDS=1 \
python scripts/skrl/play.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0 \
    --num_envs 1 --headless \
    --checkpoint <path>
```

### 2.5 RSL-RL entrypoints

`scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` run the same tasks with the RSL-RL runner
configs (`agents/rsl_rl_ppo_cfg.py`, see [Section 11.2](#112-rsl-rl-ppo-configuration)). They share
`--task`, `--num_envs`, `--seed`, `--video*`, `--agent` and (train only) `--max_iterations` /
`--export_io_descriptors` with the skrl scripts, and add the flags from `scripts/rsl_rl/cli_args.py`:

| Flag | Description |
|------|-------------|
| `--experiment_name NAME` | Log folder under `logs/rsl_rl/` (default: the runner's `experiment_name`) |
| `--run_name NAME` | Suffix for the run directory |
| `--resume` | Resume training from a checkpoint |
| `--load_run RUN` / `--checkpoint FILE` | Which run folder / checkpoint file to load (play and resume) |
| `--logger {tensorboard,wandb,neptune}` | Logger backend |
| `--log_project_name NAME` | Project name for wandb / neptune |

There is no `--override`; config values are changed with Hydra arguments instead (e.g.
`env.episode_length_s=30`). The same limitation applies: fields consumed in `__post_init__` cannot
be changed this way.

---

## 3. Task Variants

Ten racing gymnasium tasks are registered in `racing_quadcopter/__init__.py` (the hovering and
AI-GP tasks are registered by their own packages):

| Task ID | Config Class | Use |
|---------|-------------|-----|
| `Isaac-Racing-Quadcopter-Manager-v0` | `RacingQuadcopterEnvCfg` | Train on random procedural tracks |
| `Isaac-Racing-Quadcopter-Manager-Play-v0` | `RacingQuadcopterEnvCfg_PLAY` | Evaluate random-track policy |
| `Isaac-Racing-Quadcopter-Manager-Fixed-v0` | `RacingQuadcopterEnvCfg_FIXED` | Train on fixed deterministic circuit |
| `Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0` | `RacingQuadcopterEnvCfg_FIXED_PLAY` | Evaluate fixed-track policy |
| `Isaac-Racing-Quadcopter-Manager-Fixed-Play-FPV-v0` | `RacingQuadcopterEnvCfg_FIXED_PLAY_FPV` | Single-drone FPV demo/eval on the fixed circuit |
| `Isaac-Racing-Quadcopter-Manager-SkillEval-v0` | `RacingQuadcopterSkillEvalEnvCfg` | Multi-terrain skill evaluation (see `docs/skill_testing_terrains.md`) |
| `Isaac-Racing-Quadcopter-Manager-RNN-v0` / `-RNN-Play-v0` | `RacingQuadcopterEnvCfg` / `_PLAY` | Same as above, recurrent (GRU) agent config |
| `Isaac-Racing-Quadcopter-Manager-Fixed-RNN-v0` / `-Fixed-RNN-Play-v0` | `RacingQuadcopterEnvCfg_FIXED` / `_FIXED_PLAY` | Fixed-circuit variants, recurrent agent config |

**Training vs Play differences** (`RacingQuadcopterEnvCfg_PLAY` / `_FIXED_PLAY` only override these three fields; every domain-randomization and noise flag already defaults to off in both, so play does not need to disable them):

| Parameter | Training | Play |
|-----------|----------|------|
| `eval_mode` | `False` | `True` |
| `fixed_spawn` | `False` | `True` |
| `stats_warmup_s` | `0.0` | `30.0` |
| `scene.num_envs` | 4096 | 1024 |
| `randomize_mass` / `randomize_com` / `randomize_inertia` / `randomize_actuator_gains` / `apply_external_disturbances` | `False` (default; opt-in via `--override`) | `False` (unchanged) |
| `action_noise.enable` | `False` (default; opt-in) | `False` (unchanged) |
| Curriculum | advances | frozen at stage 1000 |

The Play configs set `eval_mode=True`, which:
- Freezes `_curriculum_stage` at 1000 (all collision terminations active, no upside-down exit, no recovery spawns)
- Enables the `lap_complete` termination (episode ends once every gate has been passed once)
- Suppresses recovery spawns even when `recovery_spawn_prob > 0`

---

## 4. Environment Architecture

The environment is `RacingQuadcopterRLEnv`, a subclass of `ManagerBasedRLEnv`.

### Manager hierarchy

IsaacLab's `ManagerBasedRLEnv` delegates every MDP component to a manager:

| Manager | Config class | Description |
|---------|-------------|-------------|
| `ActionManager` | `ActionsCfg` | `QuadcopterRatesAction` (default, `ACTION_TYPE=rates`) or `QuadcopterMomentsAction` (`ACTION_TYPE=moments`) |
| `ObservationManager` | `ObservationsCfg` | Sensor + gate + noise |
| `RewardManager` | `RewardsCfg` | 13 reward/penalty terms |
| `TerminationManager` | `TerminationsCfg` | 6 termination conditions |
| `EventManager` | `EventsCfg` | Mass/COM/inertia/actuator randomization, disturbances, spawn |
| `CurriculumManager` | `CurriculumCfg` | Staged progression + optional adaptive route-progress curricula |

### Initialization sequence

`load_managers()` is overridden to call `_prepare_racing_task()` **before** `super().load_managers()`. This is mandatory because reward, observation, and termination functions reference buffers (`_gate_positions`, `_actions_buf`, etc.) during manager construction to infer output tensor shapes.

```python
def load_managers(self):
    self._prepare_racing_task()   # allocate GPU buffers first
    super().load_managers()       # then build managers (they read buffers)
    init_two_stage_curriculum(self)
```

### Custom step loop

`RacingQuadcopterRLEnv.step()` overrides the default to inject:
1. `update_racing_state(self)` (`mdp/gate_utils.py`) — gate passage detection + crash buffer refresh, called once after the decimation loop completes
2. `step_stats(self)` (`mdp/stats.py`) — periodic console logging
3. `_prev_drone_pos` update — deferred until **after** reward computation so `reward_gate_progress` sees the correct prior distance
4. Dual observation computation: one pre-reset pass for the recorder (no history update), one post-reset pass for the actual returned observation (with history update)

### Key GPU buffers

All buffers are allocated in `_prepare_racing_task()` on `self.device`.

| Buffer | Shape | Description |
|--------|-------|-------------|
| `_gate_positions` | `[N, max_gates, 3]` | World-frame gate centres, padded |
| `_gate_orientations` | `[N, max_gates, 4]` | Gate quaternions wxyz, padded |
| `_gate_counts` | `[N]` long | True gate count per env (unpadded) |
| `_gate_sizes` | `[N]` | Gate aperture in metres per env |
| `_current_gate_idx` | `[N]` long | Index of next gate to pass |
| `_spawn_gate_idx` | `[N]` long | Gate the env spawned in front of |
| `_gates_passed` | `[N]` long | Cumulative passages this episode |
| `_prev_drone_pos` | `[N, 3]` | Previous step world position |
| `_gate_passed_buf` | `[N]` float | 1.0 on passage step, else 0.0 |
| `_crashed_buf` | `[N]` bool | Contact force OR upside-down |
| `_collision_termination_buf` | `[N]` bool | Contact force only (gated by curriculum) |
| `_collision_detection_enabled` | `[N]` bool | Per-env curriculum gate |
| `_velocity_limit` | `[N]` | Per-env linear speed cap (m/s) |
| `_angular_velocity_limit` | `[N]` | Per-env angular speed cap (rad/s) |
| `_speed_limit_randomization_active` | bool | One-way curriculum latch for per-episode sampling |
| `_actions_buf` | `[N, 4]` | Current step actions |
| `_prev_actions` | `[N, 4]` | Previous step actions |
| `_action_noise_bias` | `[N, 4]` | Per-episode action bias |

---

## 5. Physics Model

### Simulator

- `dt = 1/50 s` (50 Hz physics)
- `decimation = 2` — two physics steps per RL step → `step_dt = 2/50 = 0.04 s`
- `render_interval = decimation` — render every 2nd physics step when a GUI is open

### Robot

Selected by the `DRONE_MODEL` env var (`crazyflie` default, or `anemos_usd` / `anemos_urdf` / `anemos_sitl`); each entry in `DRONE_PROFILES` (`racing_quadcopter_env_cfg.py`) is loaded with `activate_contact_sensors=True` so all links contribute to the `ContactSensor`.

**Contact sensor** settings:
- `history_length = 2` — keeps two steps of contact force history
- `update_period = 0.01 s` — shorter than the 0.02 s physics step (`sim.dt = 1/50`), so the sensor refreshes once per physics step (every 20 ms)

### Thrust model (`mdp/actions.py`)

The drone is driven through `permanent_wrench_composer` (persistent force/torque applied to
the body link each step). Two action terms exist, selected by `ACTION_TYPE`:

- **`QuadcopterRatesAction`** (default, `ACTION_TYPE=rates`): `a0` maps to collective thrust;
  `[a1, a2, a3]` are desired body rates (`× rate_command_scale` rad/s) tracked by an inner-loop
  Betaflight-style PIDF controller (`τ = Kp·e + Ki·∫e − Kd·ω̇_meas + Kf·Δω_des/dt`) that outputs
  body moments. Gains and `rate_command_scale` come from the active `DroneProfile`.
- **`QuadcopterMomentsAction`** (`ACTION_TYPE=moments`): `a0` maps to collective thrust,
  `[a1, a2, a3]` map directly to body moments via `moment_scale`, with no inner PID loop.

```
thrust = thrust_to_weight × robot_weight × (a0 + 1) / 2    (N, body +Z)
```

- `thrust_to_weight` and (for the moments action) `moment_scale` come from the selected drone
  profile (e.g. crazyflie: 1.9; anemos: ~4.1).
- Robot weight is computed once at startup: `sum(masses) × |gravity|`.

**Action range:** all actions clipped to `[-1, 1]` before thrust/moment mapping. The normalized thrust command is `(a0 + 1) / 2`, mapping `[-1, 1] → [0, 1]`.

### Startup physics randomization

All default to `False`/disabled on both `RacingQuadcopterEnvCfg` and its Play subclasses; set
the corresponding top-level flag (e.g. `--override randomize_mass=True`) to enable:

| Event | Mode | Gated by flag | Parameters |
|-------|------|----------------|-----------|
| `randomize_robot_mass` | startup | `randomize_mass` | Scale in `[0.9, 1.1]`, uniform, recomputes inertia |
| `randomize_base_com` | startup | `randomize_com` | Shift `±15 mm` per axis |
| `randomize_robot_inertia` | startup | `randomize_inertia` | Scale principal inertias `[0.8, 1.2]`, uniform (runs after mass so it isn't overwritten by `recompute_inertia`) |
| `randomize_robot_actuator_gains` | startup | `randomize_actuator_gains` | Stiffness + damping scale `[0.9, 1.1]` |
| `base_external_force_torque` | interval (3-10 s) | `apply_external_disturbances` | Force/torque in `±0.02` N / N·m |
| `push_robot` | interval (3-10 s) | `apply_external_disturbances` | Velocity kick in `±0.3` m/s per axis |

---

## 6. MDP Specification

### 6.1 Observation Space

**Policy group** — concatenated, with the current default history window of 1.

| Term | Dimension (per step) | Function | Description |
|------|----------------------|----------|-------------|
| `base_lin_vel` | 3 | `base_lin_vel` (stock IsaacLab `isaaclab.envs.mdp.observations`) | Body-frame linear velocity (m/s) |
| `base_ang_vel` | 3 | `base_ang_vel` (stock IsaacLab) | Body-frame angular velocity (rad/s) |
| `projected_gravity` | 3 | `projected_gravity` (stock IsaacLab) | Gravity direction in body frame |
| `last_action` | 4 | `last_action` (stock IsaacLab) | Previous step action `[a0..a3]` |
| `obs_gates` | K×7 | `compute_next_gates_rel_and_fwd_body` | Next K gates: relative position, forward, aperture half-width |
| `speed_limits` | 0 or 2 | `speed_limits_obs` | Disabled: no columns; enabled: linear and angular episode limits |

`mdp/observations.py` does `from isaaclab.envs.mdp.observations import *`, so the first four terms
are the stock IsaacLab functions re-exported, not custom wrappers.

With default `K=3`, the base observation has 34 values. Enabling paired speed
limits appends two values for a total of 36.

**Gate observation details** (`compute_next_gates_rel_and_fwd_body`):

For each of the next K gates:
- `rel_pos [3]` — gate centre relative to drone, in drone **body frame**, divided by `_REL_POSITION_SCALE=10.0` to keep values near `[-1, 1]` for gates up to 10 m away
- `fwd [3]` — gate forward direction (+Y in gate frame), rotated to drone body frame (unit vector)
- `half_width [1]` — effective gate aperture half-width

All K gates are gathered simultaneously using vectorized `torch.gather` over the padded gate table with modular indexing.

**`enable_corruption = True`** on `PolicyCfg`: the IsaacLab manager-level corruption pipeline is active, but each term's `noise=` model (`mdp/noise.py::drone_noise_model_factory`) defaults to `enable=False`, so no noise is actually applied unless a term explicitly enables it or the adaptive-randomization curriculum turns it on (see [Section 8](#8-observation-corruption)).

**`GateObservationConfig`:** `num_next_gates: int = 3` controls lookahead depth K.
The paired limit observations are added only when `speed_limits.enabled=True`; this
changes the policy input by two values and is incompatible with older checkpoints.

**Alternative layout — `OBSERVATION_SPACE=mono_race`:** the table above is the default `ours`
layout. `OBSERVATION_SPACE=mono_race` selects `MonoRaceObservationsCfg` instead: the next gate's
position and velocity in the gate frame (6), gate orientations (3), the next two gates' poses in
the current gate's frame (12), `base_ang_vel` (3) and motor speeds (4) — 28 values, 30 with speed
limits. See [OBSERVATION_SPACE.md](../source/Racing_Quadcopter/docs/OBSERVATION_SPACE.md).

### 6.2 Action Space

Continuous 4-dimensional, each component in `[-1, 1]`. Interpretation of `[a1, a2, a3]` depends on
`ACTION_TYPE`:

- **`ACTION_TYPE=rates`** (default): `a[0]` — collective thrust (mapped to `[0, 1]` via
  `(a+1)/2`); `[a1, a2, a3]` — desired body roll/pitch/yaw **rates** (`× rate_command_scale`
  rad/s), tracked by an inner-loop PID that outputs the actual body moments.
- **`ACTION_TYPE=moments`**: `a[0]` — collective thrust (same mapping); `[a1, a2, a3]` — body
  roll/pitch/yaw **moments** directly (`× moment_scale`), no inner PID loop.

### 6.3 Reward Terms

All functions return an **unweighted** scalar per environment `[N]`. There is no `MANAGER_SCALE` —
weights configured on `RewTerm` are used directly. The `RewardManager` multiplies each term's
output by its `weight` and by `step_dt`. SKRL's `rewards_shaper_scale` for racing is `1.0` (no
extra scaling).

13 terms are registered on `RewardsCfg`, with weights coming from the active
`RACING_REWARD_PRESET` (`RACING_REWARD_PRESET` env var, `may` default or `current`;
`racing_quadcopter_env_cfg.py`):

| Term | Function | `may` weight | `current` weight | Description |
|------|----------|--------------|-------------------|-------------|
| `gate_passage` | `reward_gate_passage` | +250.0 | +250.0 | Binary 1.0 on gate crossing frame |
| `gate_progress` | `reward_gate_progress` | +20.0 | +20.0 | Distance reduction toward current gate (clamped ±1 m) |
| `gate_centering` | `reward_gate_centering` | +0.5 | +4.0 | Alignment bonus on approach side, gated by forward speed |
| `velocity_toward_gate` | `reward_velocity_toward_gate` | +2.0 | +2.0 | Velocity projection onto gate direction, approach side only |
| `velocity_limit` | `penalty_velocity_limit` | −0.05 (inert: speed limits off) | −1.5 | Quadratic penalty for exceeding per-env speed cap |
| `angular_velocity_limit` | `penalty_angular_velocity_limit` | 0.0 | −0.2 | Quadratic penalty for exceeding per-env angular speed cap |
| `orientation` | `penalty_orientation` | −0.005 | −0.005 | Squared tilt angle from upright |
| `angular_velocity` | `penalty_angular_velocity` | −0.001 | −0.03 | Sum of squared body angular rates |
| `action_smoothness` | `penalty_action_smoothness` | −0.0001 | −5.0 | Squared action deltas between steps |
| `action_magnitude` | `shared_features.penalty_action_magnitude` | −0.001 (clip_excess_multiplier ×1) | −1.0 (clip_excess_multiplier ×100 outside [-1,1]) | Squared raw action magnitude |
| `crash` | `penalty_crash` | −50.0 | −50.0 | Binary 1.0 on collision OR upside-down |
| `wrong_side` | `penalty_wrong_side` | −0.1 | −0.1 | 1.0 when drone is on the exit side of the gate |
| `perception` | `penalty_perception` | −0.01 | −0.01 | Camera-FOV penalty for the next gate (active in both presets) |

`may` additionally has `adaptive_curriculum_enabled=False` and `speed_limits.enabled=False`
(so `velocity_limit`/`angular_velocity_limit` return zero); `current` has both enabled
(speed limits: linear 2-15 m/s, angular 1-5 rad/s).

**`gate_centering` formula:**
```
centering_value = exp(-3 * lateral_dist / gate_size)
proximity_factor = exp(-0.5 * |dist_to_plane|)
motion_factor = clamp(forward_speed / reference_speed, 0, 1)   # reference_speed = 15 m/s
reward = centering_value × proximity_factor × motion_factor × on_approach_side
```
Active only while `dist_to_plane <= 0` (approach side), preventing spurious centering reward behind the gate; `motion_factor` prevents a stationary drone from farming this term.

**`wrong_side` note:** fires every step the drone is on the exit side without having passed. On a chicane track where the next gate faces the opposite direction, the drone will briefly be on the wrong side as it turns; the penalty is mild (−0.1 weight) but can discourage approach.

### 6.4 Termination Conditions

| Condition | Function | `time_out` | Active when |
|-----------|----------|------------|-------------|
| `time_out` | `time_out` | True | `episode_length_buf >= max_episode_steps` |
| `upside_down` | `upside_down` | False | `_curriculum_stage == 0` AND `projected_gravity_b[2] > 0.7` |
| `below_height` | `below_height_threshold` | False | `root_pos_w[2] < min_height (0.2 m)` |
| `runaway` | `runaway_velocity` | False | `|root_lin_vel_w| > max_velocity (50 m/s)` |
| `collision` | `collision_termination` | False | `crash_termination=True` AND `_collision_termination_buf` |
| `lap_complete` | `lap_complete` | True† | `eval_mode=True` AND `_gates_passed >= _gate_counts` |

†`lap_complete` is registered with `time_out=True` so it triggers a timeout-type reset rather than a failure reset. This prevents value bootstrap from penalizing successful lap completion.

**Collision termination gating:**
- `_collision_termination_buf` is only True when both contact force exceeds threshold AND `_collision_detection_enabled[env_id]` is True
- `_collision_detection_enabled` is set per-env based on the previous episode's performance (see [Section 6.6](#66-curriculum))
- In eval mode (`_curriculum_stage=1000`), all collision terminations are active (stage gate removed)

**`upside_down` threshold:** `projected_gravity_b[2] > 0.7` — triggers when the body +Z axis has a positive dot product with gravity exceeding 0.7 (roughly 45° past inverted). Only in stage 0.

### 6.5 Episode Reset (Events)

Reset pipeline runs in this order on each env that terminates:

1. **`collect_episode_metrics_before_reset`** (called from `_reset_idx`, before super) — snapshots episode stats, accumulates sliding windows, saves `_last_episode_*` for curriculum, zeros all live accumulators
2. **`reset_scene_to_default`** (EventManager mode=reset) — resets Isaac simulation state
3. **`reset_racing_spawn`** (EventManager mode=reset) — sets spawn position and velocity
4. Shared `reset_action_noise` and `randomize_speed_limits` EventTerms sample optional per-episode state

**Spawn modes** (set by `method` parameter in `reset_racing_spawn`):

| Mode | Description |
|------|-------------|
| `"ground"` | 2 m behind a random gate, heading directly at it, zero velocity |
| `"uniform"` | Box on approach side: depth [1,4] m, ±1.5 m lateral, ±0.75 m vertical; attitude aimed so the FPV camera looks at the gate: yaw ±60° about the gate direction, pitch ±45° about the camera-aim pitch, roll ±45°; small random velocity |
| `"mixed"` | Stochastically picks ground (prob=`ground_init_prob`) or uniform per env |
| `"fixed"` | 8-lane grid behind gate 0, used in Play configs (`fixed_spawn=True`) |

Training configs use `method="uniform"` (set in `EventsCfg.reset_racing_spawn`).

The uniform spawn height is clamped to `min=0.3 m` to avoid immediate `below_height_threshold` termination (threshold is 0.2 m).

**Recovery spawns** (stage 2+, training only):
After the normal spawn, a fraction `recovery_spawn_prob` of envs have their orientation and velocity overwritten with fully random values:
- Orientation: uniform random unit quaternion (any attitude, including inverted)
- Linear velocity: random direction, speed in `[2, 6] m/s`
- Angular velocity: ±3 rad/s per axis

Recovery spawns are suppressed in eval mode regardless of `recovery_spawn_prob` and `_curriculum_stage`.

**Speed-limit enforcement and resampling:** disabled limits produce no observation
columns and zero penalties. When enabled, both penalties enforce the configured maximum
limits. After the one-way curriculum latch activates, the reset
`randomize_speed_limits` event samples both limits for only the resetting environments. Evaluation
activates enabled randomization immediately; equal range endpoints make evaluation deterministic.
In the AI-GP experimental curriculum (its own frozen copy), both speed-limit reward weights can grow to 5× their
base magnitude when normalized route progress is high, and are held—not reduced—when progress declines.

Adaptive curriculum decisions use a shared geometric score from
`tasks/manager_based/mdp/route_progress.py`. It projects the episode start and final drone
positions onto the closest points of an ordered straight-segment polyline and divides
covered arc length by the distance reachable at that episode's linear speed limit over
the full episode duration. Every route prepends the episode spawn as its origin (the shared
module dropped the optional "official start" prefix on 2026-09-26; the AI-GP experimental
snapshot keeps its own copy of `route_progress.py` with that feature). Hovering uses its ordered
target poses and subtracts their required dwell time
from the travel budget. Original lap/gate and waypoint-completion computations are not
removed; they remain independent metrics.
The reward and randomization controllers are likewise task-independent and live in
`tasks/manager_based/mdp/adaptive_curriculums.py`; task configs only provide target names,
thresholds, and an optional completion term used for secondary logging.

### 6.6 Curriculum

Three sequential stages governed by `apply_staged_curriculum` (called every step by `CurriculumManager`):

**Stage 0 → 1** (triggered at `common_step_counter >= curriculum_stage1_steps`):
- `upside_down` termination is enabled only in stage 0. After advancing, it returns all-False, so inverted drones are no longer immediately terminated.
- Crash and angular_velocity reward weights are restored from stored `_stage2_*` values (they were multiplied by `_STAGE1_BOOST=1.0` in stage 0, so currently no actual change).

**Stage 1 → 2** (triggered at `common_step_counter >= curriculum_stage2_steps`, if > 0):
- Recovery spawns activate: `recovery_spawn_prob` fraction of envs per episode spawn with random attitude/velocity.

**`_curriculum_stage` sentinel values:**
- `0` = stage 0 (hover, upside-down termination active)
- `1` = stage 1 (full racing, no upside-down termination)
- `2` = stage 2 (recovery spawns active)
- `1000` = eval mode (all terminations active, no progression, no recovery spawns)

**Collision detection curriculum** (per-env, independent of stage):

`_collision_detection_enabled[env_id]` is recomputed every step from the **previous episode's** metrics:

| Criterion | Default threshold |
|-----------|-------------------|
| Gates passed | ≥ `curriculum_min_gates_passed` (10) |
| Mean passage velocity | ≥ `curriculum_min_mean_velocity` (2.5 m/s) |
| Mean lateral passage offset | ≤ `curriculum_max_center_offset` (0.25 m) |

An env that never passed a gate in the previous episode vacuously satisfies the velocity and centering criteria; it is gated out only by the gates-passed count. This prevents an agent that never moved from enabling its own crash termination.

---

## 7. Terrain System

### 7.1 Random Track (Procedural)

`create_racing_gates_terrain(seed)` returns a `TerrainGeneratorCfg` with a **4×4 grid** of 15×15 m cells. Each call to `racing_gates_terrain(difficulty, cfg)` generates one cell.

**Grid parameters:**
- `difficulty_range = (0.7, 1.0)` — all cells are in the harder range
- 4 rows → 4 difficulty levels; 4 columns → 4 seeds per difficulty level → 16 unique circuits

**Trajectory types** (selected randomly per cell when `trajectory_type="random"`):
- Cubic spline with random control points — `max(12, int(16 + 14 × difficulty))` control points; the generator samples difficulty 0.7–1.0, so 25–30 in practice
- Chicane — alternating lateral offsets producing tight S-curves; more sections at higher difficulty

**Gate placement:**
- Spacing: `gate_spacing_range = (7.5, 15.0)` m → average 11.25 m; gate count = `int(total_arc / avg_spacing)`, so it varies with the generated trajectory's arc length (no fixed guarantee)
- Gate size: `gate_size_range = (0.7, 1.5)` m; interpolated as `max - difficulty × (max - min)`
  - difficulty=0.7 → size ≈ 0.94 m
  - difficulty=1.0 → size = 0.7 m
- Gate height: `gate_height_range = (0.7, 4.5)` m, oscillating with difficulty-scaled amplitude

**Out-of-bounds gates:** gates whose centre is closer than 0.5 m to any tile edge are skipped (the arc position still advances, so gate count may be lower than the computed upper bound).

### 7.2 Fixed Track (Deterministic)

`create_fixed_track_terrain(gates, ...)` tiles the same circuit in all 16 cells (grid size is
configurable via `num_rows`/`num_cols`). Gate positions are loaded from a YAML file (path set by
`gate_track_yaml`, overridable via the `TRACK_YAML` env var) through
`terrains/track_io.py::load_gates_from_yaml`, which returns a list of `GateSpec`:

```python
@dataclass
class GateSpec:
    position: list[float]   # [x, y, z], corner-origin local frame (0-15 m)
    rotation: list[float]   # [roll, pitch, yaw] in degrees; yaw is CW from +Y viewed from above
```

The default track is `configs/tracks/default.yaml` (a 10-gate circuit including a split-S
maneuver and a chicane, per the comment above `RacingQuadcopterEnvCfg_FIXED` in
`racing_quadcopter_env_cfg.py`); `configs/tracks/simple.yaml` is a 4-gate square used for
debugging, and `configs/tracks/wide.yaml` (6 gates) is the track `fpv_play.sh` flies. `GateSpec`
itself is defined in `terrains/racing_gates_cfg.py`. There is no longer a hard-coded `FIXED_GATES` / `FIXED_GATES_` Python list — both are
YAML files now.

### 7.3 Gate Coordinate Convention

Gates use a right-handed local frame:
- **+X** = right (horizontal, along bar width)
- **+Y** = forward = **passage direction** (drone flies through in +Y)
- **+Z** = up

Procedurally generated (random-track) gates are always upright (roll = pitch = 0, +Z is world
up). Gates loaded from a track YAML (`GateSpec.rotation = [roll, pitch, yaw]`) may have non-zero
roll/pitch and can tilt.

`rotation[2]` (yaw) in `GateSpec`: **clockwise from +Y** viewed from above:
- 0° → drone passes in world +Y
- 90° → drone passes in world +X

Quaternions are stored in **(w, x, y, z)** order throughout the codebase (IsaacLab convention). scipy uses (x, y, z, w); all conversions are explicit.

**World-frame gate axes** (computed on the fly from stored quaternion):
```python
forward = quat_apply(gate_ori, [0, 1, 0])   # gate +Y in world
right   = quat_apply(gate_ori, [1, 0, 0])   # gate +X in world
up      = quat_apply(gate_ori, [0, 0, 1])   # gate +Z in world
```

### 7.4 Gate Registry Pattern

IsaacLab's `TerrainGenerator` calls the terrain function, collects its trimesh output, but **discards any state written to `cfg`** because it copies the dataclass. Gate data is therefore stored in a module-level dictionary:

```python
gate_registry: dict[tuple[float, int | None], GateData] = {}
```

Key: `(round(difficulty, 6), seed)`. After terrain generation, `RacingQuadcopterRLEnv._setup_gate_tracking()` reads the registry and converts positions to world frame:

```
world_pos = local_pos + centre_offset + terrain_origin
centre_offset = (-size_x/2, -size_y/2, 0)     # corner → centre frame
terrain_origin = terrain.terrain_origins[row, col]  # world origin of cell
```

`GateData` fields:

| Field | Type | Description |
|-------|------|-------------|
| `positions` | `[G, 3]` np | Gate centres in corner-origin local frame |
| `orientations` | `[G, 4]` np | Quaternions (w, x, y, z) |
| `gate_size` | float | Aperture side length in metres |
| `bar_thickness` | float | Bar diameter in metres |
| `start_position` | `[3]` np | First trajectory waypoint |
| `start_tangent` | `[3]` np | Tangent at first waypoint |
| `waypoints` | `[W, 3]` np | Full trajectory sample points |

**Env assignment:** env id `i` maps to cell `(row, col) = ((i // num_cols) % num_rows, i % num_cols)`. If `num_envs > num_cells`, the mapping wraps — multiple envs share the same layout but different terrain_origins.

### 7.5 Gate Passage Detection

Called every step in `check_gate_passages()`:

1. **Signed distance test:** compute dot product of `(drone_pos - gate_pos)` with gate forward direction at both previous and current drone positions. A crossing occurs when sign flips from negative to positive.
2. **Interpolated crossing point:** linear interpolation to find where the trajectory exactly crossed the gate plane.
3. **Aperture check:** project crossing offset onto gate right and up axes; both must be within `gate_size / 2`.

If passed:
- `_gates_passed[env]` increments
- `_current_gate_idx[env]` advances modulo `_gate_counts[env]`
- Passage velocity and lateral offset are accumulated for curriculum stats
- Lap completion is checked: if `gates_passed % gate_count == 0`, the lap duration is recorded

**Important:** `_prev_drone_pos` is updated **after** reward computation in `step()`. This is intentional: `reward_gate_progress` reads `_prev_drone_pos` as the prior step position; updating it before the reward would zero the progress signal every step.

---

## 8. Observation Corruption

There is no single `ObservationNoiseCfg`. Each observation term that wants noise gets its own
`NoiseModelWithAdditiveBiasAndDropoutCfg` (`mdp/noise.py`, built by `drone_noise_model_factory`)
passed as that `ObsTerm`'s `noise=` argument — this runs through IsaacLab's standard
`ObsTerm.noise` pipeline (`enable_corruption=True`), it is not manual post-processing.

**Per-term behaviour (applied in order, only while `cfg.enable=True`):**

1. **Per-episode bias**: `NoiseModelWithAdditiveBias` samples a bias from `bias_noise_cfg` at each reset and adds it every step.
2. **Per-step Gaussian noise**: additive noise from `noise_cfg` (`GaussianNoiseCfg`) each step.
3. **Dropout** (`dropout` → `keep_prob = 1 - dropout`): entire term zeroed with probability `dropout` each step, independently per env.

**Defaults:** every noise model built by `racing_quadcopter_env_cfg.py` (`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `obs_gates`) is constructed with `enable` left at its default (`False`), so **no observation noise is applied by default** even though `std_noise`/`std_bias` values are set — those values are the *targets* the optional adaptive-randomization curriculum (`adaptive_curriculum_enabled=True`, off by default) lerps `enable`/`std` toward as route-progress improves. `last_action` and `speed_limits` have no noise model. See `docs/README_randomization.md`.

---

## 9. Action Corruption

Configured via `ActionNoiseCfg` (`mdp/shared_features.py`) on `env_cfg.action_noise`. Applied inside
the action term's own `process_actions` (`QuadcopterRatesAction._compute_delayed_actions` /
`QuadcopterMomentsAction.process_actions` in `mdp/actions.py`) — there is no separate
`QuadcopterThrustAction` class.

**Corruption steps, in order (all gated by `action_noise.enable`, default `False`):**

1. **Per-step Gaussian noise** (`gaussian_std`, default `0.0`): additive noise on all 4 action channels.
2. **Per-episode bias** (`per_episode_bias=True` by default, `bias_range`, default `(0.0, 0.0)`): constant offset per episode, sampled at reset by the shared `reset_action_noise` event.
3. **First-order low-pass filter**: `cmd_filtered = alpha × cmd_new + (1 - alpha) × cmd_prev`. `alpha` is per-episode-sampled from `lpf_alpha_range` (default `(1.0, 1.0)`, i.e. no filtering) when `per_episode_lpf_alpha=True` (default); otherwise the static `lpf_alpha` field (default `1.0`) is used.
4. **Clipping** (`clip_actions=True` by default): after all corruption, actions are re-clamped to `[-1, 1]`.

With every field at its default, enabling `action_noise.enable=True` alone is a no-op — `gaussian_std=0`, `bias_range=(0,0)` and `lpf_alpha_range=(1,1)` add no actual corruption; each knob must be set explicitly (or the adaptive-randomization curriculum, which lerps them toward its profile's target values, must be enabled). See `docs/README_randomization.md`.

---

## 10. Statistics and Logging

### Sliding windows

Three `deque(maxlen=2000)` windows accumulate per-episode data:

| Window | Entry type | When appended |
|--------|-----------|---------------|
| `_speed_window` | Mean speed (m/s) | Once per env per reset: episode mean speed |
| `_run_window` | Track completion [0, 1] | Per completed lap: 1.0; per crash-terminated partial run: `(gp % gc) / gc` |
| `_gates_window` | Total gates passed | Once per env per reset: `_gates_passed` |

**Partial run handling:**
- Crashed/terminated before lap complete → `(gp % gc) / gc` appended (progress since last lap boundary)
- Timed-out without completing lap → nothing appended (discarded)
- Completed laps → one `1.0` appended per full lap

### Warmup period

`stats_warmup_s` (default `0.0`, set to `30.0` in Play configs) suppresses all stats accumulation for the first N simulation seconds. This prevents the early garbage episodes — before drones have settled into steady-state racing — from polluting the sliding windows.

**How it works:**
- `init_stats` computes `_stats_warmup_steps = int(stats_warmup_s / step_dt)` and sets `_stats_warmup_done = (steps == 0)`.
- `_in_warmup(env)` checks `common_step_counter < _stats_warmup_steps`. On the first step past the threshold it latches `_stats_warmup_done = True` and prints the completion message once.
- Both `step_stats` and `accumulate_episode_stats` call `_in_warmup` and return early while it is active.

**Console messages:**

On startup (if warmup > 0):
```
[EVAL Stats] Warmup: 30 s (750 steps) — stats collection paused until drones settle.
```

During warmup (printed every `speed_log_interval_s`):
```
[EVAL Stats | warming up] 12 / 30 s elapsed — waiting for drones to settle...
```

On warmup completion:
```
[EVAL Stats] Warmup complete at 30.0 s — stats collection active.
```

### Console output

After warmup, printed every `speed_log_interval_s` seconds (default 10 s):
```
[EVAL Stats | n=NNN] Avg speed: X.X km/h  |  Top speed: X.X km/h  |  Gates/ep: X.X  |  Lap%: XX.X%  |  Lap time: X.X s
```

`n` = number of entries in `_run_window` at print time. The mode prefix (`TRAIN`, `EVAL`, or `RUN`) is set by the `RACING_RUN_MODE` environment variable, which train.py sets to `TRAIN` and play.py sets to `EVAL`.

### TensorBoard / WandB

`extras["log"]` dictionary is populated each reset with:
- `Episode_Metrics/gates_passed` — mean gates passed over reset batch
- `Episode_Metrics/mean_velocity_kmh` — mean drone speed over episode (km/h)
- `Episode_Metrics/lap_completion_pct` — mean lap completion %, averaged across completed laps and terminated partial runs (skipped if no run is countable)
- `Episode_Metrics/mean_lap_time_s` — mean lap duration (only when laps completed)
- `Curriculum/stage` — current stage (0, 1, 2, or 1000)
- (skill-eval only) `terrain_{i}/...` — same metrics restricted to envs assigned to terrain `i` (each terrain becomes its own top-level tensorboard panel)

SKRL routes these to TensorBoard automatically via `environment_info: log` in the trainer config.

### Global lap totals

`env._total_lap_time_sum` and `env._total_lap_count` accumulate across the entire run (not a sliding window). Used by play.py to report mean lap time in the sweep result line.

---

## 11. Training Setup

### 11.1 SKRL PPO Configuration

File: `agents/skrl_ppo_cfg.yaml`

**Network architecture:**

| Component | Type | Layers | Activation |
|-----------|------|--------|-----------|
| Policy | `GaussianMixin` | 3 × 256 | ELU |
| Value | `DeterministicMixin` | 3 × 256 | ELU |

Both share the same network width (separate=False uses independent weights despite same architecture).

Policy outputs a Gaussian distribution. During training: sample from distribution. During eval: use `mean_actions` (deterministic).

**PPO hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `rollouts` | 24 | Steps collected per update per env |
| `learning_epochs` | 5 | Gradient updates per rollout batch |
| `mini_batches` | 4 | Batch splits per epoch |
| `discount_factor` | 0.99 | γ |
| `gae_lambda` | 0.95 | λ for GAE advantage estimation |
| `learning_rate` | 5e-4 | Initial LR |
| `learning_rate_scheduler` | `KLAdaptiveLR` | Adjusts LR to target KL=0.016 |
| `grad_norm_clip` | 1.0 | Gradient clipping |
| `ratio_clip` | 0.2 | PPO clip ε |
| `value_clip` | 0.2 | Value function clip |
| `entropy_loss_scale` | 0.01 | Entropy bonus coefficient |
| `rewards_shaper_scale` | 1.0 | No additional reward scaling |
| `observation_preprocessor` | `RunningStandardScaler` | Online obs normalization |
| `value_preprocessor` | `RunningStandardScaler` | Online value normalization |
| `timesteps` | 10000 | Default total env steps (override with `--max_iterations`) |

**Checkpoint path:** `logs/skrl/racing_quadcopter/<timestamp>_ppo_torch/checkpoints/`

### 11.2 RSL-RL PPO Configuration

File: `agents/rsl_rl_ppo_cfg.py`

A secondary trainer option (not used in sweep). `RacingQuadcopterPPORunnerCfg`: network hidden dims `[200, 100]` (tanh activation), `num_steps_per_env=48`, `max_iterations=10000`.

`RacingQuadcopterRecurrentPPORunnerCfg` (used by the `-RNN-*` task IDs): GRU actor and critic (`rnn_hidden_dim=128`, `rnn_num_layers=1`) in front of the same `[200, 100]` MLP heads, `num_steps_per_env=64`. The `-RNN-*` IDs only differ for rsl_rl; with skrl they load the normal MLP `skrl_ppo_cfg.yaml`.

### 11.3 Reward Scaling Chain

There is no `MANAGER_SCALE` or manager-scale/direct-weight distinction: `RewTerm.weight` values
in `RewardsCfg` (see [Section 6.3](#63-reward-terms)) are used as-is.

```
effective reward per step = function_output × weight × step_dt
```

SKRL's `rewards_shaper_scale` for racing is `1.0`, so no extra scaling is applied on top of the
`RewardManager`'s `weight × step_dt`. When sweeping reward weights, `DIRECT_<TERM>` in
`scripts/sweep.py` maps 1:1 onto `rewards.<term>.weight` (see [Section 13.5](#135-direct_-shorthand)).

---

## 12. Evaluation Setup

Run via `play.py`. Key differences from training:

1. **No gradient** — everything runs under `torch.inference_mode()`
2. **Deterministic actions** — `mean_actions` from the Gaussian policy
3. **`eval_mode=True`** — `_curriculum_stage=1000`, `lap_complete` termination active, no recovery spawns
4. **No noise** — observation and action corruption disabled
5. **No mass/actuator randomization**
6. **`fixed_spawn=True`** — all envs start from gate 0 in a grid formation

**Checkpoint discovery (play.py):**
- `--checkpoint PATH` takes precedence
- `--use_pretrained_checkpoint` downloads from Nucleus (if available)
- Otherwise, `get_checkpoint_path()` finds the most recent run matching `.*_ppo_torch` under the experiment's log root

**`--eval-steps N`:** terminate after N environment steps (not wall-clock steps). After exiting the loop, play.py prints a `[SWEEP_RESULT]` line for sweep.py to parse.

**`--real-time`:** sleep `step_dt - elapsed` after each step to match real time.

---

## 13. Sweep System

File: `scripts/sweep.py`

### 13.1 How It Works

For every parameter combination in the sweep:

1. **Train:** calls `train.py` with `--override` flags and a unique `--exp-name`. The exp-name encodes a hash of the parameter combination, ensuring stable and filesystem-safe directory names under `logs/skrl/sweep/<exp_name>/`.

2. **Find checkpoint:** `_find_best_checkpoint(exp_name)` searches `logs/skrl/sweep/<exp_name>/` for the most recent run directory (by mtime). Prefers `best_agent.pt`; falls back to `max(agent_*.pt, key=step_number)` using regex-based natural sort.

3. **Evaluate:** calls `play.py` with `--checkpoint` and `--eval-steps`. Streams stdout and captures for `[SWEEP_RESULT]` parsing.

4. **Parse result:** extracts `Speed`, `Lap`, `Gates`, `LapTime` from the `[SWEEP_RESULT]` line printed by play.py.

5. **Append CSV:** writes one row per combination to `logs/sweep/results_<timestamp>.csv` with all parameter values and metrics.

**Invalid override detection:** if train or play output contains any override error marker (e.g. "Unknown override key"), the sweep aborts immediately rather than silently running wrong configurations.

**Environment variables set for subprocesses:**
- `TQDM_DISABLE=1` — suppresses tqdm progress bars that pollute piped stdout with CR-update lines
- `PYTHONUNBUFFERED=1` — ensures `print()` flushes immediately when stdout is a pipe
- `RACING_RUN_MODE` — set to `TRAIN` for train.py and `EVAL` for play.py

### 13.2 Configuration via YAML

The intended workflow is to write a YAML config file and pass it with `--config`. Do not edit `sweep.py` directly; the script ships with single-value defaults that act as a smoke test.

```bash
python scripts/sweep.py --config configs/my_sweep.yaml
```

**Supported YAML keys:**

| Key | Type | Description |
|-----|------|-------------|
| `task_train` | str | Task ID used for training |
| `task_play` | str | Task ID used for evaluation |
| `train_iterations` | int | `--max_iterations` passed to train.py |
| `eval_steps` | int | `--eval-steps` passed to play.py |
| `num_envs` | int | `--num_envs` for both train and eval |
| `extra_train_args` | list[str] | Fixed extra CLI args appended to train.py |
| `extra_play_args` | list[str] | Fixed extra CLI args appended to play.py |
| `sweep_mode` | str | `"grid"` or `"sequential"` (see [Section 13.3](#133-sweep-modes)) |
| `param_grid` | dict | Parameter grid (see [Section 13.4](#134-param_grid-reference)) |

**Minimal example (`my_sweep.yaml`):**
```yaml
task_train: Isaac-Racing-Quadcopter-Manager-Fixed-v0
task_play:  Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0
train_iterations: 5000      # ~120k env steps (5000 × 24 rollouts)
eval_steps: 3000
num_envs: 4096
extra_train_args: ["--headless"]
extra_play_args:  ["--headless"]
sweep_mode: sequential

param_grid:
  recovery_spawn_prob:        [0.0, 0.1, 0.2, 0.3]
  curriculum_stage2_steps:    [0,   5000, 5000, 5000]
  DIRECT_VELOCITY_TOWARD_GATE: [2.0, 2.0,  2.0,  2.0]
```

### 13.3 Sweep Modes

**`grid`** (default) — full Cartesian product: every combination of every value across all keys.
- 2 values × 3 values × 2 values → 12 runs.
- Grid size grows multiplicatively; trim value lists if runs become too expensive.

**`sequential`** — index-aligned zip: run `i` picks `param_grid[key][i]` for every key.
- All value lists must have the same length.
- Useful for structured ablations where parameters co-vary deliberately (e.g. a curriculum schedule designed as a unit).
- Example: `p1=[0.0, 0.2], p2=[5000, 10000]` → runs `(0.0, 5000)` and `(0.2, 10000)`, not all four combinations.

### 13.4 PARAM_GRID Reference

**Key syntax:**
- Simple attribute: `"recovery_spawn_prob"` → sets `env_cfg.recovery_spawn_prob`
- Dotted path: `"rewards.crash.weight"` → sets `env_cfg.rewards.crash.weight`
- Nested: `"action_noise.gaussian_std"` → sets `env_cfg.action_noise.gaussian_std`
- Tuple strings: `"(8.0, 15.0)"` is parsed by `ast.literal_eval` and coerced to tuple

### 13.5 DIRECT_ Shorthand

Keys starting with `DIRECT_` map 1:1 onto `rewards.<term>.weight`:

```yaml
param_grid:
  DIRECT_VELOCITY_TOWARD_GATE: [0.1, 2.0]
# → --override rewards.velocity_toward_gate.weight=0.1
#   --override rewards.velocity_toward_gate.weight=2.0
```

`rewards.py` has no `MANAGER_SCALE` constant; `sweep.py` looks for one via regex (without
importing the Racing_Quadcopter package, to avoid triggering Omniverse extension loading in the
sweep script itself) and falls back to a scale of `1.0` when it isn't found, which is why the
mapping above is 1:1.

### 13.6 Output and Metrics

CSV columns: `exp_name` + all `param_grid` keys + `[speed_kmh, lap_pct, mean_gates, lap_time_s, status]`.

Metrics from `[SWEEP_RESULT]` line printed by play.py:

| Metric | Source | Description |
|--------|--------|-------------|
| `speed_kmh` | `_speed_window` | Mean episode speed in km/h |
| `lap_pct` | `_run_window` | Mean track completion (%) |
| `mean_gates` | `_gates_window` | Mean gates passed per episode |
| `lap_time_s` | `_total_lap_time_sum / _total_lap_count` | Mean lap duration (s); 0 if no laps completed |

`status` field: `"ok"` on success; `"train_failed(<rc>)"`, `"no_checkpoint"`, or `"no_result(exit=<rc>)"` on failure.

---

## 14. Override Utility

File: `scripts/skrl/override_utils.py`

Both train.py and play.py support `--override KEY=VAL`. Multiple `--override` flags can be combined. The KEY can be a dotted attribute path.

**Parsing rules:**
- `"none"` / `"null"` → `None`
- `"true"` / `"false"` → `bool`
- Otherwise: attempt `ast.literal_eval` (handles ints, floats, tuples, lists, dicts)
- Fallback: raw string

**Type coercion:** after parsing, the value is coerced to match the current field type. This means:
- A float field receiving `"10"` gets `10.0`, not `10`
- A tuple field receiving `"[8.0, 15.0]"` gets `(8.0, 15.0)`
- A bool field receiving `"1"` gets `True`

**Examples:**
```bash
--override episode_length_s=30.0
--override speed_limits.enabled=True
--override speed_limits.linear_range="(2.0, 15.0)"
--override speed_limits.angular_range="(1.0, 8.0)"
--override rewards.crash.weight=-8000.0
--override action_noise.enable=True
--override action_noise.gaussian_std=0.02
--override recovery_spawn_prob=0.3
```

If a key does not exist, `apply_overrides` raises `AttributeError`. If the value cannot be coerced, it raises `ValueError`.

---

## 15. Environment Configuration Reference

All parameters live in `RacingQuadcopterEnvCfg`. Play configs override a subset.

### Core timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decimation` | 2 | Physics steps per RL step |
| `episode_length_s` | 20.0 s | Maximum episode duration |
| `sim.dt` | 1/50 s | Physics timestep |

Derived: `step_dt = decimation × dt = 0.04 s`, `max_episode_steps = episode_length_s / step_dt = 500`.

### Scene

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scene.num_envs` | 4096 (1024 on Play configs) | Number of parallel environments |
| `scene.env_spacing` | 5.0 m | Spacing between env origins |
| `scene.replicate_physics` | False | Use PhysX's scene replication |

### Physics limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_height` | 0.2 m | Below this → `below_height_threshold` termination |
| `max_velocity` | 50.0 m/s | Above this → `runaway_velocity` termination |
| `collision_force_threshold` | 0.001 N | Minimum contact force for collision detection |
| `upside_down_threshold` | 0.7 | `projected_gravity_b[2]` above this → upside-down termination (stage 0 only) |
| `crash_termination` | True | Master switch for collision termination |

### Optional speed limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speed_limits.enabled` | Depends on `RACING_REWARD_PRESET`: `False` (`may`), `True` (`current`) | Add paired observations, penalties, and curriculum-latched randomization |
| `speed_limits.linear_range` | (2.0, 15.0) m/s | Used only while `enabled=True` |
| `speed_limits.angular_range` | (1.0, 5.0) rad/s | Used only while `enabled=True` |
| `adaptive_curriculum_enabled` | Depends on `RACING_REWARD_PRESET`: `False` (`may`), `True` (`current`) | Enable shared reward/noise/dynamics progression |

### Spawn

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fixed_spawn` | False | True → 8-lane grid at gate 0 (Play configs) |
| `recovery_spawn_prob` | 0.2 | Fraction overwritten with random attitude in stage 2 |

`EventsCfg.reset_racing_spawn` is configured with `params={"method": "uniform"}`, so every default
training run uses the uniform-box spawn; `"mixed"` (which would read a `ground_init_prob`
attribute via `getattr(env.cfg, "ground_init_prob", 0.5)`) is reachable in `mdp/events.py` but is
never selected by any current task config, and `ground_init_prob` is not a declared field on any
env cfg.

### Curriculum thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `curriculum_stage1_steps` | 10000 | Steps before stage 0→1 transition |
| `curriculum_stage2_steps` | 10000 | Steps before stage 1→2 transition (0 = disabled) |
| `curriculum_min_gates_passed` | 10 | Gates required to enable collision termination |
| `curriculum_min_mean_velocity` | 2.5 m/s | Speed required to enable collision termination |
| `curriculum_max_center_offset` | 0.25 m | Max lateral offset for collision termination |

### Eval

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_mode` | False | True → _curriculum_stage=1000, lap_complete active, no recovery spawns |
| `seed` | 42 | Random seed for terrain generation and env initialization |

### Logging

| Parameter | Default | Play default | Description |
|-----------|---------|-------------|-------------|
| `speed_log_interval_s` | 10.0 s | 10.0 s | How often to print stats to console |
| `stats_warmup_s` | 0.0 s | 30.0 s | Suppress stats accumulation for this many simulation seconds at run start |

### Observation config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `observation.num_next_gates` | 3 | Gate lookahead depth K |

---

## 16. Visualization

All visualization objects are USD `BasisCurves` or `SphereLight` prims created in `_setup_gate_tracking()`. They have no `CollisionAPI`, so they are invisible to the physics engine.

Functions in `racing_quadcopter/trajectory_vis.py`:

| Function | Controlled by | Description |
|----------|--------------|-------------|
| `draw_trajectory_usd` | `draw_trajectory` | Green closed-loop polyline through all waypoints |
| `draw_gate_indicators_usd` | `draw_gate_indicators` | Blue stick forward of each gate showing passage direction |
| `draw_gate_labels_usd` | `draw_gate_labels` | 7-segment digit labels above each gate |
| `draw_gate_outlines_usd` | `demo_mode` | Emissive gate outlines (demo only) |
| `add_gate_sphere_lights_usd` | `demo_mode` | `UsdLux.SphereLight` at each gate centre |
| `apply_emissive_to_robots` | `demo_mode` | OmniPBR emissive material on all drone prims |
| `setup_dark_scene` | `demo_mode` | Removes default dome light, adds dim ambient |

All are created once at env startup; they are static for the entire run.

Labels use 7-segment encoding: each digit is drawn as a set of line segments, lying flat in the XY plane, above the gate centre.

---

## 17. Demo Mode

Set `demo_mode=True` (in cfg or via `--override demo_mode=True`).

Activates all visual overlays and switches to a cinematic rendering style suitable for recording:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `demo_emissive_intensity` | 30.0 | Emissive factor for trajectory/label/indicator curves |
| `demo_gate_color` | (1.0, 0.3, 0.0) | Orange, matches gate mesh color |
| `demo_gate_outline_emissive` | 20.0 | Emissive for gate outline overlay |
| `demo_gate_outline_width` | 0.06 m | Tube width of outline BasisCurves |
| `demo_gate_light_intensity` | 3000.0 nits | Sphere light brightness |
| `demo_gate_light_radius` | 1.5 m | Sphere light glow spread radius |
| `demo_drone_glow_color` | (1.0, 0.5, 0.0) | Orange drone glow |
| `demo_drone_glow_intensity` | 5.0 | Drone emissive scale |

Requires RTX rendering; not visible with rasterization.

---

## 18. Debug Flags

Two environment variables activate verbose logging without requiring code changes:

**`DEBUG_TERRAIN=1`**

Printed during environment initialization:
- Gate registry keys and count
- Terrain origins shape and unique origin count
- Gate positions sample for envs 0 and 1
- Gate counts and first-gate positions after reset

Printed during each reset:
- Reset env IDs sample
- Current gate index after reset
- Gate counts after reset
- First gate position after reset

**`DEBUG_REWARDS=1`**

Printed every step:
- `reward_buf` sample (first 5 envs)
- Per-term unweighted output (mean), weight, and weighted mean for all 13 reward terms
- Summed unweighted reward sample

Usage:
```bash
DEBUG_TERRAIN=1 DEBUG_REWARDS=1 python scripts/skrl/play.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0 \
    --num_envs 1 \
    --checkpoint <path>
```

---

## 19. Known Limitations and Design Notes

### Reward design

- **`wrong_side` on chicane tracks:** if the next gate faces opposite to the drone's direction of travel (common on chicane/random tracks), the drone will briefly be on the exit side as it turns. The penalty is mild (weight −0.1) but can create a small adverse gradient.

- **`gate_progress` after passage:** after passing a gate, `_current_gate_idx` advances. On the same step, `reward_gate_progress` computes progress toward the *new* gate, which may be far away, giving a temporary negative distance (increasing distance). This is expected behavior.

- **`_STAGE1_BOOST = 1.0`:** currently a no-op. The intent was to scale up crash/angular_velocity in stage 0 for extra stability emphasis; this can be made > 1.0 to increase hover-phase penalty severity.

### Terrain

- **Random track difficulty range (0.7, 1.0):** all tracks are on the harder end. The agent must handle sharp turns and tight gates from the start. For easier initial training, lower the minimum to 0.0–0.3.

- **Gate count varies by circuit:** a procedural circuit may produce anywhere from ~6 to ~20 gates depending on arc length and spacing randomization. The padding system handles variable counts correctly.

- **Seed correlation:** two RNG objects are created from the same seed for trajectory and gate placement, creating slight correlation between waypoint noise and gate spacing. This is intentional for reproducibility.

### Curriculum

- **Collision curriculum evaluated per-env independently:** collision termination can be enabled for some envs but not others in the same batch. This means some envs can crash-terminate while others (with lower performance last episode) do not.

- **Snapshot timing:** `_last_episode_*` buffers are updated at reset, not at collision. The curriculum uses the *previous* episode's performance, not the current episode's. This adds a one-episode lag to curriculum progression.

### Sweep

- **Checkpoint mtime ordering:** if a sweep experiment is reused (same `exp_name`), the most recent run directory by filesystem mtime is preferred. This is robust to naming changes but assumes filesystem timestamps are reliable.

- **Sweep abort on invalid overrides:** `PARAM_GRID` values are validated against the env_cfg attribute types only when the subprocess runs. A sweep with many combinations will abort on the first invalid override, even if it's in the middle of the grid. Validate keys manually before running a large sweep.

### Physics

- **Permanent wrench vs. motor model:** this env uses a single aggregate wrench (thrust + 3 moments) rather than simulating 4 individual motors. This is a simplification: real Crazyflies have rotor speed limits and cross-coupling. The model is sufficient for learning gate navigation but does not capture motor saturation dynamics.

- **Contact sensor update period:** `update_period=0.01 s` is shorter than the 0.02 s physics step, so the sensor refreshes once per physics step. At `decimation=2`, that is twice per RL step; `net_forces_w_history` with `history_length=2` captures both sub-steps.

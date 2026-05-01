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
│   └── sweep.py                 # Hyperparameter grid-search orchestrator
│
├── source/
│   └── Racing_Quadcopter/
│       └── Racing_Quadcopter/
│           ├── tasks/
│           │   └── manager_based/
│           │       └── racing_quadcopter/
│           │           ├── __init__.py                  # gym.register for all 4 tasks
│           │           ├── racing_quadcopter_env.py     # RacingQuadcopterRLEnv class
│           │           ├── racing_quadcopter_env_cfg.py # All configs + FIXED_GATES layout
│           │           ├── agents/
│           │           │   ├── skrl_ppo_cfg.yaml        # SKRL PPO hyperparameters
│           │           │   └── rsl_rl_ppo_cfg.py        # RSL-RL PPO hyperparameters
│           │           ├── mdp/
│           │           │   ├── __init__.py              # Re-exports all mdp submodules
│           │           │   ├── actions.py               # QuadcopterThrustAction + cfg
│           │           │   ├── observations.py          # Gate observations + noise wrappers
│           │           │   ├── rewards.py               # Reward/penalty functions + WEIGHT constants
│           │           │   ├── terminations.py          # Termination conditions
│           │           │   ├── events.py                # Episode reset + metrics collection
│           │           │   ├── curriculums.py           # Staged curriculum + collision gating
│           │           │   ├── gate_utils.py            # Gate helpers, passage detection, step update
│           │           │   └── stats.py                 # Sliding-window statistics + console logging
│           │           └── vis/
│           │               └── trajectory_vis.py        # USD BasisCurves visualization
│           └── terrains/
│               ├── __init__.py
│               ├── gate_base.py          # GateData, gate_registry, create_gate_frame
│               ├── racing_gates.py       # Procedural track generator
│               ├── racing_gates_cfg.py   # MeshRacingGatesTerrainCfg, MeshFixedGatesTerrainCfg
│               ├── racing_terrain.py     # create_racing_gates_terrain, create_fixed_track_terrain
│               └── fixed_gates.py       # Fixed circuit terrain function
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

Logs land in `logs/skrl/<task>/YYYY-MM-DD_HH-MM-SS_ppo_torch/`.

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

**Checkpoint auto-discovery:** if `--checkpoint` is not given, play.py searches `logs/skrl/<task>/` for the most recent directory matching `*_ppo_torch` and loads `best_agent.pt` from its `checkpoints/` subdirectory.

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
# Train with observation noise disabled and custom episode length
python scripts/skrl/train.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-v0 \
    --num_envs 4096 --headless \
    --override episode_length_s=30.0 \
    --override observation_noise.enable=False

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

---

## 3. Task Variants

Four gymnasium tasks are registered in `__init__.py`:

| Task ID | Config Class | Use |
|---------|-------------|-----|
| `Isaac-Racing-Quadcopter-Manager-v0` | `RacingQuadcopterEnvCfg` | Train on random procedural tracks |
| `Isaac-Racing-Quadcopter-Manager-Play-v0` | `RacingQuadcopterEnvCfg_PLAY` | Evaluate random-track policy |
| `Isaac-Racing-Quadcopter-Manager-Fixed-v0` | `RacingQuadcopterEnvCfg_FIXED` | Train on fixed deterministic circuit |
| `Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0` | `RacingQuadcopterEnvCfg_FIXED_PLAY` | Evaluate fixed-track policy |

**Training vs Play differences:**

| Parameter | Training | Play |
|-----------|----------|------|
| `eval_mode` | `False` | `True` |
| `fixed_spawn` | `False` | `True` |
| `stats_warmup_s` | `0.0` | `30.0` |
| `observation_noise.enable` | `True` | `False` |
| `action_noise.enable` | `True` | `False` |
| `randomize_robot_mass` | on | off |
| `randomize_robot_actuator_gains` | on | off |
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
| `ActionManager` | `ActionsCfg` | `QuadcopterThrustAction` |
| `ObservationManager` | `ObservationsCfg` | Sensor + gate + noise |
| `RewardManager` | `RewardsCfg` | 11 reward/penalty terms |
| `TerminationManager` | `TerminationsCfg` | 5 termination conditions |
| `EventManager` | `EventsCfg` | Mass/actuator randomization, spawn |
| `CurriculumManager` | `CurriculumCfg` | Staged progression |

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
| `_velocity_limit` | `[N]` | Per-env speed cap (m/s) |
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

**Crazyflie** from `isaaclab_assets`, loaded with `activate_contact_sensors=True` so all links contribute to the `ContactSensor`.

**Contact sensor** settings:
- `history_length = 2` — keeps two steps of contact force history
- `update_period = 0.01 s` — sensor refreshes every 10 ms (every physics step at 50 Hz → actual rate matches physics dt)

### Thrust model (`actions.py`)

The Crazyflie is driven by a `permanent_wrench_composer` (persistent force/torque applied to the body link each step). The action vector `[a0, a1, a2, a3]` maps to thrust and body moments:

```
thrust = thrust_to_weight × robot_weight × (a0 + 1) / 2    (N, body +Z)
moment = moment_scale × [a1, a2, a3]                         (N·m, body XYZ)
```

Parameters:
- `thrust_to_weight = 1.9` — maximum thrust is 1.9× robot weight. At action `a0=0` (neutral), thrust = 0.95× weight → slight hover margin.
- `moment_scale = 0.01` — scales moments to physically reasonable magnitudes for the Crazyflie.
- Robot weight is computed once at startup: `sum(masses) × |gravity|`.

**Action range:** all actions clipped to `[-1, 1]` before thrust/moment mapping. The normalized thrust command is `(a0 + 1) / 2`, mapping `[-1, 1] → [0, 1]`.

### Startup physics randomization

| Event | Mode | Parameters |
|-------|------|-----------|
| `randomize_robot_mass` | startup | Scale in `[0.9, 1.1]`, uniform |
| `randomize_robot_actuator_gains` | startup | Stiffness + damping scale `[0.9, 1.1]` |

Both are disabled in Play configs.

---

## 6. MDP Specification

### 6.1 Observation Space

**Policy group** — concatenated, history length 3 (each term stacked over 3 consecutive steps).

| Term | Dimension (per step) | Function | Description |
|------|----------------------|----------|-------------|
| `base_lin_vel` | 3 | `wrap_base_lin_vel` | Body-frame linear velocity (m/s) |
| `base_ang_vel` | 3 | `wrap_base_ang_vel` | Body-frame angular velocity (rad/s) |
| `projected_gravity` | 3 | `wrap_projected_gravity` | Gravity direction in body frame |
| `last_action` | 4 | `wrap_last_action` | Previous step action `[a0..a3]` |
| `obs_gates` | K×6 | `compute_next_gates_rel_and_fwd_body` | Next K gates: relative position + forward |
| `velocity_limit` | 1 | `velocity_limit_obs` | Per-env speed cap / 10 |

With default `K=3` and `history_length=3`: total observation dimension = `(3+3+3+4+18+1) × 3 = 96`.

**Gate observation details** (`compute_next_gates_rel_and_fwd_body`):

For each of the next K gates:
- `rel_pos [3]` — gate centre relative to drone, in drone **body frame**, divided by `_REL_POSITION_SCALE=10.0` to keep values near `[-1, 1]` for gates up to 10 m away
- `fwd [3]` — gate forward direction (+Y in gate frame), rotated to drone body frame (unit vector)

All K gates are gathered simultaneously using vectorized `torch.gather` over the padded gate table with modular indexing.

**`enable_corruption = False`** in `PolicyCfg.__post_init__`: the IsaacLab manager-level corruption pipeline is disabled. All noise is applied manually in `_apply_obs_noise` instead (see [Section 8](#8-observation-corruption)).

**`GateObservationConfig`:**
- `num_next_gates: int = 3` — lookahead depth K
- `include_velocity_limit: bool = True` — whether `velocity_limit` obs is included (currently always included regardless of this flag; the flag is informational)

### 6.2 Action Space

Continuous 4-dimensional, each component in `[-1, 1]`:
- `a[0]` — collective thrust (mapped to `[0, 1]` via `(a+1)/2`)
- `a[1]` — roll moment
- `a[2]` — pitch moment
- `a[3]` — yaw moment

### 6.3 Reward Terms

All functions return an **unweighted** scalar per environment `[N]`. The `RewardManager` multiplies by `weight`. SKRL further scales by `rewards_shaper_scale=0.01`.

**Effective reward = function output × WEIGHT_* × 0.01 = function output × DIRECT_*.**

| Term | Function | DIRECT weight | WEIGHT (×100) | Description |
|------|----------|--------------|---------------|-------------|
| `gate_passage` | `reward_gate_passage` | +250.0 | +25000 | Binary 1.0 on gate crossing frame |
| `gate_progress` | `reward_gate_progress` | +20.0 | +2000 | Distance reduction toward current gate (clamped ±1 m) |
| `gate_centering` | `reward_gate_centering` | +0.5 | +50 | Exponential alignment bonus on approach side |
| `velocity_toward_gate` | `reward_velocity_toward_gate` | +2.0 | +200 | Velocity projection onto gate direction, approach side only |
| `velocity_limit` | `penalty_velocity_limit` | −0.05 | −5 | Quadratic penalty for exceeding per-env speed cap |
| `orientation` | `penalty_orientation` | −0.005 | −0.5 | Squared tilt angle from upright |
| `angular_velocity` | `penalty_angular_velocity` | −0.001 | −0.1 | Sum of squared body angular rates |
| `action_smoothness` | `penalty_action_smoothness` | −0.0001 | −0.01 | Squared action deltas between steps |
| `action_magnitude` | `penalty_action_magnitude` | −0.001 | −0.1 | Squared action magnitudes |
| `crash` | `penalty_crash` | −50.0 | −5000 | Binary 1.0 on collision OR upside-down |
| `wrong_side` | `penalty_wrong_side` | −0.1 | −10 | 1.0 when drone is on the exit side of the gate |

**`gate_centering` formula:**
```
centering_value = exp(-3 * lateral_dist / gate_size)
proximity_factor = exp(-0.5 * |dist_to_plane|)
reward = centering_value × proximity_factor × on_approach_side
```
Active only while `dist_to_plane <= 0` (approach side), preventing spurious centering reward behind the gate.

**`wrong_side` note:** fires every step the drone is on the exit side without having passed. On a chicane track where the next gate faces the opposite direction, the drone will briefly be on the wrong side as it turns; the penalty is mild (−0.1 effective weight) but can discourage approach.

**Unused term:** `penalty_perception` (camera FOV penalty) is defined but commented out in `RewardsCfg`.

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
4. Observation/action noise biases resampled in `_reset_idx` (after super)

**Spawn modes** (set by `method` parameter in `reset_racing_spawn`):

| Mode | Description |
|------|-------------|
| `"ground"` | 2 m behind a random gate, heading directly at it, zero velocity |
| `"uniform"` | Box on approach side: depth [1,4] m, ±1.5 m lateral, ±0.75 m vertical, random yaw, small random velocity |
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

**Velocity limit resampling:** if `randomize_velocity_limit=True`, `_velocity_limit[env_ids]` is resampled uniformly from `velocity_limit_range` each episode.

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
- Cubic spline with random control points — 16–30 control points at difficulty 0.7–1.0
- Chicane — alternating lateral offsets producing tight S-curves; more sections at higher difficulty

**Gate placement:**
- Spacing: `gate_spacing_range = (1.0, 7.0)` m → average 4.0 m → ~10–16 gates per circuit
- Gate size: `gate_size_range = (0.45, 0.8)` m; interpolated as `max - difficulty × (max - min)`
  - difficulty=0.7 → size ≈ 0.69 m
  - difficulty=1.0 → size = 0.45 m
- Gate height: `gate_height_range = (0.7, 4.5)` m, oscillating with difficulty-scaled amplitude

**Out-of-bounds gates:** gates whose centre is closer than 0.5 m to any tile edge are skipped (the arc position still advances, so gate count may be lower than the computed upper bound).

### 7.2 Fixed Track (Deterministic)

`create_fixed_track_terrain(gates, ...)` tiles the same circuit in all 16 cells. Gate positions are defined via `GateSpec`:

```python
@dataclass
class GateSpec:
    position: tuple[float, float, float]   # corner-origin local frame (0-15 m)
    yaw_deg: float = 0.0                   # CW from +Y viewed from above
```

The default `FIXED_GATES` (10-gate circuit in `racing_quadcopter_env_cfg.py`) includes a split-S maneuver (gates 4-5, 2.7 m altitude drop) and a chicane (gates 8-9).

An alternative `FIXED_GATES_` (4-gate square) is also defined for debugging.

### 7.3 Gate Coordinate Convention

Gates use a right-handed local frame:
- **+X** = right (horizontal, along bar width)
- **+Y** = forward = **passage direction** (drone flies through in +Y)
- **+Z** = up (always world up — gates never tilt)

`yaw_deg` in `GateSpec`: **clockwise from +Y** viewed from above:
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

Configured via `ObservationNoiseCfg` in `env_cfg`. Applied in `_apply_obs_noise()`.

All noise is applied manually by the custom `wrap_*` functions and `compute_next_gates_rel_and_fwd_body`, bypassing IsaacLab's built-in `ObsTerm.noise` pipeline (which is disabled by `enable_corruption=False`).

**Three corruption types (applied in order):**

1. **Per-episode bias** (if `per_episode_bias=True`): a bias is sampled uniformly from `bias_range_<term>` at each reset and added every step. Simulates persistent sensor calibration errors.

2. **Per-step Gaussian noise**: additive `N(0, gaussian_std_<term>²)` noise each step.

3. **Dropout** (`dropout_prob`): entire term zeroed with probability `p` each step, independently per env. Applied as a binary mask over the first dimension.

**Per-term standard deviations (training defaults):**

| Term | `gaussian_std` | `bias_range` |
|------|----------------|--------------|
| `base_lin_vel` | 0.02 m/s | ±0.05 m/s |
| `base_ang_vel` | 0.02 rad/s | ±0.02 rad/s |
| `projected_gravity` | 0.01 | ±0.01 |
| `obs_gates` | 0.02 (scaled units) | ±0.1 (scaled units ≈ ±1 m) |
| `velocity_limit` | 0.0 | ±0.5 m/s |
| `last_action` | 0.0 | ±0.05 |

All noise is disabled in Play configs (`observation_noise.enable=False`).

---

## 9. Action Corruption

Configured via `ActionNoiseCfg` in `env_cfg`. Applied in `QuadcopterThrustAction.process_actions()`.

**Three corruption types:**

1. **Per-step Gaussian noise** (`gaussian_std=0.015`): additive noise on all 4 action channels.

2. **Per-episode bias** (`per_episode_bias=True`, `bias_range=(-0.05, 0.05)`): constant offset per episode, sampled at reset. Simulates motor trim offsets.

3. **First-order low-pass filter** (`lpf_alpha=0.9`):
   ```
   cmd_filtered = alpha × cmd_new + (1 - alpha) × cmd_prev
   ```
   `alpha=0.9` → ~10% of old command retained each step. Simulates actuator bandwidth limitation. `alpha=1.0` = no filtering.

4. **Clipping** (`clip_actions=True`): after all corruption, actions are re-clamped to `[-1, 1]`.

Applied corruption order: Gaussian → bias → LPF → clip. The `_prev_actions` buffer is read before updating so the LPF uses the previous **post-corruption** command.

All corruption disabled in Play configs (`action_noise.enable=False`).

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
[EVAL Stats | n=NNN] Speed: X.X km/h  |  Gates/ep: X.X  |  Lap%: XX.X%  |  Lap time: X.X s
```

`n` = number of entries in `_run_window` at print time. The mode prefix (`TRAIN`, `EVAL`, or `RUN`) is set by the `RACING_RUN_MODE` environment variable, which train.py sets to `TRAIN` and play.py sets to `EVAL`.

### TensorBoard / WandB

`extras["log"]` dictionary is populated each reset with:
- `Episode_Metrics/gates_passed` — mean gates passed over reset batch
- `Episode_Metrics/mean_velocity` — mean drone speed over episode (m/s)
- `Episode_Metrics/collision_count` — mean non-timeout terminations per episode
- `Episode_Metrics/mean_lap_time_s` — mean lap duration (only when laps completed)
- `Curriculum/stage` — current stage (0, 1, 2, or 1000)

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
| `rewards_shaper_scale` | 0.01 | Divides all rewards by 100 |
| `observation_preprocessor` | `RunningStandardScaler` | Online obs normalization |
| `value_preprocessor` | `RunningStandardScaler` | Online value normalization |
| `timesteps` | 20000 | Default total env steps (override with `--max_iterations`) |

**Checkpoint path:** `logs/skrl/racing_quadcopter/<timestamp>_ppo_torch/checkpoints/`

### 11.2 RSL-RL PPO Configuration

File: `agents/rsl_rl_ppo_cfg.py`

A secondary trainer option (not used in sweep). Smaller network (128 × 64) and different step count (`num_steps_per_env=48`, `max_iterations=10000`).

### 11.3 Reward Scaling Chain

```
function_output × WEIGHT_* × rewards_shaper_scale
= function_output × (MANAGER_SCALE × DIRECT_*) × 0.01
= function_output × (100 × DIRECT_*) × 0.01
= function_output × DIRECT_*
```

This ensures the effective reward magnitude matches the direct-env defaults. When sweeping `rewards.<term>.weight`, multiply the desired "direct" coefficient by 100 (or use the `DIRECT_` sweep shorthand — see [Section 13.3](#133-direct_-shorthand)).

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
- Nested: `"observation_noise.gaussian_std_obs_gates"` → sets `env_cfg.observation_noise.gaussian_std_obs_gates`
- Tuple strings: `"(8.0, 15.0)"` is parsed by `ast.literal_eval` and coerced to tuple

### 13.5 DIRECT_ Shorthand

Keys starting with `DIRECT_` are converted to manager-scale `rewards.<term>.weight`:

```yaml
param_grid:
  DIRECT_VELOCITY_TOWARD_GATE: [0.1, 2.0]
# → --override rewards.velocity_toward_gate.weight=10.0  (0.1 × 100)
#   --override rewards.velocity_toward_gate.weight=200.0 (2.0 × 100)
```

`MANAGER_SCALE` is read directly from `rewards.py` using regex, without importing the Racing_Quadcopter package, to avoid triggering Omniverse extension loading in the sweep script itself.

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
--override velocity_limit_range="(8.0, 15.0)"
--override rewards.crash.weight=-8000.0
--override observation_noise.gaussian_std_obs_gates=0.05
--override recovery_spawn_prob=0.3
--override randomize_velocity_limit=True
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
| `scene.num_envs` | 1024 | Number of parallel environments |
| `scene.env_spacing` | 5.0 m | Spacing between env origins |
| `scene.replicate_physics` | True | Use PhysX's scene replication |

### Physics limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_height` | 0.2 m | Below this → `below_height_threshold` termination |
| `max_velocity` | 50.0 m/s | Above this → `runaway_velocity` termination |
| `collision_force_threshold` | 0.001 N | Minimum contact force for collision detection |
| `upside_down_threshold` | 0.7 | `projected_gravity_b[2]` above this → upside-down termination (stage 0 only) |
| `crash_termination` | True | Master switch for collision termination |

### Velocity limit

| Parameter | Default | Description |
|-----------|---------|-------------|
| `velocity_limit` | 15.0 m/s | Fixed per-env speed cap when not randomizing |
| `randomize_velocity_limit` | False | Re-sample cap each episode from range |
| `velocity_limit_range` | (8.0, 15.0) | Uniform sampling range when randomizing |

### Spawn

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fixed_spawn` | False | True → 8-lane grid at gate 0 (Play configs) |
| `ground_init_prob` | 0.5 | Fraction using ground spawn in "mixed" mode |
| `recovery_spawn_prob` | 0.2 | Fraction overwritten with random attitude in stage 2 |

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

Functions in `vis/trajectory_vis.py`:

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
- Per-term unweighted output (mean), weight, and weighted mean for all 11 reward terms
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

- **`wrong_side` on chicane tracks:** if the next gate faces opposite to the drone's direction of travel (common on chicane/random tracks), the drone will briefly be on the exit side as it turns. The penalty is mild (DIRECT weight −0.1) but can create a small adverse gradient.

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

- **Contact sensor update period:** `update_period=0.01 s` matches the 50 Hz physics rate. At `decimation=2`, the sensor refreshes twice per RL step; `net_forces_w_history` with `history_length=2` captures both sub-steps.

# Codebase index — start here

This repository is an Isaac Lab extension for training quadcopter policies with reinforcement
learning. It contains three tasks: **racing** (fly through gates in order), **hovering**
(reach and hold waypoints) and **AI-GP** (the AI Grand Prix competition course, frozen at the
version we competed with plus a frozen copy of the later experiments). This page explains how the code is organized and where to find things. Every
other document is linked from [§8](#8-all-documentation).

## Contents

1. [The short version](#1-the-short-version)
2. [Where to look for a given job](#2-where-to-look-for-a-given-job)
3. [How a task is put together](#3-how-a-task-is-put-together)
4. [The three tasks](#4-the-three-tasks)
5. [What is in racing_quadcopter](#5-what-is-in-racing_quadcopter)
6. [Repository map](#6-repository-map)
7. [Configuration rules](#7-configuration-rules)
8. [All documentation](#8-all-documentation)
9. [Glossary](#9-glossary)
10. [Leftovers and non-working pieces](#10-leftovers-and-non-working-pieces)

---

## 1. The short version

- All the code is one Isaac Lab extension, `source/Racing_Quadcopter` (Python package
  `Racing_Quadcopter`), installed with `pip install -e source/Racing_Quadcopter`. Importing it
  registers every task ID with Gym.
- Each task lives in its own package under
  `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/`:
  `racing_quadcopter/`, `hovering_quadcopter/` and `ai_gp_quadcopter/`.
- Racing and hovering share one library of MDP terms (`tasks/manager_based/mdp/`: observations,
  actions, rewards, events, terminations, curricula) and the robot assets. AI-GP consists of
  frozen, self-contained copies and shares nothing.
- You train with `scripts/skrl/train.py` or `scripts/rsl_rl/train.py --task <task ID>`.
  **Environment variables** choose the variant: drone model, action type, observation space,
  reward preset, track.

```bash
# racing
DRONE_MODEL=anemos_urdf python scripts/skrl/train.py --task Isaac-Racing-Quadcopter-Manager-Fixed-v0 --headless
# hovering
python scripts/rsl_rl/train.py --task Isaac-Hovering-Quadcopter-Manager-RNN-v0 --headless
# AI-GP (frozen June 11 version; AI_GP_VARIANT=experimental for the later experiments)
python scripts/skrl/train.py --task Isaac-AI-GP-Quadcopter-v0 --headless
```

To watch a trained policy, run the same command with `play.py`, the `-Play-v0` task ID, the
**same environment variables**, and `--checkpoint <path>`.

## 2. Where to look for a given job

| I want to… | Start at | Then |
|---|---|---|
| Install and run something | [README.md](../README.md) | [§7](#7-configuration-rules) |
| Understand the racing environment end to end | [CODEBASE.md](CODEBASE.md) | `racing_quadcopter/racing_quadcopter_env.py` |
| Change racing rewards | `RewardsCfg` + `MayRewardWeights` in `racing_quadcopter/racing_quadcopter_env_cfg.py` | term functions in `mdp/rewards.py` |
| Change what the policy observes | `ObservationsCfg` in the task's `*_env_cfg.py` | `mdp/observations.py`, [OBSERVATION_SPACE.md](../source/Racing_Quadcopter/docs/OBSERVATION_SPACE.md) |
| Change how actions become motor forces | `ActionsCfg` in the task's `*_env_cfg.py` | `mdp/actions.py` |
| Add or edit a racing track | `configs/tracks/*.yaml` + `TRACK_YAML=...` | [scripts/custom_track/README.md](../scripts/custom_track/README.md) (GUI designer) |
| Work on hovering / waypoint following | [hovering_quadcopter/SPEC.md](hovering_quadcopter/SPEC.md) | `hovering_quadcopter/hovering_quadcopter_env_cfg.py` |
| Run the AI-GP task | [ai_gp_quadcopter/README.md](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/ai_gp_quadcopter/README.md) | — (frozen; do not edit without intent) |
| Add noise / domain randomization | [README_randomization.md](README_randomization.md) | `mdp/noise.py`, `mdp/events.py`, `mdp/shared_features.py` |
| Test robustness on special tracks | [skill_testing_terrains.md](skill_testing_terrains.md) | `RacingQuadcopterSkillEvalRLEnv` / `RacingQuadcopterSkillEvalEnvCfg` in `racing_quadcopter/racing_quadcopter_env.py` / `_env_cfg.py` |
| Run parameter sweeps (local or AWS) | [CODEBASE.md §13](CODEBASE.md#13-sweep-system), [AWS_BATCH_USAGE.md](AWS_BATCH_USAGE.md) | `scripts/sweep.py`, `configs/sweeps/` |
| Replay real-flight actions in sim | [DEPLOYMENT.md](DEPLOYMENT.md) | `scripts/deployment/` |
| Change the drone model | `assets/robots_cfg.py` + the task's drone profiles | `tasks/manager_based/drone_profile.py` |
| Know the current state of the refactor and open issues | [HANDOFF.md](../HANDOFF.md) | — |

Paths without a prefix are relative to `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/`,
or to `source/Racing_Quadcopter/Racing_Quadcopter/` for `assets/` and `terrains/`.

## 3. How a task is put together

Isaac Lab "manager-based" environments separate **runtime code** from **configuration**. A
task ID ties the pieces together in the package's `__init__.py`:

```
--task Isaac-Racing-Quadcopter-Manager-Fixed-v0
  │   registered by gym.register(...) in racing_quadcopter/__init__.py
  │
  ├─ entry_point            → ENV CLASS (runtime): RacingQuadcopterRLEnv
  │                           racing_quadcopter_env.py — buffers, gate tracking, step loop, logging
  │
  ├─ env_cfg_entry_point    → ENV CONFIG: RacingQuadcopterEnvCfg_FIXED
  │                           racing_quadcopter_env_cfg.py — scene, sim, and the managers:
  │                             observations / actions / rewards / events / terminations / curriculum
  │                             every term points at a function or class in mdp/
  │
  ├─ skrl_cfg_entry_point   → agents/skrl_ppo_cfg.yaml       (skrl PPO hyperparameters)
  └─ rsl_rl_cfg_entry_point → agents/rsl_rl_ppo_cfg.py       (rsl_rl PPO runner)
```

- **Env class** = how the world is simulated and tracked. It rarely changes.
- **Env config** = what the task is: which terms, weights and parameters. This is where most
  changes happen.
- **`mdp/`** = the term functions themselves (e.g. `rewards.reward_gate_passage`).
- **Variant configs** (`_PLAY`, `_FIXED`, …) are small subclasses that change a few fields in
  `__post_init__`.
- **Environment variables** are read when the config module is imported and pick the drone,
  action term, observation layout, reward preset and track. See [§7](#7-configuration-rules).

## 4. The three tasks

| | Racing | Hovering | AI-GP |
|---|---|---|---|
| Package | `racing_quadcopter/` | `hovering_quadcopter/` | `ai_gp_quadcopter/` |
| Goal | Pass gates in order as fast as possible, on random or fixed tracks | Reach waypoints and hold position | Fly the AI Grand Prix competition course: one fixed, descending 6-gate track |
| Env class | `RacingQuadcopterRLEnv` | `HoveringQuadcopterRLEnv` | A frozen copy of `RacingQuadcopterRLEnv` in each snapshot |
| Task IDs | `Isaac-Racing-Quadcopter-Manager-*` (10, see [§5](#5-what-is-in-racing_quadcopter)) | `Isaac-Hovering-Quadcopter-Manager-{v0, Play-v0, RNN-v0, RNN-Play-v0}` | `Isaac-AI-GP-Quadcopter-{v0, Play-v0, RNN-v0, RNN-Play-v0}` |
| Environment variables | `DRONE_MODEL`, `ACTION_TYPE`, `OBSERVATION_SPACE`, `RACING_REWARD_PRESET`, `TRACK_YAML` | `HOVERING_DRONE_MODEL`, `WAYPOINTS` | `AI_GP_VARIANT`, `AI_GP_TRACK_YAML` |
| Shared code used | `mdp/`, `terrains/`, `assets/`, `drone_profile.py` | `mdp/`, `assets/`, `drone_profile.py` (own `terrain.py`) | None |
| Status | Active | Active | Frozen: `june11` (default) and `experimental` |
| Main doc | [CODEBASE.md](CODEBASE.md) | [SPEC.md](hovering_quadcopter/SPEC.md) | [README.md](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/ai_gp_quadcopter/README.md) |

The task packages never import each other, and AI-GP imports nothing from the rest of the
extension at all. `tasks/manager_based/__init__.py` imports all three, which is what registers
the task IDs.

### Hovering files

| File | What it is |
|---|---|
| `hovering_quadcopter_env.py` | Env class `HoveringQuadcopterRLEnv` |
| `hovering_quadcopter_env_cfg.py` | Hovering's drone profiles (`HOVERING_DRONE_PROFILES`), managers, reward weights, `_PLAY` variant |
| `hovering_randomization.py` | Maximum noise/randomization levels for the adaptive curriculum |
| `terrain.py` | Flat terrain with one waypoint track per terrain cell |
| `vis.py` | Waypoint markers |
| `agents/` | skrl and rsl_rl configs (`train.sh` trains hovering with rsl_rl) |

### AI-GP files

`ai_gp_quadcopter/` holds two frozen snapshots of the AI Grand Prix task. `AI_GP_VARIANT` picks
which one the task IDs run:

- `june11/` (default): the version we competed with (commit `f6828d9`, 2026-06-11).
- `experimental/`: the experiments after the competition (second-order rate action, curricula,
  heavy randomization), as of 2026-09-26. Its training runs so far never completed a lap.

Each snapshot has its own env class, `mdp/`, `terrains/`, robot asset with meshes, track and
agent configs. That is why the same code appears three times (shared, `june11/`,
`experimental/`): no change elsewhere in the repo can alter a snapshot. **Do not edit a snapshot
unless you mean to change it.** The [README](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/ai_gp_quadcopter/README.md)
lists what each copy came from, what was adapted, and how the decoupling was checked.

## 5. What is in racing_quadcopter

The folder holds one runtime/config/visualization file each, plus the agent configs:

| File | What it is | Used by |
|---|---|---|
| `__init__.py` | `gym.register` for the 10 racing task IDs (below) | — |
| `racing_quadcopter_env.py` | Env classes: `RacingQuadcopterRLEnv` (gate tracking, crash detection, step loop, statistics), `RacingQuadcopterMultiTrackRLEnv` (fixed per-terrain-column tracks; overrides gate-data loading), `RacingQuadcopterSkillEvalRLEnv` (skill-eval spawn routing, per-terrain TensorBoard, observation delay) | Every racing task ID |
| `racing_quadcopter_env_cfg.py` | Every racing config class: scene, sim, the managers (observations/actions/rewards/events/terminations/curriculum), drone profiles, reward presets (`may` / `current`), the `_PLAY` / `_FIXED` / `_FIXED_PLAY` / `_FIXED_PLAY_FPV` variants, and the SkillEval config (`SkillTerrainCfg`, `RacingQuadcopterSkillEvalEnvCfg`) | Every racing task ID |
| `trajectory_vis.py` | USD `BasisCurves` / `SphereLight` visualization helpers (trajectory line, gate indicators/labels, demo-mode atmosphere) | Every racing task ID (feature-gated by cfg flags) |
| `agents/` | skrl YAML, rsl_rl PPO runner | Every racing task ID |

How the classes relate:

```
CONFIGS (racing_quadcopter_env_cfg.py)             ENV CLASSES (racing_quadcopter_env.py)

RacingQuadcopterEnvCfg  (ManagerBasedRLEnvCfg)      RacingQuadcopterRLEnv
 ├── _PLAY                                           └── RacingQuadcopterMultiTrackRLEnv
 ├── _FIXED  track from TRACK_YAML                        └── RacingQuadcopterSkillEvalRLEnv
 │    └── _FIXED_PLAY                                          runs RacingQuadcopterSkillEvalEnvCfg only
 │         └── _FIXED_PLAY_FPV  1 drone + camera
 └── RacingQuadcopterSkillEvalEnvCfg
```

**Why these are still separate files, and why SkillEval still needs its own env subclass:**

- One file for runtime code, one for configuration, one for visualization — the usual
  manager-based split ([§3](#3-how-a-task-is-put-together)). There is no longer a reason to split
  further: the env-var-free base config that used to be shared with the AI-GP experimental task
  is gone (below), and skill evaluation is small enough to live alongside normal racing in the
  same two files.
- **Skill evaluation still needs its own env class.** Normal racing uses one kind of track per
  run: random tracks, or one YAML track copied into every terrain cell. Skill evaluation puts
  *different* tracks in different columns, so the env must look up each drone's track from its
  own cell. It also adds per-column observation delay and writes per-column TensorBoard metrics
  to `logs/skill_eval/`.

History: until 2026-09-26 this folder had 7 `.py` files plus `vis/`. The experimental AI-GP
config lived here, and `gate_racing_base_cfg.py` held the config it shared with racing. Once
that config moved to its own frozen copy (`ai_gp_quadcopter/experimental/`), the base config,
`racing_randomization.py` and the separate SkillEval files were merged into the two main files,
and `vis/trajectory_vis.py` moved up to `trajectory_vis.py`.

**Racing task IDs** (all in `racing_quadcopter/__init__.py`):

| Task ID | Config class | Use |
|---|---|---|
| `…-Manager-v0` | `RacingQuadcopterEnvCfg` | Train on random procedural tracks (different in every terrain cell) |
| `…-Manager-Play-v0` | `…_PLAY` | Evaluate on random tracks |
| `…-Manager-Fixed-v0` | `…_FIXED` | Train on one track from `TRACK_YAML` (default `configs/tracks/default.yaml`) |
| `…-Manager-Fixed-Play-v0` | `…_FIXED_PLAY` | Evaluate on that track |
| `…-Manager-Fixed-Play-FPV-v0` | `…_FIXED_PLAY_FPV` | One drone with an onboard camera view |
| `…-Manager-SkillEval-v0` | `RacingQuadcopterSkillEvalEnvCfg` | Robustness evaluation on the four `configs/skill_tracks/` |
| `…-Manager-RNN-v0`, `-RNN-Play-v0`, `-Fixed-RNN-v0`, `-Fixed-RNN-Play-v0` | same as the non-RNN IDs | Same env with a GRU policy. **rsl_rl only**: with skrl these IDs use the normal MLP config |

## 6. Repository map

```
.
├── README.md                   setup, train/play commands, environment variables
├── HANDOFF.md                  state of the Sep 2026 audit/refactor, open issues
├── train.sh, play.sh, fpv_play.sh        wrappers; edit the variables at the top
├── tensorboard.sh, tensorboard_hovering.sh
├── Makefile                    AWS Batch sweep pipeline (docs/AWS_BATCH_USAGE.md)
├── .github/workflows/regression.yml      CI: regression sweeps on AWS for every push to main
├── configs/
│   ├── tracks/                 racing tracks for the Fixed tasks: default, simple, wide; ai_gp (reference copy)
│   ├── skill_tracks/           the four SkillEval tracks
│   ├── waypoint_tracks/        hovering waypoint tracks (WAYPOINTS=...)
│   └── sweeps/                 sweep configs for scripts/sweep.py
│       └── single/regression_test/       the CI regression configs ("DO NOT TOUCH")
├── scripts/
│   ├── skrl/                   train.py, play.py, override_utils.py (--override key=value)
│   ├── rsl_rl/                 train.py, play.py, cli_args.py
│   ├── sweep.py                train + evaluate over a parameter grid, write a results CSV
│   ├── list_envs.py, random_agent.py, zero_agent.py
│   ├── custom_track/           Qt track designer; play a checkpoint on a designed track
│   ├── deployment/             replay recorded real-flight actions in sim; plot sim vs. real
│   └── tools/                  generate_anemos_usd.py, view_drone.py, design_rate_pd_gains.py, …
├── docs/                       this page and the documents in §8
├── logs/                       runs: logs/skrl/<directory>/, logs/rsl_rl/<experiment_name>/, logs/skill_eval/
├── outputs/                    Hydra output folders, one per day
└── source/Racing_Quadcopter/   the Isaac Lab extension
    ├── config/extension.toml
    ├── docs/OBSERVATION_SPACE.md
    └── Racing_Quadcopter/
        ├── assets/
        │   ├── robots_cfg.py            Anemos articulation configs (URDF, USD, SITL URDF)
        │   └── robots/anemos/           anemos.urdf, anemos_sitl.urdf, meshes/
        ├── terrains/                    gate meshes, random and fixed track terrains, track YAML I/O
        └── tasks/manager_based/
            ├── __init__.py              imports the three task packages (registers task IDs)
            ├── mdp/                     shared MDP terms (table below)
            ├── drone_profile.py         DroneProfile: the fields a drone profile has; values live in each task
            ├── action_diagnostics.py    optional action statistics logging
            ├── racing_quadcopter/       §5
            ├── hovering_quadcopter/     §4
            └── ai_gp_quadcopter/        AI-GP: frozen june11/ + experimental/ snapshots (§4)
```

The Crazyflie drone (`DRONE_MODEL=crazyflie`) comes from `isaaclab_assets`, not from this repo.

### `mdp/` modules

Most modules have a gate-racing section followed by a waypoint (hovering) section.

| Module | Contents |
|---|---|
| `actions.py` | Action terms: `QuadcopterRatesAction` (body rates + thrust through a rate controller; default), `QuadcopterMomentsAction` (direct torques). `QuadcopterSecondOrderRatesAction` was removed from here on 2026-09-26 as dead code (unused by any active task); the AI-GP experimental snapshot's own `mdp/actions.py` keeps a copy, which it still uses |
| `observations.py` | Observation terms: gate-relative observations, `mono_race` layout, waypoint observations |
| `rewards.py` | Reward and penalty terms |
| `terminations.py` | Termination terms |
| `events.py` | Spawn/reset logic and domain-randomization events (mass, inertia, center of mass, pushes) |
| `gate_utils.py` | Gate frames, gate-passage detection, the per-step racing state update (including the crash flag) |
| `curriculums.py` | Racing's staged curriculum (when collisions start to end episodes) |
| `adaptive_curriculums.py`, `route_progress.py` | Curricula that raise noise and randomization as the policy makes progress along the route (hovering; racing `current` preset) |
| `shared_features.py` | Action noise config, speed limits, action-magnitude penalty |
| `noise.py` | Observation noise models |
| `stats.py` | Periodic console statistics |
| `waypoint_io.py`, `waypoint_utils.py` | Hovering waypoint YAML loading and per-step tracking |

## 7. Configuration rules

1. **Environment variables pick the variant** and are read when the config module is imported.
   The full table is in [README.md](../README.md#key-environment-variables); [§4](#4-the-three-tasks)
   shows which task reads which variable.
2. **`--override key=value` (skrl scripts) and Hydra `env.*=` arguments are applied after the
   config is built.** They cannot change anything decided in `__post_init__` or at import time:
   drone model, track and terrain, randomization on/off flags, camera resolution. Use the
   environment variable, or edit the config class.
3. **Train and play with the same environment variables.** The reward preset and observation
   space change the observation size (e.g. racing `may` = 34 values, `current` = 36), so a
   checkpoint only loads into a matching environment.
4. **Logs:** skrl writes to `logs/skrl/<directory>/`, rsl_rl to `logs/rsl_rl/<experiment_name>/`.
   Each task has its own directory name: racing `racing_quadcopter` / `racing_quadcopter_manager`,
   hovering `hovering_quadcopter` / `hovering_quadcopter_manager`, AI-GP `ai_gp_jun11`
   (experimental: `ai_gp_experimental`); RNN runs add `_gru`.
5. `DRONE_MODEL=anemos_usd` (and `HOVERING_DRONE_MODEL=anemos_usd`) need the USD file created
   first with `scripts/tools/generate_anemos_usd.py`.

## 8. All documentation

| Document | Covers |
|---|---|
| [README.md](../README.md) | Setup, train/evaluate commands, environment variables |
| [docs/INDEX.md](INDEX.md) | This page: orientation and map |
| [docs/CODEBASE.md](CODEBASE.md) | Full reference for the **racing** env: architecture, physics, MDP terms, terrain, logging, training configs, sweeps |
| [docs/README_randomization.md](README_randomization.md) | Observation noise, action noise and dynamics randomization: what exists, defaults, how to enable |
| [docs/hovering_quadcopter/SPEC.md](hovering_quadcopter/SPEC.md) | Hovering task specification: actions, observations, rewards, waypoint tracks |
| [docs/skill_testing_terrains.md](skill_testing_terrains.md) | SkillEval: the four robustness tracks and how to add one |
| [docs/DEPLOYMENT.md](DEPLOYMENT.md) | Replaying recorded real-flight actions in sim and comparing trajectories |
| [docs/AWS_BATCH_USAGE.md](AWS_BATCH_USAGE.md) | Running sweeps on AWS Batch with the Makefile |
| [OBSERVATION_SPACE.md](../source/Racing_Quadcopter/docs/OBSERVATION_SPACE.md) | The `mono_race` observation layout and the MonoRace / Swift reference observation spaces |
| [scripts/custom_track/README.md](../scripts/custom_track/README.md) | Track YAML format, the GUI track designer, testing a policy on a custom track |
| [ai_gp_quadcopter/README.md](../source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/ai_gp_quadcopter/README.md) | The AI-GP task: its two frozen snapshots, how to run them, provenance, decoupling checks, known behaviors |
| [HANDOFF.md](../HANDOFF.md) | The Sep 2026 audit and refactor: what changed, what was verified, open decisions |

## 9. Glossary

| Term | Meaning |
|---|---|
| Manager-based env | Isaac Lab style where a task is a config of terms (observations, rewards, …) run by managers |
| Task ID | The Gym name passed to `--task`; it selects the env class, env config and agent configs |
| Play | Evaluation config: `eval_mode` on, fixed spawn at gate 0, randomization off, fewer drones |
| Fixed | One track read from a YAML file, identical in every terrain cell (vs. random procedural tracks) |
| RNN | Same task with a GRU policy (rsl_rl runner) |
| FPV | One drone with an onboard camera |
| SkillEval | Robustness evaluation on four special tracks at once |
| Drone profile | Mass, arm geometry, rate-controller gains etc. for one drone, chosen by `DRONE_MODEL` / `HOVERING_DRONE_MODEL` |
| Rates / moments | Action types (`ACTION_TYPE`): the policy commands thrust + body rates (tracked by a simulated rate controller) or thrust + body torques directly |
| `ours` / `mono_race` | Racing observation layouts (`OBSERVATION_SPACE`) |
| Reward preset `may` / `current` | Racing reward sets (`RACING_REWARD_PRESET`). `may` (default) is the May 2026 set that trains reliably; `current` is the later set with a strong action penalty, speed limits and adaptive curricula |
| AI-GP | AI Grand Prix, the drone-racing competition we entered; the `ai_gp_quadcopter` task |
| `june11` / `experimental` | AI-GP variants (`AI_GP_VARIANT`): frozen June 11 snapshot (default) or the frozen post-June experiments |
| DR | Domain randomization: randomized mass, inertia, center of mass, pushes, noise |
| Curriculum stage | Racing's step-based schedule for when collisions start to end episodes |
| Adaptive curriculum | Noise and randomization that grow with route progress |
| `eval_mode` | Evaluation switch: disables training-only curricula and always ends episodes on collision |

## 10. Leftovers and non-working pieces

- `ui_extension_example.py` (package root) is the Isaac Lab extension template's example UI.
  It is imported by the package but has nothing to do with the tasks.
- `assets/robots/anemos/incomplete_anemos.urdf` is not referenced anywhere.
- `scripts/tools/check_sitl_parity_static.py` and several sweep configs are marked
  **NON-WORKING** or **STALE** in their headers. Read the header before running any script or
  config you have not used before.
- The Betaflight SITL surrogate action (`BetaflightSitlSurrogateActionCfg` in `mdp/actions.py`)
  is unstable at the simulation rate and unused.
- Some settings only look configurable: a few randomization flags and the FPV camera resolution
  cannot be changed by `--override` (see [§7](#7-configuration-rules)). The full list is in
  [HANDOFF.md §2.5](../HANDOFF.md#25-marked-non-working--note-commented-not-fixed).

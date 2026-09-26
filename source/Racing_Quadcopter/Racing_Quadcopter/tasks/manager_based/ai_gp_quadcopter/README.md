# AI-GP task — two frozen snapshots

The AI Grand Prix (AI-GP) competition course: 6 gates, about 136 m long with a 26 m descent,
flown with our competition drone. This package holds two **frozen, self-contained** snapshots of
the task:

| Folder | What it is | Selected by |
|---|---|---|
| `june11/` | The version we competed with: commit `f6828d9` (2026-06-11). | `AI_GP_VARIANT=june11` (default) |
| `experimental/` | The experiments after the competition (commits `b1dd396` … `3bde690`, Jun 29 – Aug 24): second-order rate action, route-progress curricula, heavy domain randomization. Snapshot of 2026-09-26. | `AI_GP_VARIANT=experimental` |

**Frozen** means: change a snapshot only when you explicitly intend to change that snapshot.
Racing, hovering and the shared code (`tasks/manager_based/mdp`, `terrains/`, `assets/`) can
change freely without affecting either one.

## Running

```bash
# train (June 11)
python scripts/skrl/train.py --task Isaac-AI-GP-Quadcopter-v0 --headless
# train (experimental)
AI_GP_VARIANT=experimental python scripts/skrl/train.py --task Isaac-AI-GP-Quadcopter-v0 --headless
# evaluate
python scripts/skrl/play.py --task Isaac-AI-GP-Quadcopter-Play-v0 --checkpoint <path>
```

| Task ID | Use |
|---|---|
| `Isaac-AI-GP-Quadcopter-v0` | Train |
| `Isaac-AI-GP-Quadcopter-Play-v0` | Evaluate |
| `Isaac-AI-GP-Quadcopter-RNN-v0`, `-RNN-Play-v0` | Same env with a GRU policy. rsl_rl only: with skrl these use the MLP config. Added after June 11 (Jun 29) and kept as an experiment. |

| Environment variable | Meaning |
|---|---|
| `AI_GP_VARIANT` | `june11` (default) or `experimental`. Anything else raises an error. |
| `AI_GP_TRACK_YAML` | Optional track override; default is the snapshot's own `tracks/competition.yaml`. |

Racing's variables (`DRONE_MODEL`, `TRACK_YAML`, `RACING_REWARD_PRESET`, …) are never read.

| Logs | rsl_rl `experiment_name` | skrl `directory` |
|---|---|---|
| june11 | `ai_gp_jun11`, `ai_gp_jun11_gru` | `ai_gp_jun11` |
| experimental | `ai_gp_experimental`, `ai_gp_experimental_gru` | `ai_gp_experimental` |

The names are unique so that rsl_rl play/resume, which pick the newest run in a directory, can
never load a racing checkpoint.

### Renamed from "competition" (2026-09-26)

| Old | New |
|---|---|
| package `competition_quadcopter/` (June 11) | `ai_gp_quadcopter/june11/` |
| `racing_quadcopter/competition_env_cfg.py` + `competition_randomization.py` + `agents/*competition*` (experimental) | `ai_gp_quadcopter/experimental/` |
| `Isaac-Racing-Quadcopter-Competition-{v0,Play-v0,RNN-v0,RNN-Play-v0}` | `Isaac-AI-GP-Quadcopter-{v0,Play-v0,RNN-v0,RNN-Play-v0}` |
| `COMPETITION_VARIANT`, `COMPETITION_TRACK_YAML` | `AI_GP_VARIANT`, `AI_GP_TRACK_YAML` |
| logs `competition_jun11*`, `competition_experimental*` | `ai_gp_jun11*`, `ai_gp_experimental*` |
| `configs/tracks/competition.yaml` | `configs/tracks/ai_gp.yaml` (reference copy only; each snapshot reads its own) |

Earlier experimental runs are in `logs/skrl/competition_quadcopter/` (Sep 24). To play them,
use `AI_GP_VARIANT=experimental` and pass `--checkpoint`.

## How the snapshots are decoupled

Each snapshot has its own copy of everything it runs:

```
<snapshot>/
  competition_env.py        env class (RacingQuadcopterRLEnv, frozen copy)
  competition_env_cfg.py    task config: CompetitionEnvCfg, CompetitionEnvCfg_PLAY
  mdp/                      MDP terms
  terrains/                 gate meshes and track terrain
  assets/                   robots_cfg.py, anemos.urdf, meshes/
  tracks/competition.yaml   the course
  agents/                   skrl and rsl_rl configs
  vis/                      visualization
  (june11 only)        base_env_cfg.py: the June racing config that CompetitionEnvCfg subclassed
  (experimental only)  gate_racing_base_cfg.py, competition_randomization.py, drone_profile.py,
                       action_diagnostics.py
```

Neither snapshot imports anything from the rest of `Racing_Quadcopter` or from the other
snapshot. `__init__.py` references the selected snapshot by string, so the other one is never
imported. This was checked three ways:
1. No `Racing_Quadcopter.*` import in either snapshot points outside that snapshot.
2. At runtime, after building and stepping each variant (train and play), no `Racing_Quadcopter`
   module outside the snapshot was loaded (only the package roots that register task IDs).
3. Those runs had every racing and old competition variable set to invalid values.

One link remains: `tasks/manager_based/__init__.py` imports all three task packages to register
them, so an import error in the racing or hovering `__init__.py` would also stop AI-GP from
registering.

## june11 — provenance

Extracted from `f6828d9` with only these substitutions:

| File here | June 11 source | Changes |
|---|---|---|
| `competition_env_cfg.py` | `racing_quadcopter/competition_env_cfg.py` | import paths; track path → `tracks/competition.yaml`; track variable → `AI_GP_TRACK_YAML` |
| `base_env_cfg.py` | `racing_quadcopter/racing_quadcopter_env_cfg.py` | import paths; `DRONE_MODEL` / `ACTION_TYPE` / `OBSERVATION_SPACE` pinned to the June defaults (`crazyflie` / `moments` / `ours`; the competition config overrides robot, actions and observations); unused racing subclasses (`_PLAY`, `_FIXED`, `_FIXED_PLAY`) removed |
| `competition_env.py` | `racing_quadcopter/racing_quadcopter_env.py` | import paths |
| `terrains/racing_terrain.py`, `assets/robots_cfg.py` | same files | one import path / the asset directory |
| `agents/skrl_ppo_competition_cfg.yaml`, `agents/rsl_rl_ppo_cfg.py` | same files | log directory names only |
| `agents/rsl_rl_ppo_rnn_cfg.py` | — | post-June addition (GRU runner) |

Everything else is byte-identical to `f6828d9`: all of `mdp/`, `vis/`, the other terrain files,
`anemos.urdf`, the meshes and the track.

### Known June 11 behaviors kept as-is

These were fixed elsewhere later but are deliberately **not** changed here:

- The origin (competition-start) spawn copies the gate-0 quaternion, so the drone's nose points
  along the gate's lateral axis (90° off down-track).
- The crash penalty fires on contact only (upside-down terminates without the crash penalty).
- `randomize_robot_actuator_gains` is a no-op (the actuators have zero stiffness/damping; thrust
  and torque are applied as a body wrench), and COM randomization has no dynamic effect (the
  wrench is applied at the COM).
- The DR flags (`domain_randomization`, `randomize_*`, `observation_noise_enable`) are consumed in
  `__post_init__`, so `--override` / Hydra `env.*` cannot change them; use a subclass default
  (as `CompetitionEnvCfg_PLAY` does).

## experimental — provenance

A copy of the working tree on 2026-09-26, after the September audit fixes (crash penalty
includes upside-down, eval always terminates on collision, origin-spawn nose direction fixed,
symmetric noise curriculum). Sources and changes:

| File here | Source | Changes |
|---|---|---|
| `competition_env_cfg.py`, `competition_randomization.py`, `gate_racing_base_cfg.py` | `racing_quadcopter/` | import paths; track path → `tracks/competition.yaml`; track variable → `AI_GP_TRACK_YAML`; module docstrings updated to the new location |
| `competition_env.py` | `racing_quadcopter/racing_quadcopter_env.py` | import paths; file renamed |
| `vis/trajectory_vis.py` | `racing_quadcopter/vis/` | none |
| `mdp/*` | `tasks/manager_based/mdp/*` | import paths. Copied whole, including the hovering waypoint modules this task does not use |
| `drone_profile.py`, `action_diagnostics.py` | `tasks/manager_based/` | none |
| `terrains/*` | `terrains/*` | import paths |
| `assets/robots_cfg.py` | `assets/robots_cfg.py` | only the URDF config kept; asset directory |
| `assets/anemos.urdf`, `assets/meshes/*`, `tracks/competition.yaml` | `assets/robots/anemos/`, `configs/tracks/` | none |
| `agents/*` | `racing_quadcopter/agents/*competition*` | log directory names; runner module docstring |

### Known behaviors

- Its training runs so far (Sep 24) never completed a lap. It uses the heavy regularization
  (×100 out-of-range action penalty, smoothness −5.0) that also stopped racing from learning.
  Racing's fix was the May reward preset; that fix was not applied here.
- With the second-order rate action, inertia, COM and torque randomization have no effect.
- DR flags are consumed in `__post_init__`, as in June 11.

## Verification (2026-09-26)

- The fully resolved env and agent configs of every task ID are identical before and after the
  move, apart from module and file paths and log names: 2,381 values for June 11 and 3,576 for
  experimental.
- Each variant was built and stepped (train, play, and RNN for June 11). Rewards are identical to
  the runs before the move.

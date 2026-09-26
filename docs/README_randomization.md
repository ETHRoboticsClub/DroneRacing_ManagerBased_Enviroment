# Randomization README (State Estimation, Action, and Dynamics)

This document explains the randomization stack implemented in this manager-based IsaacLab drone racing environment.

It covers:
- what is randomized,
- where it is implemented,
- how each knob works,
- recommended tuning practice,
- and how to evaluate robustly.

The intent follows conventional sim-to-real practice used in top autonomous racing systems:
- keep corruption physically plausible,
- separate fast noise from slow bias,
- keep train and eval configurations explicit,
- and increase difficulty progressively.

## 1) Design Goals

The policy receives near ground-truth simulator state by default. That is useful for stable learning, but can overfit to perfect sensing and perfectly repeatable actuation.

This randomization layer introduces controlled imperfection in three places:
- **state-estimation pipeline** (observation corruption),
- **control pipeline** (action corruption and lag),
- **plant parameters** (dynamics randomization).

High-level inspiration is aligned with robust racing pipelines (for example SWIFT and MonoRacer-era methodology):
- domain randomization should be **structured**, not arbitrary;
- biases should be **episode-consistent**;
- fast perturbations should be **small and frequent**;
- actuator dynamics should include **lag/noise**;
- and evaluation should be isolated from training randomness.

## 2) Implementation Map

The implementation is deliberately split into two layers:

- Shared behavior lives in `tasks/manager_based/mdp/shared_features.py` and
  `tasks/manager_based/mdp/adaptive_curriculums.py`.  Those modules allocate,
  reset, sample, observe, penalize, and progressively scale the features for
  every manager-based task.
- Each task configuration is only a client: it enables or disables the shared
  features, declares local limits/reward policy, and selects one named
  randomization profile.  Base racing and hovering default to disabled zero
  values; the AI-GP experimental snapshot opts in (with its own frozen copy of this code).

Each task keeps its profile declarations beside its task code:

- `racing_quadcopter/racing_quadcopter_env_cfg.py` (class `RacingRandomizationProfile`)
- `ai_gp_quadcopter/experimental/competition_randomization.py` (frozen snapshot)
- `hovering_quadcopter/hovering_randomization.py`

Profiles contain values and typed bindings only.  They do not implement
randomization, speed-limit, or curriculum behavior.

## 2.1 Observation Randomization (State Estimation)

### Files
- `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/mdp/noise.py`
- each task's observation configuration

### How it works
Observation corruption is applied through Isaac Lab observation-term noise models.

Corruption components:
1. **Per-episode bias** (optional): sampled once at reset and held constant for that episode.
2. **Per-step Gaussian noise**: i.i.d. additive perturbation every step.
3. **Per-step dropout** (optional): entire term for an environment may be masked by a Bernoulli gate.

The shared adaptive target builders can enable and scale compatible terms in any
task. Racing configures linear velocity, angular velocity, and gravity; hovering
configures angular velocity and gravity.

## 2.2 Action Randomization (Control Channel)

### File
- `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/mdp/actions.py`
- `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/mdp/shared_features.py`

### Where applied
In `QuadcopterRatesAction._compute_delayed_actions(...)` (default `ACTION_TYPE=rates`) or
`QuadcopterMomentsAction.process_actions(...)` (`ACTION_TYPE=moments`), before converting action
commands to thrust and moments. There is no `QuadcopterThrustAction` class.

### Operation order
Given incoming normalized action $a_t \in [-1,1]^4$:

1. Clamp raw command:
$$
a_t \leftarrow \mathrm{clip}(a_t, -1, 1)
$$

2. Add per-step Gaussian action noise (if enabled):
$$
a_t \leftarrow a_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,\sigma_a^2)
$$

3. Add per-episode bias (if enabled):
$$
a_t \leftarrow a_t + b, \quad b \sim \mathcal{U}(b_{\min}, b_{\max})
$$

4. Apply first-order low-pass (actuator lag model):
$$
a_t \leftarrow \alpha a_t + (1-\alpha)a_{t-1}
$$

5. Optional final clamp to `[-1, 1]`.

Then thrust/moment mapping proceeds using this randomized command.

This is important: randomization is done in the action term itself, so it affects the actual applied wrench, not just logging buffers.

## 2.3 Dynamics Randomization

### Files
- `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/mdp/route_progress.py`
- `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/mdp/adaptive_curriculums.py`
- `source/Racing_Quadcopter/Racing_Quadcopter/tasks/manager_based/mdp/shared_features.py`
- each task's environment configuration

### Mechanisms
1. **Startup mass randomization** (`randomize_robot_mass` EventTerm)
2. **Startup actuator gain randomization** (`randomize_robot_actuator_gains` EventTerm)
3. **Curriculum-latched speed limits** (`speed_limits.enabled`)

Disabled speed limits are a true no-op: zero buffers, no observation columns, and zero
penalties. When enabled, the policy sees both limits from episode zero; they stay at the
maxima until curriculum activation.
After activation, the reset `randomize_speed_limits` event samples both per environment and never
deactivates if later normalized route progress falls.
The AI-GP experimental reward curriculum also scales `velocity_limit` and
`angular_velocity_limit` upward on successful route-progress windows (up to 5× their
base weights). Unlike centering and action-smoothness scales, these two scales never
decrease during a later low-progress window.

Curriculum performance is continuous geometry rather than an integer gate count. The
episode's first and last drone positions are projected onto the closest points of an
ordered polyline. Racing supplies gate centers (the AI-GP experimental snapshot prepends
its official start); hovering supplies ordered target-pose centers. Covered arc length is
divided by the distance physically reachable from the episode's linear limit and full
time budget. Hovering additionally removes required pose dwell time from the travel
budget. Existing lap- and waypoint-completion metrics remain unchanged and separately logged.

## 3) Configuration Reference

Shared knobs live in `tasks/manager_based/mdp/shared_features.py`; each task owns only
its selected targets, ranges, reward weights, and opt-in flags.

## 3.1 Per-term observation noise (`mdp/noise.py`)

There is no single `ObservationNoiseCfg` dataclass. Each observation term that wants noise gets
its own `NoiseModelWithAdditiveBiasAndDropoutCfg`, built by `drone_noise_model_factory(std_noise,
std_bias, enable=False, dropout=0.0)` and passed as that `ObsTerm`'s `noise=` argument. Relevant
fields (on the built noise-model cfg, per term):

| Field | Meaning | Default |
|---|---|---|
| `enable` | Master switch for that term's corruption | `False` |
| `noise_cfg.std` | Per-step Gaussian std | term-specific (see below) |
| `bias_noise_cfg.std` | Per-episode additive-bias std | term-specific (see below) |
| `dropout` | Per-step term dropout probability | `0.0` |

Racing's `base_lin_vel` / `base_ang_vel` / `projected_gravity` / `obs_gates` terms are all
constructed with `enable` left at its default `False`. The `(std_noise, std_bias)` values passed
in (e.g. `RACING_RANDOMIZATION.base_lin_vel_noise = (0.2, 1.0)`) are the *targets* the optional
adaptive-randomization curriculum lerps `enable`/`std` toward — they have no effect while that
curriculum is disabled. `last_action` and `speed_limits` have no noise model.

## 3.2 `ActionNoiseCfg` (`mdp/shared_features.py`)

| Field | Meaning | Default |
|---|---|---|
| `enable` | Master switch for action randomization | `False` |
| `per_episode_bias` | Enable per-episode action bias | `True` |
| `gaussian_std` | Action Gaussian std | `0.0` |
| `bias_range` | Uniform action bias range | `(0.0, 0.0)` |
| `lpf_alpha` | Static LPF factor (1.0 = no lag); used only when `per_episode_lpf_alpha=False` | `1.0` |
| `per_episode_lpf_alpha` | Sample `lpf_alpha` per episode from `lpf_alpha_range` instead of using the static `lpf_alpha` | `True` |
| `lpf_alpha_range` | Per-episode LPF sampling range | `(1.0, 1.0)` |
| `clip_actions` | Clamp randomized action to `[-1, 1]` | `True` |

Every numeric default is a no-op value (`gaussian_std=0`, `bias_range=(0,0)`,
`lpf_alpha_range=(1,1)`), so setting `enable=True` alone adds no corruption — see
[§11](#11-enabling-randomization--quick-reference).

## 3.3 Existing dynamics knobs

| Field | Meaning | Default |
|---|---|---|
| `adaptive_curriculum_enabled` | Route-progress reward/noise/dynamics curriculum | Racing: depends on `RACING_REWARD_PRESET` (`False` for `may`, `True` for `current`). Hovering: `True`. |
| `speed_limits.enabled` | Add both observations and curriculum-latched randomization | Racing: depends on `RACING_REWARD_PRESET` (`False` for `may`, `True` for `current`). Hovering: `True`. |
| `speed_limits.linear_range` | Linear speed-limit sampling range (m/s) | Racing (`current`): `(2.0, 15.0)`. Hovering: `(0.2, 2.0)`. |
| `speed_limits.angular_range` | Angular speed-limit sampling range (rad/s) | Racing (`current`): `(1.0, 5.0)`. Hovering: `(0.5, 1.5)`. |

## 4) Train vs Eval Behavior

By default, general racing and fixed-track racing expose the shared terms but keep adaptive
curriculum and speed limits disabled (`RACING_REWARD_PRESET=may`, the default; setting it to
`current` enables both). Hovering enables both paired speed limits and its route-progress
curriculum by default. The AI-GP experimental snapshot likewise enables paired limits and its route-progress
curricula; its play config disables curriculum.

This explicit split is conventional and recommended:
- training with noise,
- deterministic (or separately controlled) evaluation.

## 5) Why this structure is conventional

This implementation follows a robust and conventional decomposition:
- **Sensor side**: fast noise + slow bias,
- **Control side**: noise + bias + lag,
- **Plant side**: mass/gain variation,
- **Task side**: optional speed-cap variation.

This avoids one common anti-pattern: injecting only high-frequency white noise and calling it sim-to-real.

## 6) Practical Tuning Protocol

Use a staged schedule rather than maximum randomization from step 0.

### Phase A: Stabilization
- Keep action noise mild (or off briefly).
- Keep observation Gaussian low.
- Keep dropout at `0.0`.

### Phase B: Robustness growth
- Increase the `obs_gates` and `base_lin_vel` observation-noise terms gradually (either by
  enabling `adaptive_curriculum_enabled` so they lerp toward the profile's targets, or by
  overriding e.g. `observations.policy.obs_gates.noise.noise_cfg.std` directly).
- Enable moderate action lag (`action_noise.lpf_alpha` from `1.0` toward `0.9`-`0.8`, with
  `action_noise.per_episode_lpf_alpha=False`).
- Add small per-episode biases.

### Phase C: Hardening
- Enable paired speed limits when the policy should observe episode-conditioned constraints;
  fixed maximum-limit penalties still apply before curriculum-latched sampling begins.
- Widen mass/gain distributions only after stable convergence.
- Keep distribution physically plausible.

## 7) Suggested Initial Ranges (Conservative)

These are reasonable starting points for racing. There is no flat `observation_noise.*` cfg
object — override the per-term noise model directly (e.g.
`observations.policy.base_lin_vel.noise.noise_cfg.std`) or enable `adaptive_curriculum_enabled`:
- `base_lin_vel` observation-noise std: `0.01` to `0.03`
- `base_ang_vel` observation-noise std: `0.01` to `0.03`
- `obs_gates` observation-noise std: `0.01` to `0.04`
- `action_noise.gaussian_std`: `0.005` to `0.02`
- `action_noise.lpf_alpha` (with `action_noise.per_episode_lpf_alpha=False`): `0.85` to `0.98`
- `action_noise.bias_range`: around `(-0.03, 0.03)` to `(-0.08, 0.08)`

If the policy stops learning entirely, first reduce:
- action bias amplitude,
- action noise std,
- and LPF aggressiveness.

## 8) Validation Checklist

When changing randomization, validate with ablations:

1. Baseline: no new randomization.
2. + Observation noise only.
3. + Action noise only.
4. + Dynamics randomization only.
5. Full stack.

Track:
- gates passed per episode,
- lap completion rate,
- crash rate,
- mean speed through gates,
- behavior under evaluation tasks.

If possible, keep one deterministic eval benchmark and one noisy eval benchmark.

## 9) Known Implementation Notes

1. Adaptive observation-noise targets enable both the observation group and each
   selected noise model; changing only one is insufficient.
2. Action delay buffers reserve the configured curriculum maximum while the active
   delay remains zero, allowing delay to grow safely during training.
3. The action per-episode bias tensor is sampled by the `reset_action_noise` `EventTerm` (`mode="reset"`, `mdp/shared_features.py`), and zeroed when `action_noise.enable` is `False`. Per-term observation noise models are sampled by IsaacLab's own `ObsTerm.noise` pipeline, not a shared reset event.

## 10) Usage Examples

### Train with manager task
```bash
python scripts/skrl/train.py --task Isaac-Racing-Quadcopter-Manager-v0
python scripts/rsl_rl/train.py --task Isaac-Racing-Quadcopter-Manager-v0
```

### Play/Eval with play configs (noise disabled by config)
```bash
python scripts/skrl/play.py --task Isaac-Racing-Quadcopter-Manager-Play-v0
python scripts/rsl_rl/play.py --task Isaac-Racing-Quadcopter-Manager-Play-v0
```

### Fixed-track play
```bash
python scripts/skrl/play.py --task Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0
python scripts/rsl_rl/play.py --task Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0
```

## 11) Enabling Randomization — Quick Reference

Startup physics randomization is **off by default** on `RacingQuadcopterEnvCfg`.

### Flags and what they control

| Flag | Default | Controls |
|---|---|---|
| `randomize_mass` | `False` | Startup mass scale ±10 % (`randomize_robot_mass` event) |
| `randomize_com` | `False` | Startup CoM shift ±15 mm per axis (`randomize_base_com` event) |
| `randomize_inertia` | `False` | Startup principal-inertia scale `[0.8, 1.2]` (`randomize_robot_inertia` event) |
| `randomize_actuator_gains` | `False` | Startup stiffness/damping scale ±10 % (`randomize_robot_actuator_gains` event) |
| `apply_external_disturbances` | `False` | Interval random force/torque + velocity push (`base_external_force_torque`, `push_robot` events) |
| `observation_noise_enable` | `False` | Declared on the cfg and logged (`Config/observation_noise_enable` in TensorBoard), but **not wired to anything** in the racing task — it does not gate `enable_corruption` or any noise model |
| `action_noise.enable` | `False` | Per-step Gaussian noise, per-episode bias, and LPF lag on actions (`ActionNoiseCfg`) — see [§3.2](#32-actionnoisecfg-mdpshared_featurespy) for why the other knobs must also be set |
| `recovery_spawn_prob` | `0.2` | Fraction of envs that respawn with fully random attitude and velocity (stage 2+ only) |
| `speed_limits.enabled` | Depends on `RACING_REWARD_PRESET` (see [§3.3](#33-existing-dynamics-knobs)) | Add paired observations and permanently enable sampling after curriculum activation; maximum penalties always apply |

### `--override randomize_*=True` does **not** work

`_apply_dr_flags()` (which nulls out the corresponding `EventTerm` when the flag is `False`) runs
inside each env cfg's `__post_init__`, and both `--override` and Hydra's `env.*` CLI overrides are
applied to the already-constructed cfg object **after** `__post_init__` has already run. Setting
`randomize_mass`, `randomize_com`, `randomize_inertia`, `randomize_actuator_gains`, or
`apply_external_disturbances` via `--override` therefore has no effect — the event term stays
`None`. To actually enable startup DR, edit the default value on `RacingQuadcopterEnvCfg` (or a
subclass) in `racing_quadcopter_env_cfg.py` and rerun. `action_noise.*` and `speed_limits.*` are
read live every step/reset (not consumed once in `__post_init__`), so overriding those does work.

### Enable action noise for a training run

```bash
python scripts/skrl/train.py \
    --task Isaac-Racing-Quadcopter-Manager-Fixed-v0 \
    --headless \
    --override action_noise.enable=True \
    --override action_noise.gaussian_std=0.02 \
    --override action_noise.bias_range="(-0.05, 0.05)" \
    --override action_noise.per_episode_lpf_alpha=False \
    --override action_noise.lpf_alpha=0.9
```

`action_noise.enable=True` alone is a no-op — every numeric knob defaults to zero/no-op (see
[§3.2](#32-actionnoisecfg-mdpshared_featurespy)); set `gaussian_std`, `bias_range`, and either
`lpf_alpha_range` or (`per_episode_lpf_alpha=False` + `lpf_alpha`) explicitly.

### Note on Play configs

Play configs (`*-Play-v0`) inherit the same startup-DR defaults (off) and additionally set
`eval_mode=True`, `fixed_spawn=True`, and `stats_warmup_s=30.0`. No extra flags are needed to run
a clean evaluation.

---

## 12) Recommended Next Extensions

If you want to push this further in a conventional way:
- add correlated IMU noise (random walk + bias drift),
- add latency buffers for observation transport delay,
- randomize motor time constants / thrust curve nonlinearity,
- randomize aerodynamic drag and crosswind,
- add camera/perception corruption if moving from state to vision.

These should be introduced incrementally and validated with ablations.

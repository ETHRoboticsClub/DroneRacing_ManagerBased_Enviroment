# Deployment replay

Replay a recorded CTBR action sequence in IsaacLab and compare the simulated trajectory against the recorded one.

The replay is **direct-actuation**: no env wrapper, no manager system, no rate-PID. We spawn the anemos drone in a bare `SimulationContext`, then for each recorded action push `thrust` into body +Z and convert the body-rate command to torque with one proportional gain. Everything is bypassed except PhysX itself.

## Files

- [scripts/deployment/replay_actions.py](../scripts/deployment/replay_actions.py) — applies actions, logs sim state.
- [scripts/deployment/visualize_trajectories.py](../scripts/deployment/visualize_trajectories.py) — overlays real vs sim, or plots either alone.
- [scripts/deployment/examples/](../scripts/deployment/examples/) — reference recordings (`hover_test.json`, `climb_test.json`, `ballistic_test.json`, `aggressive/`).

There is currently no `scripts/deployment/tests/run_validation.py` or `tests/` directory — the
`## Validation` section below describes a workflow that does not exist yet in this repo.

## JSON schema

```json
{
  "metadata": {
    "dt": 0.01,
    "action_format": "physical",         // only "physical" is supported
    "action_order":  "roll_pitch_yaw_thrust",
    "quaternion_order": "xyzw"
  },
  "actions":    [[roll_rate, pitch_rate, yaw_rate, thrust], ...],
  "trajectory": [
    {"t": 0.0,
     "position":         [x, y, z],
     "orientation":      [x, y, z, w],   // scalar LAST
     "linear_velocity":  [vx, vy, vz],
     "angular_velocity": [wx, wy, wz]},
    ...
  ]
}
```

- `"physical"` → rates in rad/s, thrust in N.
- The `trajectory[start_step]` entry seeds the drone state at reset; the rest is reference for the visualizer.
- The replay output JSON adds a `reference_trajectory` field — the recorded section sliced to the replay window with `t` rebased to zero — so the visualizer doesn't need the original recording to compare.

## Workflow

```bash
# 1. Replay — applies recorded actions, logs sim trajectory
python scripts/deployment/replay_actions.py \
    --input  recordings/my_run.json \
    --output logs/deployment/<name>/sim_run.json \
    --start_step 1200 --end_step 1400

# 2. Visualize — sim file already embeds the recorded reference
python scripts/deployment/visualize_trajectories.py \
    --sim logs/deployment/<name>/sim_run.json \
    --save logs/deployment/<name>/compare.png
```

- `--start_step N` / `--end_step M` slice a `[start, end)` window. The drone is seeded from `trajectory[start_step]` so you can skip pre-takeoff ground frames.
- `--real_time` throttles to wall-clock and **loops** until the viewer is closed (drop `--headless` to see it).
- `--clip_thrust` clamps recorded thrust to `[0, THRUST_MAX_N]`.
- Visualizer accepts either `--real` (a separate recording), `--sim` (uses the embedded reference), or both. With only one of them given, you plot that single trajectory.

## Validation

There is no automated validator script in this repo (`scripts/deployment/tests/run_validation.py`
does not exist). The `examples/` recordings can still be replayed manually with
`replay_actions.py` and compared visually via `visualize_trajectories.py`.

If a validator is added later, the convention would be to check the sim trajectory against a
`metadata.validation` block on the recording:

```json
"validation": {
  "final_position":         [x, y, z],
  "final_linear_velocity":  [vx, vy, vz],
  "final_orientation_xyzw": [x, y, z, w],
  "tol_pos_m": 0.10, "tol_vel_mps": 0.10, "tol_rpy_deg": 1.0
}
```

## Tuning knobs

Top of [replay_actions.py](../scripts/deployment/replay_actions.py):

```python
SIM_DT: float       = 1.0 / 100.0    # physics step [s]
THRUST_MAX_N: float = 34.192         # only used by --clip_thrust
KP_RATE: float      = 0.28           # rate-error → torque [N·m / (rad/s)]
```

`KP_RATE` is the **only** rate-loop term — strict proportional. Sets the rate-tracking time constant roughly to `I / KP_RATE`; bump it if the drone tracks too sluggishly, drop it if it oscillates. `KP_RATE=0.28` numerically matches the `anemos_sitl` drone profile's raw `pid_p` (`0.281`, `racing_quadcopter_env_cfg.py`) — but the replay script actually spawns the `anemos_usd` profile (`PROFILE = DRONE_PROFILES["anemos_usd"]` in `replay_actions.py`), whose effective rate-loop gain is `pid_p × pid_scales` (≈ `45.0 × 0.005 = 0.225` roll, `47.0 × 0.005 = 0.235` pitch — not `0.28`). Treat `KP_RATE` as an independently-tuned proportional gain for this bare-PhysX replay, not as "the same PID the racing env uses". `anemos.usd` must be generated first via `scripts/tools/generate_anemos_usd.py` before this script (or anything else selecting `anemos_usd`) will run.

## Common gotchas

- **Thrust units are N, rate units are rad/s.** Don't mix in normalized data; only `action_format: "physical"` is accepted.
- **Quaternions in the JSON are `[x, y, z, w]` (scalar last).** IsaacLab internally uses `[w, x, y, z]`; the script converts at the boundary.
- **Don't seed the drone at z ≈ 0.** If the recording starts pre-takeoff with the drone on the floor, the override puts it inside the terrain mesh and PhysX launches it skyward on contact resolution. Use `--start_step` to jump to a frame where the drone is airborne.

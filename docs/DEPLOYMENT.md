# Deployment replay

Replay a recorded CTBR action sequence in IsaacLab and compare the simulated trajectory against the recorded one.

## Files

- [scripts/deployment/replay_actions.py](../scripts/deployment/replay_actions.py) — applies actions, logs sim state.
- [scripts/deployment/visualize_trajectories.py](../scripts/deployment/visualize_trajectories.py) — overlays real vs sim trajectories.
- [scripts/deployment/tests/run_validation.py](../scripts/deployment/tests/run_validation.py) — sim-in-the-loop validator.
- [scripts/deployment/examples/](../scripts/deployment/examples/) — reference recordings with analytical truth.

## JSON schema

```json
{
  "metadata": {
    "dt": 0.01,
    "action_format": "physical",         // or "normalized"
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

- `"physical"` → rates in rad/s, thrust in N. `"normalized"` → all four channels in `[-1, 1]`.
- The first `trajectory` entry seeds the drone state at reset; the rest is reference for comparison.

## Workflow

```bash
# 1. Replay — applies recorded actions, logs sim trajectory
python scripts/deployment/replay_actions.py \
    --input  path/to/recording.json \
    --output logs/deployment/<name>/sim_run.json

# 2. Visualize real vs sim
python scripts/deployment/visualize_trajectories.py \
    --real path/to/recording.json \
    --sim  logs/deployment/<name>/sim_run.json \
    --save logs/deployment/<name>/compare.png
```

Add `--real_time` to the replay (and drop `--headless`) to watch it in the viewer.

## Validation

```bash
python scripts/deployment/tests/run_validation.py --headless
```

Runs every JSON in [scripts/deployment/examples/](../scripts/deployment/examples/) through the replay and checks the resulting sim trajectory against the analytical reference inside the JSON. Exit code `0` = all pass, `1` = any fail. Pass `--save_sim_runs` to dump each test's sim trajectory under `logs/deployment/validation/<ts>/`.

To validate your own recording, add a `metadata.validation` block:

```json
"validation": {
  "final_position":         [x, y, z],
  "final_linear_velocity":  [vx, vy, vz],
  "final_orientation_xyzw": [x, y, z, w],
  "tol_pos_m": 0.10, "tol_vel_mps": 0.10, "tol_rpy_deg": 1.0
}
```

## Tuning knobs

Top of both [replay_actions.py](../scripts/deployment/replay_actions.py) and [run_validation.py](../scripts/deployment/tests/run_validation.py) — **keep them in sync**:

```python
SIM_DT: float = 1.0 / 100.0          # physics step [s]
DECIMATION: int = 1                  # physics steps per env.step()
THRUST_MAX_N: float = 34.192         # total thrust at normalized action = +1
RATES_SCALE_OVERRIDE: float | None = None   # None → use env cfg's RATES_SCALE
```

The replay env uses single-env, no domain randomization, no action noise, no termination — deterministic by construction.

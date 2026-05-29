# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded CTBR action sequence in the hovering quadcopter env.

The script spawns a single drone the same way as
``Isaac-Hovering-Quadcopter-Manager-v0`` (using the manager-based hovering
config), seeds it with the recorded initial pose / velocity, applies the
recorded action sequence one step at a time, and dumps a rigorous per-step
sim state to ``logs/deployment/<run_name>/sim_run.json``.

Input JSON schema (any subset of these keys is accepted)::

    {
      "metadata": {
        "dt": 0.01,                       # seconds per action step (optional)
        "action_format": "normalized",    # "normalized" (in [-1, 1]) or "physical"
                                          # physical: [roll_rate, pitch_rate, yaw_rate, thrust_N] (rad/s, rad/s, rad/s, N)
        "frame": "world"
      },
      "actions": [[roll_rate, pitch_rate, yaw_rate, thrust], ...],
      "trajectory": [                    # optional; first entry seeds the drone state
        {"t": 0.0,
         "position":          [x, y, z],
         "orientation":       [x, y, z, w],          # quaternion, scalar LAST
         "linear_velocity":   [vx, vy, vz],
         "angular_velocity":  [wx, wy, wz]},
        ...
      ]
    }

Conventions (this codebase):
  * Action order: ``[roll_rate, pitch_rate, yaw_rate, thrust]`` (thrust LAST).
    The IsaacLab action term internally expects ``[thrust, roll, pitch, yaw]``
    — reordering is centralized in ``_actions_to_env_order``.
  * Quaternion order: ``[x, y, z, w]`` (scalar LAST). IsaacLab stores
    ``[w, x, y, z]`` internally — reordering is centralized in ``_quat_xyzw``
    / ``_quat_wxyz``.

The same JSON can hold both ``actions`` and ``trajectory`` — that is the file
the operator records during a teleop run, and is also the "real" reference
fed to ``visualize_trajectories.py`` for the comparison plots.
"""

from __future__ import annotations

# =============================================================================
# Tuning knobs — kept at the top so they are not buried inside _build_env_cfg.
# Change these if you need a different physics rate / decimation / dynamics
# profile for the replay; everything downstream picks them up automatically.
# =============================================================================
SIM_DT: float = 1.0 / 100.0          # physics step [s]                 → 100 Hz
DECIMATION: int = 1                  # physics steps per env.step()    → policy at 100 Hz
THRUST_MAX_N: float = 34.192         # mirrors the hardcoded total-thrust ceiling in
                                     # racing_quadcopter/mdp/actions.py:328 — used to
                                     # map physical thrust [N] → normalized [-1, 1].
RATES_SCALE_OVERRIDE: float | None = None  # None → use the env cfg's RATES_SCALE
# =============================================================================

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI / AppLauncher (must be done before importing isaaclab.* / isaacsim.*)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Replay a recorded action sequence in IsaacLab.")
parser.add_argument("--input", type=str, required=True, help="Path to the recorded JSON (actions + optional trajectory).")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path of the sim-trajectory output JSON (default: logs/deployment/<timestamp>/sim_run.json).",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Hovering-Quadcopter-Manager-v0",
    help="Gym task id to use (must spawn the same drone as the hovering env).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle replay to wall-clock.")
parser.add_argument("--seed", type=int, default=0, help="Env seed (deterministic replay).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

# Strip our extra args so Hydra-style downstream parsers don't choke.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Late imports (post-AppLauncher).
# ---------------------------------------------------------------------------
import json
import os
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

import Racing_Quadcopter.tasks  # noqa: F401
from Racing_Quadcopter.tasks.manager_based.hovering_quadcopter.hovering_quadcopter_env_cfg import (
    HoveringQuadcopterEnvCfg,
    RATES_SCALE as _RATES_SCALE_DEFAULT,
)

RATES_SCALE: float = float(RATES_SCALE_OVERRIDE) if RATES_SCALE_OVERRIDE is not None else float(_RATES_SCALE_DEFAULT)


# ---------------------------------------------------------------------------
# Convention-bridging helpers
# ---------------------------------------------------------------------------
def _actions_to_env_order(actions_rpyt: np.ndarray) -> np.ndarray:
    """User format ``[roll, pitch, yaw, thrust]`` → env format ``[thrust, roll, pitch, yaw]``."""
    out = np.empty_like(actions_rpyt)
    out[:, 0] = actions_rpyt[:, 3]
    out[:, 1:] = actions_rpyt[:, :3]
    return out


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray | torch.Tensor):
    """Reorder a single quaternion from scalar-last to scalar-first."""
    if isinstance(q_xyzw, torch.Tensor):
        return torch.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], dim=-1)
    q = np.asarray(q_xyzw, dtype=np.float32)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def _quat_wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Reorder a single quaternion from scalar-first to scalar-last (for logging out)."""
    return torch.stack([q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3], q_wxyz[..., 0]], dim=-1)


def _load_input(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)
    if "actions" not in data:
        raise ValueError(f"Input JSON {path} has no 'actions' field.")
    return data


def _actions_to_normalized(actions: np.ndarray, fmt: str) -> np.ndarray:
    """Map raw recorded actions ``[roll, pitch, yaw, thrust]`` → normalized [-1, 1].

    "physical" → roll/pitch/yaw in rad/s, thrust in N.
    "normalized" → already in [-1, 1] in the same channel order.
    """
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"Expected actions of shape (T, 4); got {actions.shape}.")
    if fmt == "normalized":
        return np.clip(actions, -1.0, 1.0)
    if fmt == "physical":
        out = np.empty_like(actions)
        out[:, 0] = actions[:, 0] / RATES_SCALE                   # roll rate
        out[:, 1] = actions[:, 1] / RATES_SCALE                   # pitch rate
        out[:, 2] = actions[:, 2] / RATES_SCALE                   # yaw rate
        out[:, 3] = actions[:, 3] / (0.5 * THRUST_MAX_N) - 1.0    # thrust
        return np.clip(out, -1.0, 1.0)
    raise ValueError(f"Unknown action_format={fmt!r} (expected 'normalized' or 'physical').")


def _build_env_cfg() -> HoveringQuadcopterEnvCfg:
    """Single-env, deterministic, no-DR config for faithful replay."""
    cfg = HoveringQuadcopterEnvCfg()

    cfg.seed = int(args_cli.seed)
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 5.0
    cfg.terrain_num_rows = 1
    cfg.terrain_num_cols = 1
    cfg.cell_size = (20.0, 20.0)

    # Big enough that we never hit a timeout mid-replay; we manage the loop ourselves.
    cfg.episode_length_s = 1.0e9

    # Physics rate / decimation knobs from the top of this file.
    cfg.sim.dt = SIM_DT
    cfg.decimation = DECIMATION
    cfg.sim.render_interval = DECIMATION

    # Deterministic dynamics: no DR, no action noise, no disturbances.
    cfg.randomize_mass = False
    cfg.randomize_com = False
    cfg.randomize_inertia = False
    cfg.randomize_actuator_gains = False
    cfg.apply_external_disturbances = False
    cfg.action_noise.enable = False

    # Disable forced terminations — we capture the full sequence even on crash.
    cfg.terminations.crashed = None
    cfg.terminations.all_collected = None

    cfg.draw_waypoints = False
    return cfg


def _override_initial_state(env, init_state: dict | None) -> None:
    """Write the recorded initial pose + velocity to the simulated drone.

    Accepts orientation as ``[x, y, z, w]`` (scalar last) and converts to the
    IsaacLab internal ``[w, x, y, z]`` convention.
    """
    if init_state is None:
        return
    robot = env.unwrapped._robot
    device = env.unwrapped.device

    pos = torch.as_tensor(init_state["position"], device=device, dtype=torch.float32).view(1, 3)
    quat_xyzw = torch.as_tensor(init_state["orientation"], device=device, dtype=torch.float32).view(1, 4)
    quat_wxyz = _quat_xyzw_to_wxyz(quat_xyzw)
    quat_wxyz = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    root_pose = torch.cat([pos, quat_wxyz], dim=-1)

    lin_v = torch.as_tensor(
        init_state.get("linear_velocity", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32
    ).view(1, 3)
    ang_v = torch.as_tensor(
        init_state.get("angular_velocity", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32
    ).view(1, 3)
    root_vel = torch.cat([lin_v, ang_v], dim=-1)

    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    robot.write_root_pose_to_sim(root_pose, env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()


def _capture_state(env, t: float, step_idx: int, raw_action_rpyt: np.ndarray, norm_action_rpyt: np.ndarray) -> dict:
    """Pull a rigorous snapshot of the drone state from the simulator.

    Quaternions are emitted as ``[x, y, z, w]`` and actions as ``[roll, pitch, yaw, thrust]``
    to match the input-file convention.
    """
    robot = env.unwrapped._robot
    data = robot.data

    def to_list(x: torch.Tensor) -> list:
        return x.detach().cpu().numpy().reshape(-1).tolist()

    quat_xyzw = _quat_wxyz_to_xyzw(data.root_quat_w[0])

    snapshot = {
        "t": float(t),
        "step": int(step_idx),
        "position": to_list(data.root_pos_w[0]),
        "orientation": to_list(quat_xyzw),
        "linear_velocity": to_list(data.root_lin_vel_w[0]),
        "linear_velocity_body": to_list(data.root_lin_vel_b[0]),
        "angular_velocity": to_list(data.root_ang_vel_w[0]),
        "angular_velocity_body": to_list(data.root_ang_vel_b[0]),
        "projected_gravity": to_list(data.projected_gravity_b[0]),
        "action_normalized": norm_action_rpyt.tolist(),
        "action_raw": raw_action_rpyt.tolist(),
    }
    if data.joint_pos.shape[1] > 0:
        snapshot["joint_pos"] = to_list(data.joint_pos[0])
        snapshot["joint_vel"] = to_list(data.joint_vel[0])
    return snapshot


def _resolve_output_path() -> Path:
    if args_cli.output is not None:
        return Path(args_cli.output)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs/deployment") / ts / "sim_run.json"


def main() -> None:
    input_path = Path(args_cli.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    recording = _load_input(input_path)
    meta = recording.get("metadata", {})
    action_format = meta.get("action_format", "normalized")
    raw_actions_rpyt = np.asarray(recording["actions"], dtype=np.float32)
    norm_actions_rpyt = _actions_to_normalized(raw_actions_rpyt, action_format)
    norm_actions_env = _actions_to_env_order(norm_actions_rpyt)  # → [thrust, roll, pitch, yaw]

    init_state = None
    traj = recording.get("trajectory")
    if traj:
        init_state = traj[0]

    env_cfg = _build_env_cfg()
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(
        f"[INFO] Replay env: {args_cli.task} | sim.dt={SIM_DT:.5f}s | decimation={DECIMATION} | "
        f"step_dt={env.unwrapped.step_dt:.5f}s | RATES_SCALE={RATES_SCALE:.3f} | THRUST_MAX_N={THRUST_MAX_N:.3f}"
    )

    env.reset(seed=int(args_cli.seed))
    _override_initial_state(env, init_state)

    step_dt = float(env.unwrapped.step_dt)
    recorded_dt = float(meta.get("dt", step_dt))
    if abs(recorded_dt - step_dt) > 1e-6:
        print(
            f"[WARN] Recorded dt={recorded_dt:.5f}s differs from env.step_dt={step_dt:.5f}s. "
            "Actions are applied 1-per-env-step regardless; the resulting time axis uses env.step_dt."
        )

    log_states: list[dict] = []
    zero4 = np.zeros(4, dtype=np.float32)
    log_states.append(
        _capture_state(env, t=0.0, step_idx=0, raw_action_rpyt=zero4, norm_action_rpyt=zero4)
    )

    device = env.unwrapped.device
    norm_actions_env_t = torch.from_numpy(norm_actions_env).to(device)
    n_steps = norm_actions_env_t.shape[0]

    # In --real_time mode the replay loops indefinitely (re-seeding the drone
    # state between iterations) until the simulator window is closed. Only the
    # first iteration is logged — subsequent ones are just for viewing.
    loop_forever = bool(args_cli.real_time)
    iteration = 0
    aborted = False
    print(
        f"[INFO] Replaying {n_steps} actions (format={action_format})."
        + (" Looping until simulator window closes." if loop_forever else "")
    )
    while True:
        if iteration > 0:
            _override_initial_state(env, init_state)
            print(f"[INFO] Replay loop iteration {iteration + 1}.")

        for i in range(n_steps):
            if not simulation_app.is_running():
                print("[INFO] Simulator window closed — aborting replay.")
                aborted = True
                break
            start = time.time()
            action = norm_actions_env_t[i:i + 1]  # (1, 4) in env order
            with torch.inference_mode():
                env.step(action)
            if iteration == 0:
                log_states.append(
                    _capture_state(
                        env,
                        t=(i + 1) * step_dt,
                        step_idx=i + 1,
                        raw_action_rpyt=raw_actions_rpyt[i],
                        norm_action_rpyt=norm_actions_rpyt[i],
                    )
                )
            if args_cli.real_time:
                slack = step_dt - (time.time() - start)
                if slack > 0:
                    time.sleep(slack)

        iteration += 1
        if aborted or not loop_forever:
            break

    output_path = _resolve_output_path().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "source_input": str(input_path),
            "task": args_cli.task,
            "sim_dt": SIM_DT,
            "decimation": DECIMATION,
            "step_dt": step_dt,
            "rates_scale": RATES_SCALE,
            "thrust_max_N": THRUST_MAX_N,
            "n_states": len(log_states),
            "n_actions": n_steps,
            "action_format": action_format,
            "action_order": "roll_pitch_yaw_thrust",
            "quaternion_order": "xyzw",
            "frame": "world",
            "drone": os.environ.get("HOVERING_DRONE_MODEL", "anemos_usd"),
            "seed": int(args_cli.seed),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "trajectory": log_states,
        "actions": raw_actions_rpyt.tolist(),
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Wrote sim trajectory ({len(log_states)} states) to {output_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

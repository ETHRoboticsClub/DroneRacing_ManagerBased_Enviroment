# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal direct-actuation replay for recorded CTBR action sequences.

No env wrapper. No manager system. No PID rate loop. No action delay buffer.
Just a bare ``SimulationContext`` with the anemos drone spawned in it, and
for every recorded action we:

  1. apply ``thrust`` (Newtons) as a **world-frame force along body +z**,
  2. snap the drone's angular velocity to the **commanded body rates (rad/s)**
     — kinematic, no inertia, no PID,
  3. step physics by ``SIM_DT``.

Linear motion is therefore F = ma + gravity (PhysX integrates honestly).
Orientation follows the recorded rates exactly. This makes "thrust = 0" mean
"no upward force" with no possible reinterpretation by upstream layers.

Input JSON schema (same as before)::

    {
      "metadata": {"dt": 0.01, "action_format": "physical",
                   "action_order": "roll_pitch_yaw_thrust",
                   "quaternion_order": "xyzw"},
      "actions": [[roll_rate, pitch_rate, yaw_rate, thrust_N], ...],
      "trajectory": [{"t": ..., "position": [...], "orientation": [x,y,z,w],
                      "linear_velocity": [...], "angular_velocity": [...]}, ...]
    }

Only ``action_format = "physical"`` is supported here — units are unambiguous.
"""

from __future__ import annotations

# =============================================================================
# Tuning knobs. Keep visible at the top.
# =============================================================================
SIM_DT: float = 1.0 / 100.0          # physics step [s]
THRUST_MAX_N: float = 34.192         # used only when --clip_thrust is on

# Rate command is in rad/s. The wrench composer needs N·m. This Kp converts
# rate error → torque: ``tau = KP_RATE * (omega_des - omega_actual)``.
# 0.28 matches the racing env's PID P-gain on this drone.
KP_RATE: float = 0.28
# =============================================================================

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Direct-actuation replay (no env wrapper).")
parser.add_argument("--input", type=str, required=True, help="Recorded JSON.")
parser.add_argument("--output", type=str, default=None, help="Sim-trajectory output JSON.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle to wall-clock & loop until window closes.")
parser.add_argument("--start_step", type=int, default=0, help="Begin at this index into the action sequence.")
parser.add_argument("--end_step", type=int, default=None, help="Stop at this index (exclusive). Default: end of file.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--clip_thrust",
    action="store_true",
    default=False,
    help=f"Clip thrust to [0, {THRUST_MAX_N}] N. Off by default — recorded thrust is applied as-is.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Late imports
# ---------------------------------------------------------------------------
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
import Racing_Quadcopter.tasks  # noqa: F401 — ensures asset paths are importable
from Racing_Quadcopter.tasks.manager_based.racing_quadcopter.racing_quadcopter_env_cfg import DRONE_PROFILES


PROFILE = DRONE_PROFILES["anemos_usd"]


# ---------------------------------------------------------------------------
# Quaternion helpers — JSON is xyzw, IsaacLab is wxyz.
# ---------------------------------------------------------------------------
def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], dim=-1)


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)


# ---------------------------------------------------------------------------
# State capture
# ---------------------------------------------------------------------------
def _capture(robot: Articulation, t: float, action_rpyt: np.ndarray) -> dict:
    data = robot.data
    quat_xyzw = _wxyz_to_xyzw(data.root_quat_w[0])
    return {
        "t": float(t),
        "position":         data.root_pos_w[0].detach().cpu().tolist(),
        "orientation":      quat_xyzw.detach().cpu().tolist(),
        "linear_velocity":  data.root_lin_vel_w[0].detach().cpu().tolist(),
        "angular_velocity": data.root_ang_vel_w[0].detach().cpu().tolist(),
        "action_raw":       np.asarray(action_rpyt, dtype=np.float32).tolist(),
    }


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
def _resolve_output_path() -> Path:
    if args_cli.output is not None:
        return Path(args_cli.output)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs/deployment") / ts / "sim_run.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    inp = Path(args_cli.input).resolve()
    if not inp.exists():
        raise FileNotFoundError(inp)
    rec = json.loads(inp.read_text())

    meta = rec.get("metadata", {})
    fmt = meta.get("action_format", "physical")
    if fmt != "physical":
        raise ValueError(
            f"Only action_format='physical' is supported in this replay (got {fmt!r}). "
            "Re-export your recording with thrust in N and rates in rad/s."
        )

    actions_full = np.asarray(rec["actions"], dtype=np.float32)
    traj = rec.get("trajectory") or []

    # Resolve [start, end) window.
    n_total = actions_full.shape[0]
    start = max(0, int(args_cli.start_step))
    end = n_total if args_cli.end_step is None else int(args_cli.end_step)
    end = max(start, min(end, n_total))
    if start >= n_total or end <= start:
        raise ValueError(f"Empty replay window [{start}:{end}] (recording has {n_total} actions).")
    actions = actions_full[start:end]
    n_steps = actions.shape[0]
    if start > 0 or end < n_total:
        print(f"[INFO] Replay window: actions[{start}:{end}] ({n_steps}/{n_total} actions).")

    if not traj:
        raise ValueError("Recording has no 'trajectory' field — need at least one state to seed the drone.")
    init_state = traj[min(start, len(traj) - 1)]

    if args_cli.clip_thrust:
        n_clipped = int(((actions[:, 3] < 0.0) | (actions[:, 3] > THRUST_MAX_N)).sum())
        if n_clipped > 0:
            print(f"[INFO] --clip_thrust: clamped {n_clipped} of {n_steps} thrust commands to [0, {THRUST_MAX_N}] N.")
        actions[:, 3] = np.clip(actions[:, 3], 0.0, THRUST_MAX_N)

    # ----------------------------------------------------------------------
    # Build sim — no env, no scene, no managers.
    # ----------------------------------------------------------------------
    sim_cfg = SimulationCfg(dt=SIM_DT, render_interval=1)
    sim = SimulationContext(sim_cfg)
    try:
        sim.set_camera_view(eye=(4.0, 4.0, 4.0), target=tuple(float(v) for v in init_state["position"]))
    except Exception:
        pass

    # Lighting only. No ground plane on purpose — recordings that start with
    # the drone on the floor would otherwise be launched skyward by PhysX
    # contact resolution against the terrain mesh.
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Spawn the anemos at the recorded initial position.
    spawn = PROFILE.robot_cfg.spawn
    spawn.func("/World/Robot", spawn, translation=tuple(float(v) for v in init_state["position"]))

    robot_cfg = copy.deepcopy(PROFILE.robot_cfg)
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(robot_cfg)

    sim.reset()
    device = sim.device

    # Body id for force application — anemos uses 'base_link'. The wrench
    # composer accepts a list of body indices, so we keep ``body_ids`` as the
    # tuple/list returned by ``find_bodies`` to match the racing action term
    # ([racing_quadcopter/mdp/actions.py] uses ``self._body_id`` the same way).
    body_ids, _ = robot.find_bodies("base_link")
    if not body_ids:
        raise RuntimeError("Could not find 'base_link' body on the anemos articulation.")

    # ----------------------------------------------------------------------
    # Initial state override (apply our recorded pose / velocity).
    # ----------------------------------------------------------------------
    pos = torch.tensor(init_state["position"], device=device, dtype=torch.float32).view(1, 3)
    quat_xyzw = torch.tensor(init_state["orientation"], device=device, dtype=torch.float32).view(1, 4)
    quat_wxyz = _xyzw_to_wxyz(quat_xyzw)
    quat_wxyz = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    root_pose = torch.cat([pos, quat_wxyz], dim=-1)
    lin_v = torch.tensor(init_state.get("linear_velocity", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32).view(1, 3)
    ang_v = torch.tensor(init_state.get("angular_velocity", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32).view(1, 3)
    root_vel = torch.cat([lin_v, ang_v], dim=-1)

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_vel)
    sim.forward()

    actions_t = torch.from_numpy(actions).to(device)
    log_states: list[dict] = [_capture(robot, t=0.0, action_rpyt=np.zeros(4, dtype=np.float32))]

    loop_forever = bool(args_cli.real_time)
    print(
        f"[INFO] Direct replay: {n_steps} actions @ SIM_DT={SIM_DT:.5f}s."
        + (" Looping until simulator window closes." if loop_forever else "")
    )

    # Pre-allocate the wrench tensors — shape (num_envs=1, num_bodies=1, 3).
    force = torch.zeros(1, 1, 3, device=device)
    torque = torch.zeros(1, 1, 3, device=device)

    # ----------------------------------------------------------------------
    # Replay loop. Each step: thrust → body +Z force, action[0:3] → body torques,
    # apply via the wrench composer (same path as racing's QuadcopterRatesAction),
    # step physics. That's it.
    # ----------------------------------------------------------------------
    iteration = 0
    aborted = False
    while True:
        if iteration > 0:
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_vel)
            sim.forward()
            print(f"[INFO] Loop iteration {iteration + 1}.")

        for i in range(n_steps):
            if not simulation_app.is_running():
                print("[INFO] Simulator window closed — aborting.")
                aborted = True
                break
            t_start = time.time()

            roll, pitch, yaw, thrust = actions_t[i].unbind()

            # Rate command → torque via proportional control.  Rates are in
            # rad/s, torque slot expects N·m — without this conversion they're
            # different physical quantities and the drone spins out.
            omega_des = torch.stack([roll, pitch, yaw])
            omega_b = robot.data.root_ang_vel_b[0]
            tau = KP_RATE * (omega_des - omega_b)

            force[0, 0, 2] = thrust
            torque[0, 0, :] = tau
            robot.permanent_wrench_composer.set_forces_and_torques(
                forces=force, torques=torque, body_ids=body_ids
            )

            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(SIM_DT)

            if iteration == 0:
                log_states.append(_capture(robot, t=(i + 1) * SIM_DT, action_rpyt=actions[i]))

            if args_cli.real_time:
                target = t_start + SIM_DT
                while time.time() < target:
                    if not simulation_app.is_running():
                        break
                    simulation_app.update()
                    remaining = target - time.time()
                    if remaining > 0.001:
                        time.sleep(min(0.005, remaining))

        iteration += 1
        if aborted or not loop_forever:
            break

    # ----------------------------------------------------------------------
    # Save — same schema the visualiser already consumes.
    # ----------------------------------------------------------------------
    out_path = _resolve_output_path().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Rebase reference time so it starts at 0 (matches the sim time axis).
    ref_window: list[dict] = []
    if traj:
        sliced = traj[start : end + 1]
        if sliced:
            t0 = float(sliced[0].get("t", 0.0))
            for st in sliced:
                shifted = dict(st)
                if "t" in shifted:
                    shifted["t"] = float(shifted["t"]) - t0
                ref_window.append(shifted)

    payload = {
        "metadata": {
            "source_input": str(inp),
            "method": "direct_force_and_kinematic_rates",
            "sim_dt": SIM_DT,
            "thrust_max_N": THRUST_MAX_N,
            "n_states": len(log_states),
            "n_actions": n_steps,
            "n_reference_states": len(ref_window),
            "replay_window": {"start_step": int(start), "end_step": int(end)},
            "action_format": "physical",
            "action_order": "roll_pitch_yaw_thrust",
            "quaternion_order": "xyzw",
            "frame": "world",
            "drone": "anemos_usd",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "trajectory": log_states,
        "reference_trajectory": ref_window,
        "actions": actions.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Wrote sim trajectory ({len(log_states)} states) to {out_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

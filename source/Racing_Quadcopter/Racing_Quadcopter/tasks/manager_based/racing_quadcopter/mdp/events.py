# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter: episode metrics at reset and racing spawn (event manager reset mode)."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply


def collect_episode_metrics_before_reset(env: ManagerBasedRLEnv, env_ids_tensor: torch.Tensor) -> dict:
    """Log episode stats and reset per-episode buffers for envs about to reset. Call before ``super()._reset_idx``."""
    gate_log: dict = {}
    if env_ids_tensor.numel() == 0:
        return gate_log

    gate_log["Episode_Metrics/gates_passed"] = env._gates_passed[env_ids_tensor].float().mean().item()

    step_counts = env._episode_step_count[env_ids_tensor]
    valid_steps = step_counts > 0
    if valid_steps.any():
        mean_vel = (
            env._episode_velocity_sum[env_ids_tensor][valid_steps] / step_counts[valid_steps]
        ).mean().item()
        gate_log["Episode_Metrics/mean_velocity"] = mean_vel
    else:
        gate_log["Episode_Metrics/mean_velocity"] = 0.0

    gate_log["Episode_Metrics/collision_count"] = env._episode_collision_count[env_ids_tensor].mean().item()

    env._last_episode_gates_passed[env_ids_tensor] = env._gates_passed[env_ids_tensor].float()
    has_passages = env._passage_count[env_ids_tensor] > 0
    env._last_episode_mean_velocity[env_ids_tensor] = torch.where(
        has_passages,
        env._passage_velocity_sum[env_ids_tensor] / env._passage_count[env_ids_tensor],
        torch.zeros_like(env._passage_velocity_sum[env_ids_tensor]),
    )
    env._last_episode_center_offset[env_ids_tensor] = torch.where(
        has_passages,
        env._passage_centering_sum[env_ids_tensor] / env._passage_count[env_ids_tensor],
        torch.zeros_like(env._passage_centering_sum[env_ids_tensor]),
    )

    env._episode_velocity_sum[env_ids_tensor] = 0.0
    env._episode_step_count[env_ids_tensor] = 0.0
    env._episode_collision_count[env_ids_tensor] = 0.0
    env._passage_velocity_sum[env_ids_tensor] = 0.0
    env._passage_centering_sum[env_ids_tensor] = 0.0
    env._passage_count[env_ids_tensor] = 0.0

    return gate_log


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion from XYZ Euler angles in wxyz order."""
    hr = 0.5 * roll
    hp = 0.5 * pitch
    hy = 0.5 * yaw
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qw, qx, qy, qz], dim=-1)


def _write_spawn_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    start_pos: torch.Tensor,
    start_quat: torch.Tensor,
    root_vel: torch.Tensor,
    init_actions: torch.Tensor | None = None,
) -> None:
    root_poses = torch.zeros(len(env_ids), 7, device=env.device)
    root_poses[:, :3] = start_pos
    root_poses[:, 3:7] = start_quat

    joint_pos = env._robot.data.default_joint_pos[env_ids]
    joint_vel = env._robot.data.default_joint_vel[env_ids]
    env._robot.write_root_pose_to_sim(root_poses, env_ids)
    env._robot.write_root_velocity_to_sim(root_vel, env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    env._prev_drone_pos[env_ids] = start_pos
    if init_actions is None:
        env._actions_buf[env_ids] = 0.0
        env._prev_actions[env_ids] = 0.0
    else:
        env._actions_buf[env_ids] = init_actions
        env._prev_actions[env_ids] = init_actions



def _apply_ground_initialization(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Original ground initialization: start behind first gate with gate orientation."""
    n = len(env_ids)
    env._current_gate_idx[env_ids] = 0
    env._gates_passed[env_ids] = 0
    first_gate_pos = env._gate_positions[env_ids, 0]
    first_gate_ori = env._gate_orientations[env_ids, 0]
    fwd = quat_apply(first_gate_ori, env._y_local.expand(n, -1))

    start_pos = first_gate_pos - 2.0 * fwd
    start_pos[:, 2] = first_gate_pos[:, 2]
    start_quat = first_gate_ori
    root_vel = torch.zeros(n, 6, device=env.device)

    _write_spawn_state(env, env_ids, start_pos, start_quat, root_vel)


def _apply_uniform_initialization(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Uniform spawn in a small box on the approach side of gate 0.

    Box is defined in the gate's own frame so it is always before the gate
    regardless of world orientation:
      - depth  [1.0, 4.0] m behind the gate (negative forward axis)
      - right  ±1.5 m along gate right axis
      - up     ±0.75 m around gate center height
    """
    n = len(env_ids)
    env._current_gate_idx[env_ids] = 0
    env._gates_passed[env_ids] = 0

    gate_pos = env._gate_positions[env_ids, 0]    # [N, 3]
    gate_ori = env._gate_orientations[env_ids, 0]  # [N, 4]

    fwd   = quat_apply(gate_ori, env._y_local.expand(n, -1))
    right = quat_apply(gate_ori, env._x_local.expand(n, -1))
    up    = quat_apply(gate_ori, env._z_local.expand(n, -1))

    depth   = 1.0 + torch.rand(n, device=env.device) * 3.0          # [1, 4] m before gate
    lateral = 1.5 * (2.0 * torch.rand(n, device=env.device) - 1.0)  # ±1.5 m right/left
    vert    = 0.75 * (2.0 * torch.rand(n, device=env.device) - 1.0) # ±0.75 m up/down

    start_pos = (
        gate_pos
        - depth.unsqueeze(-1)   * fwd
        + lateral.unsqueeze(-1) * right
        + vert.unsqueeze(-1)    * up
    )
    # Never go below ground.
    start_pos[:, 2] = start_pos[:, 2].clamp(min=0.2)

    roll  = -torch.pi / 9.0 + torch.rand(n, device=env.device) * (2.0 * torch.pi / 9.0)
    pitch = -torch.pi / 9.0 + torch.rand(n, device=env.device) * (2.0 * torch.pi / 9.0)
    yaw   = -torch.pi + torch.rand(n, device=env.device) * (2.0 * torch.pi)
    start_quat = _quat_from_euler_xyz(roll, pitch, yaw)

    lin_vel = -0.5 + torch.rand(n, 3, device=env.device) * 1.0
    ang_vel = -0.1 + torch.rand(n, 3, device=env.device) * 0.2
    root_vel = torch.cat([lin_vel, ang_vel], dim=-1)

    init_actions = -1.0 + torch.rand(n, 4, device=env.device) * 2.0
    _write_spawn_state(env, env_ids, start_pos, start_quat, root_vel, init_actions=init_actions)


def reset_racing_spawn(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None, method: str = "mixed"
) -> None:
    """Reset gate progress and sample initialization method.

    Args:
        method: One of ``"ground"``, ``"uniform"``, or ``"mixed"``.
    """
    n = len(env_ids)
    if n == 0:
        return

    if method == "ground":
        _apply_ground_initialization(env, env_ids)
    elif method == "uniform":
        _apply_uniform_initialization(env, env_ids)
    elif method == "mixed":
        ground_prob = float(getattr(env.cfg, "ground_init_prob", 0.5))
        use_ground = torch.rand(n, device=env.device) < ground_prob
        ground_ids = env_ids[use_ground]
        uniform_ids = env_ids[~use_ground]
        if len(ground_ids) > 0:
            _apply_ground_initialization(env, ground_ids)
        if len(uniform_ids) > 0:
            _apply_uniform_initialization(env, uniform_ids)
    else:
        raise ValueError(f"Unknown reset method '{method}'. Use 'ground', 'uniform', or 'mixed'.")

    if env.cfg.randomize_velocity_limit:
        lo, hi = env.cfg.velocity_limit_range
        env._velocity_limit[env_ids] = torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo
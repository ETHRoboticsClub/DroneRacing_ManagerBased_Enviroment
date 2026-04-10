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


def apply_racing_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset gate progress and place the robot at the start line (behind gate 0)."""
    env._current_gate_idx[env_ids] = 0
    env._gates_passed[env_ids] = 0

    first_gate_pos = env._gate_positions[env_ids, 0]
    first_gate_ori = env._gate_orientations[env_ids, 0]
    fwd = quat_apply(first_gate_ori, env._y_local.expand(len(env_ids), -1))

    start_pos = first_gate_pos - 2.0 * fwd
    start_pos[:, 2] = first_gate_pos[:, 2]

    root_poses = torch.zeros(len(env_ids), 7, device=env.device)
    root_poses[:, :3] = start_pos
    root_poses[:, 3:7] = first_gate_ori
    root_vel = torch.zeros(len(env_ids), 6, device=env.device)

    joint_pos = env._robot.data.default_joint_pos[env_ids]
    joint_vel = env._robot.data.default_joint_vel[env_ids]

    env._robot.write_root_pose_to_sim(root_poses, env_ids)
    env._robot.write_root_velocity_to_sim(root_vel, env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    env._prev_drone_pos[env_ids] = start_pos
    env._actions_buf[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0

    if env.cfg.randomize_velocity_limit:
        lo, hi = env.cfg.velocity_limit_range
        env._velocity_limit[env_ids] = torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo


def reset_racing_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Event term: place the drone at the task start after :func:`reset_scene_to_default`.

    ``env_ids`` follows the event manager convention (tensor of indices, or ``slice(None)`` for all).
    """
    if env_ids is None:
        return
    if isinstance(env_ids, slice):
        env_ids_t = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
    if env_ids_t.numel() == 0:
        return
    apply_racing_spawn(env, env_ids_t)

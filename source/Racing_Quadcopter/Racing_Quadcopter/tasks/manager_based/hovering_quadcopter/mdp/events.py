# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event callables for the hovering / waypoint-following env.

Re-exports the stock IsaacLab event terms (mass / CoM / actuator-gain
randomisation, external force/torque, reset_scene_to_default) and adds the
hovering-specific spawn term.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.events import *  # noqa: F401, F403

from Racing_Quadcopter.tasks.manager_based.route_progress import snapshot_episode_route_progress


def collect_episode_metrics_before_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> dict:
    """Snapshot episode metrics for resetting envs, then zero the accumulators."""
    if env_ids.numel() == 0:
        return {}
    out = {
        "Metrics/waypoints_collected": env._episode_collected_count[env_ids].float().mean().item(),
        "Metrics/episode_length": env.episode_length_buf[env_ids].float().mean().item(),
    }
    steps = env.episode_length_buf[env_ids].float().clamp(min=1.0)
    capture_steps = env._hold_capture_steps[env_ids]
    capture_denominator = capture_steps.clamp(min=1.0)
    out.update(
        {
            "Holding/capture_fraction": (capture_steps / steps).mean().item(),
            "Holding/linear_speed_ok_given_capture": (
                env._hold_linear_ok_capture_steps[env_ids] / capture_denominator
            ).mean().item(),
            "Holding/angular_speed_ok_given_capture": (
                env._hold_angular_ok_capture_steps[env_ids] / capture_denominator
            ).mean().item(),
            "Holding/tilt_ok_given_capture": (
                env._hold_tilt_ok_capture_steps[env_ids] / capture_denominator
            ).mean().item(),
            "Holding/stable_capture_fraction": (
                env._hold_stable_capture_steps[env_ids] / capture_denominator
            ).mean().item(),
            "Holding/timer_resets": env._hold_timer_reset_count[env_ids].mean().item(),
            "Holding/dwell_time_s": env._hold_dwell_time_s[env_ids].mean().item(),
        }
    )
    out.update(snapshot_episode_route_progress(env, env_ids))
    env._episode_collected_count[env_ids] = 0.0
    env._hold_capture_steps[env_ids] = 0.0
    env._hold_linear_ok_capture_steps[env_ids] = 0.0
    env._hold_angular_ok_capture_steps[env_ids] = 0.0
    env._hold_tilt_ok_capture_steps[env_ids] = 0.0
    env._hold_stable_capture_steps[env_ids] = 0.0
    env._hold_timer_reset_count[env_ids] = 0.0
    env._hold_dwell_time_s[env_ids] = 0.0
    return out


def reset_waypoint_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Spawn near waypoint 0 of each env's track with small horizontal jitter.

    Resets dwell-state buffers and per-env action history so the next episode
    starts clean.
    """
    if env_ids.numel() == 0:
        return
    n = env_ids.numel()
    # Active target → first waypoint of the per-env track.
    env._current_wp_idx[env_ids] = 0
    env._holding[env_ids] = False
    env._collection_timer[env_ids] = 0.0
    env._all_collected_buf[env_ids] = False
    env._collected_step_flag[env_ids] = 0.0

    # Spawn pose: small jitter above the first waypoint to avoid all drones
    # stacking on the exact point.
    first_wp = env._waypoint_positions[env_ids, 0]  # [n, 3]
    jitter = torch.zeros(n, 3, device=env.device)
    jitter[:, 0].uniform_(-0.3, 0.3)
    jitter[:, 1].uniform_(-0.3, 0.3)
    jitter[:, 2].uniform_(-0.1, 0.1)
    start_pos = first_wp + jitter
    start_pos[:, 2] = start_pos[:, 2].clamp(min=env.cfg.min_height + 0.1)

    # Identity quaternion (level, facing world +X).
    start_quat = torch.zeros(n, 4, device=env.device)
    start_quat[:, 0] = 1.0
    root_pose = torch.cat([start_pos, start_quat], dim=-1)

    root_vel = torch.zeros(n, 6, device=env.device)
    joint_pos = env._robot.data.default_joint_pos[env_ids]
    joint_vel = env._robot.data.default_joint_vel[env_ids]

    env._robot.write_root_pose_to_sim(root_pose, env_ids)
    env._robot.write_root_velocity_to_sim(root_vel, env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    env._route_episode_start_position[env_ids] = start_pos

    env._actions_buf[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0
    # Reset progress and finite-difference state to the spawned pose so the
    # first control step has neither a false progress spike nor false speed.
    distance = torch.linalg.vector_norm(first_wp - start_pos, dim=1)
    env._curr_dist_to_waypoint[env_ids] = distance
    env._previous_dist_to_waypoint[env_ids] = distance
    env._reward_previous_dist_to_waypoint[env_ids] = distance
    env._reward_dist_to_waypoint[env_ids] = distance
    env._mocap_prev_pos_w[env_ids] = start_pos
    env._mocap_lin_vel_b[env_ids] = 0.0

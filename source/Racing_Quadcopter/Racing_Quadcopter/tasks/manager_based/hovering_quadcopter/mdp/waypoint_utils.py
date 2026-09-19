# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step waypoint tracking logic for the hovering env.

Implements the two-radius proximity hysteresis and the dwell-time
"collection" counter. Run once per RL step from the env, before
TerminationManager / RewardManager are evaluated, so reward terms can read
``_collected_step_flag``, ``_holding``, ``_collection_timer``, and the
updated ``_current_wp_idx``.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def _gather_current_waypoint(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World-frame position of the per-env active waypoint. Shape [N, 3]."""
    idx = env._current_wp_idx
    env_arange = torch.arange(env.num_envs, device=env.device)
    return env._waypoint_positions[env_arange, idx]


def update_waypoint_state(env: ManagerBasedRLEnv) -> None:
    """Refresh proximity / dwell state and detect collections this step.

    State buffers used:
        env._waypoint_positions   [N, G_max, 3]   world frame, padded
        env._waypoint_counts      [N]             true count per env
        env._collection_time      [N]             T_c per env
        env._current_wp_idx       [N] long        active target
        env._holding              [N] bool        in proximity → counting
        env._collection_timer     [N] float       seconds spent in holding
        env._collected_step_flag  [N] float       1.0 on the step a wp is collected
        env._all_collected_buf    [N] bool        episode-complete flag
        env._episode_collected_count [N] float    running total this episode
    """
    n = env.num_envs
    drone_pos = env._robot.data.root_pos_w  # [N, 3]
    target_pos = _gather_current_waypoint(env)
    dist = torch.norm(target_pos - drone_pos, dim=1)  # [N]

    # Preserve reward distances before potentially advancing the waypoint.
    # The reward manager runs after this function, while observations should
    # already expose the next target after a successful collection.
    env._reward_previous_dist_to_waypoint[:] = env._previous_dist_to_waypoint
    env._reward_dist_to_waypoint[:] = dist

    r_capture = float(env.cfg.r_capture)
    r_hold = float(env.cfg.r_hold)
    step_dt = float(env.step_dt)

    holding = env._holding
    timer = env._collection_timer
    tc = env._collection_time

    lin_speed = torch.linalg.vector_norm(env._robot.data.root_lin_vel_b, dim=1)
    ang_speed = torch.linalg.vector_norm(env._robot.data.root_ang_vel_b, dim=1)
    tilt_sin = torch.linalg.vector_norm(env._robot.data.projected_gravity_b[:, :2], dim=1)
    linear_ok = lin_speed <= float(env.cfg.collection_max_linear_speed_m_s)
    angular_ok = ang_speed <= float(env.cfg.collection_max_angular_speed_rad_s)
    tilt_ok = tilt_sin <= float(env.cfg.collection_max_tilt_sin)
    stable = linear_ok & angular_ok & tilt_ok
    in_capture_radius = dist < r_capture
    in_hold_radius = dist < r_hold
    requires_dwell = tc > 1.0e-6

    # Fly-through cells (T_c == 0) intentionally collect on spatial capture
    # alone.  Requiring a near-stationary hover there made the early curriculum
    # contradict its own zero-dwell training examples.
    inside_capture = in_capture_radius & ((~requires_dwell) | stable)
    inside_hold = in_hold_radius & ((~requires_dwell) | stable)

    env._hold_capture_steps += in_capture_radius.float()
    env._hold_linear_ok_capture_steps += (in_capture_radius & linear_ok).float()
    env._hold_angular_ok_capture_steps += (in_capture_radius & angular_ok).float()
    env._hold_tilt_ok_capture_steps += (in_capture_radius & tilt_ok).float()
    env._hold_stable_capture_steps += (in_capture_radius & stable).float()

    # Transition: idle → holding (entered tight radius).
    enter = (~holding) & inside_capture
    if enter.any():
        holding[enter] = True
        timer[enter] = 0.0

    # Stay in holding while we remain inside the lax radius — accumulate dwell.
    stay = holding & inside_hold
    if stay.any():
        timer[stay] = timer[stay] + step_dt
        env._hold_dwell_time_s += stay.float() * step_dt

    # Dropped out of the lax radius → reset proximity state.
    drop = holding & (~inside_hold)
    if drop.any():
        holding[drop] = False
        timer[drop] = 0.0
        env._hold_timer_reset_count += drop.float()

    # Collection condition. T_c == 0 collects the moment we entered capture
    # range, which is precisely when ``enter`` was set this step.
    eps = 1e-6
    collected = holding & (timer + eps >= tc)
    env._collected_step_flag = collected.float()

    # Advance the target waypoint and reset proximity state for collected envs.
    if collected.any():
        env._route_confirmed_segments[collected] += 1
        env._episode_collected_count[collected] = env._episode_collected_count[collected] + 1.0
        new_idx = env._current_wp_idx[collected] + 1
        counts = env._waypoint_counts[collected]
        # Detect episode completion before wrapping the index.
        completed_all = new_idx >= counts
        env._all_collected_buf[collected] = completed_all
        safe_counts = counts.clamp(min=1)
        env._current_wp_idx[collected] = new_idx % safe_counts
        holding[collected] = False
        timer[collected] = 0.0

    # Prepare the next step's progress reference.  Collected environments now
    # point to a different waypoint, so their reference must be measured to
    # that new target rather than to the one just collected.
    env._previous_dist_to_waypoint[:] = dist
    if collected.any():
        next_target = _gather_current_waypoint(env)
        env._previous_dist_to_waypoint[collected] = torch.linalg.vector_norm(
            next_target[collected] - drone_pos[collected], dim=1
        )

    env._holding = holding
    env._collection_timer = timer
    env._curr_dist_to_waypoint = env._previous_dist_to_waypoint

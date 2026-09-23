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
    drone_pos = env._robot.data.root_pos_w  # [N, 3]
    target_pos = _gather_current_waypoint(env)
    dist = torch.norm(target_pos - drone_pos, dim=1)  # [N]

    r_capture = float(env.cfg.r_capture)
    r_hold = float(env.cfg.r_hold)
    step_dt = float(env.step_dt)

    holding = env._holding
    timer = env._collection_timer
    tc = env._collection_time

    inside_capture = dist < r_capture
    inside_hold = dist < r_hold

    # Snapshot pre-transition state: ``enter`` below mutates ``holding`` in
    # place, and since entering the capture radius also satisfies inside_hold,
    # reading ``holding`` (instead of this snapshot) for ``stay`` would give
    # envs that just entered an extra +step_dt on top of the timer=0.0 just set.
    was_holding = holding.clone()

    # Transition: idle → holding (entered tight radius).
    enter = (~was_holding) & inside_capture
    if enter.any():
        holding[enter] = True
        timer[enter] = 0.0

    # Stay in holding while we remain inside the lax radius — accumulate dwell.
    stay = was_holding & inside_hold
    if stay.any():
        timer[stay] = timer[stay] + step_dt

    # Dropped out of the lax radius → reset proximity state.
    drop = holding & (~inside_hold)
    if drop.any():
        holding[drop] = False
        timer[drop] = 0.0

    # Collection condition. T_c == 0 collects the moment we entered capture
    # range, which is precisely when ``enter`` was set this step.
    eps = 1e-6
    collected = holding & (timer + eps >= tc)
    env._collected_step_flag = collected.float()

    # Advance the target waypoint and reset proximity state for collected envs.
    if collected.any():
        env._gates_passed[collected] = env._gates_passed[collected] + 1  # legacy alias
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

    env._holding = holding
    env._collection_timer = timer
    env._curr_dist_to_waypoint = dist
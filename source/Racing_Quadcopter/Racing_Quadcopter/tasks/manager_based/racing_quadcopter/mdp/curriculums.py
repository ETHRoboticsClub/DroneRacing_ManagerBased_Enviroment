# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum helpers for collision detection gating."""

from __future__ import annotations

import torch


def check_collision_curriculum_enabled(
    env: object,
    gates_passed_per_env: torch.Tensor,
    passage_velocity_per_env: torch.Tensor,
    passage_centering_per_env: torch.Tensor,
    passage_count_per_env: torch.Tensor,
    min_gates_passed: int = 3,
    min_mean_velocity: float = 2.5,
    max_center_offset: float = 0.15,
) -> torch.Tensor:
    """Return per-env bool [N]: True when collision termination should be active.

    Collision termination is enabled for an env once all three criteria are met
    in the PREVIOUS episode:
      1. gates_passed >= min_gates_passed  — agent can navigate the course.
      2. mean passage velocity >= min_mean_velocity (m/s)  — flying fast enough.
      3. mean lateral passage offset <= max_center_offset (m)  — flying accurately.

    Inputs from refresh_crash_buffers_for_step pass passage_count_per_env=ones
    because _last_episode_mean_velocity and _last_episode_center_offset are already
    averaged at episode end in collect_episode_metrics_before_reset. The division
    inside this function is therefore a no-op (x/1 = x) in that call path.

    Envs that never passed a gate (!has_passages) are treated as passing the velocity
    and centering criteria vacuously — only the gates_passed count gates them out.
    This prevents the trivially-True velocity/centering check from enabling termination
    for an agent that never moved toward a gate.

    Args:
        env: Unused; kept for API parity with other curriculum helpers.
        gates_passed_per_env: [N] float, cumulative gates from the last episode.
        passage_velocity_per_env: [N] float, sum of passage speeds (divided by count below).
        passage_centering_per_env: [N] float, sum of lateral offsets (divided by count below).
        passage_count_per_env: [N] float, number of passages (or ones if already averaged).
        min_gates_passed: Minimum gate count to unlock termination.
        min_mean_velocity: Minimum mean passage speed (m/s).
        max_center_offset: Maximum mean lateral passage offset (m).

    Returns:
        [N] bool tensor: True where all criteria are satisfied.
    """
    enough_gates = gates_passed_per_env >= min_gates_passed

    has_passages = passage_count_per_env > 0
    mean_velocity = torch.where(
        has_passages,
        passage_velocity_per_env / passage_count_per_env,
        torch.zeros_like(passage_velocity_per_env),
    )
    # Vacuously True when no passages occurred so only enough_gates gates these envs.
    good_velocity = (mean_velocity >= min_mean_velocity) | (~has_passages)

    mean_centering = torch.where(
        has_passages,
        passage_centering_per_env / passage_count_per_env,
        torch.zeros_like(passage_centering_per_env),
    )
    good_centering = (mean_centering <= max_center_offset) | (~has_passages)

    return enough_gates & good_velocity & good_centering

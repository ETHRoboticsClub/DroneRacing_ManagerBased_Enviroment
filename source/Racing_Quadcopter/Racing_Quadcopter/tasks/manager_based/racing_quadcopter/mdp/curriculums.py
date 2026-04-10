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
    """Return [N] bool: which envs have collision curriculum enabled (env unused; API parity)."""
    enough_gates = gates_passed_per_env >= min_gates_passed

    has_passages = passage_count_per_env > 0
    mean_velocity = torch.where(
        has_passages,
        passage_velocity_per_env / passage_count_per_env,
        torch.zeros_like(passage_velocity_per_env),
    )
    good_velocity = (mean_velocity >= min_mean_velocity) | (~has_passages)

    mean_centering = torch.where(
        has_passages,
        passage_centering_per_env / passage_count_per_env,
        torch.zeros_like(passage_centering_per_env),
    )
    good_centering = (mean_centering <= max_center_offset) | (~has_passages)

    return enough_gates & good_velocity & good_centering

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for racing quadcopter."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .curriculums import check_collision_curriculum_enabled


def refresh_crash_buffers_for_step(env: ManagerBasedRLEnv) -> None:
    """Contact collision + curriculum flags + :attr:`_crashed_buf` for rewards. Call from env after physics."""
    env._collision_detection_enabled = check_collision_curriculum_enabled(
        env=env,
        gates_passed_per_env=env._last_episode_gates_passed,
        passage_velocity_per_env=env._last_episode_mean_velocity,
        passage_centering_per_env=env._last_episode_center_offset,
        passage_count_per_env=torch.ones_like(env._last_episode_gates_passed),
        min_gates_passed=env.cfg.curriculum_min_gates_passed,
        min_mean_velocity=env.cfg.curriculum_min_mean_velocity,
        max_center_offset=env.cfg.curriculum_max_center_offset,
    )

    net_contact_forces = env._contact_sensor.data.net_forces_w_history
    collision_detected = torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, env._undesired_contact_body_ids], dim=-1),
            dim=1,
        )[0]
        > env.cfg.collision_force_threshold,
        dim=1,
    )

    upside = env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold
    env._collision_termination_buf = collision_detected
    env._crashed_buf = collision_detected | upside


def upside_down(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold


def runaway_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.norm(env._robot.data.root_lin_vel_w, dim=1) > env.cfg.max_velocity


def collision_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._collision_termination_buf


def lap_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not env.cfg.eval_mode:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._gates_passed >= env._num_gates

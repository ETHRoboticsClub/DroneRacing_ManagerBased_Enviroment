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
    """Update _crashed_buf and _collision_termination_buf after each physics step.

    Two separate buffers are maintained intentionally:
      _crashed_buf            = contact force > threshold  OR  upside-down
      _collision_termination_buf = contact force > threshold ONLY

    The crash PENALTY (_crashed_buf) always fires for both conditions so the agent
    learns to avoid poor orientations even before the curriculum enables termination.
    The termination (_collision_termination_buf) only fires for hard contacts and is
    additionally gated by _collision_detection_enabled (the curriculum).  This means
    upside-down drones are penalised but not terminated until they are competent
    enough that early termination provides a useful training signal.

    Called from update_racing_state() in rewards.py after sensors are refreshed.
    """
    # Re-evaluate the curriculum gate using last-episode metrics (not current-episode,
    # because those are mid-episode and would give noisy signals).
    env._collision_detection_enabled = check_collision_curriculum_enabled(
        env=env,
        gates_passed_per_env=env._last_episode_gates_passed,
        passage_velocity_per_env=env._last_episode_mean_velocity,
        passage_centering_per_env=env._last_episode_center_offset,
        # passage_count_per_env is passed as ones so the curriculum function receives
        # the already-averaged _last_episode_mean_velocity / _last_episode_center_offset
        # directly (no further averaging needed inside check_collision_curriculum_enabled).
        passage_count_per_env=torch.ones_like(env._last_episode_gates_passed),
        min_gates_passed=env.cfg.curriculum_min_gates_passed,
        min_mean_velocity=env.cfg.curriculum_min_mean_velocity,
        max_center_offset=env.cfg.curriculum_max_center_offset,
    )

    # net_forces_w_history: [N, history_steps, num_bodies, 3] world-frame contact forces.
    net_contact_forces = env._contact_sensor.data.net_forces_w_history
    # For each env: take the norm over the 3 force components, max over history steps,
    # then check whether any body exceeds the threshold.
    collision_detected = torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, env._undesired_contact_body_ids], dim=-1),
            dim=1,
        )[0]
        > env.cfg.collision_force_threshold,
        dim=1,
    )

    # projected_gravity_b[:, 2] > 0 means gravity has a positive body-Z component,
    # i.e. the drone's +Z is pointing downward → it is upside-down.
    upside = env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold

    # _collision_termination_buf does NOT include upside-down on purpose; see docstring.
    # Gated by _collision_detection_enabled: early in training all envs are False here,
    # so episodes always run to timeout even when collisions occur.
    env._collision_termination_buf = collision_detected & env._collision_detection_enabled
    env._crashed_buf = collision_detected | upside


def upside_down(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when the drone's body +Z points downward (gravity positive in body Z). Shape [N]."""
    return env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold


def runaway_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when world-frame speed exceeds cfg.max_velocity. Shape [N]."""
    return torch.norm(env._robot.data.root_lin_vel_w, dim=1) > env.cfg.max_velocity


def collision_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True for contact-force collisions that pass the curriculum gate. Shape [N].

    Reads the pre-computed _collision_termination_buf set by refresh_crash_buffers_for_step.
    Does NOT re-detect collision here to avoid double-reading sensor history every step.
    """
    return env._collision_termination_buf


def lap_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when the env has passed every gate at least once (eval mode only). Shape [N].

    Disabled during training (returns all-False) to let episodes run until timeout or
    crash; the gate-passage reward already incentivises completing laps.
    """
    if not env.cfg.eval_mode:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Compare against per-environment gate counts (supports padded gate arrays)
    return env._gates_passed >= env._gate_counts

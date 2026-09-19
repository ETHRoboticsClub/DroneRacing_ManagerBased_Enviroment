# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum helpers for collision detection gating and staged training."""

from __future__ import annotations

import torch

from Racing_Quadcopter.tasks.manager_based.shared_features import activate_speed_limit_randomization

from collections.abc import Sequence

_STAGE1_BOOST = 1.0  # multiplier applied to crash and angular_velocity in stage 1


def init_two_stage_curriculum(env: object) -> None:
    """Boost stage-1 weights immediately after managers are built.

    Reads the original weights from the reward manager via the public API and stores
    them on the env for restoration when advancing to stage 2.
    Call once from load_managers, after super().load_managers().
    """
    if env._curriculum_stage != 0:
        return
    env._stage2_crash_weight = env.reward_manager.get_term_cfg("crash").weight
    env._stage2_ang_vel_weight = env.reward_manager.get_term_cfg("angular_velocity").weight

    crash_cfg = env.reward_manager.get_term_cfg("crash")
    crash_cfg.weight = env._stage2_crash_weight * _STAGE1_BOOST
    env.reward_manager.set_term_cfg("crash", crash_cfg)

    ang_cfg = env.reward_manager.get_term_cfg("angular_velocity")
    ang_cfg.weight = env._stage2_ang_vel_weight * _STAGE1_BOOST
    env.reward_manager.set_term_cfg("angular_velocity", ang_cfg)


def apply_staged_curriculum(
    env: object, _env_ids: Sequence[int]
) -> dict[str, float] | None:
    """Curriculum term: advance through up to three training stages.

    Stage 0 → 1  (after curriculum_stage1_steps):
        Restore crash and angular_velocity reward weights; upside-down termination
        is disabled automatically because it checks _curriculum_stage == 0.

    Stage 1 → 2  (after curriculum_stage2_steps, only when > 0):
        Enable recovery spawns — a fraction of envs each episode spawn with a
        random orientation and initial velocity (cfg.recovery_spawn_prob).
    """
    if getattr(env.cfg, "eval_mode", False):
        return {"Curriculum/stage": float(env._curriculum_stage)}

    stage = env._curriculum_stage

    if stage == 0 and env.common_step_counter >= env.cfg.curriculum_stage1_steps:
        env._curriculum_stage = 1
        activate_speed_limit_randomization(env)

        crash_cfg = env.reward_manager.get_term_cfg("crash")
        crash_cfg.weight = env._stage2_crash_weight
        env.reward_manager.set_term_cfg("crash", crash_cfg)

        ang_cfg = env.reward_manager.get_term_cfg("angular_velocity")
        ang_cfg.weight = env._stage2_ang_vel_weight
        env.reward_manager.set_term_cfg("angular_velocity", ang_cfg)

        print(f"[Curriculum] Advanced to stage 1 (full racing) at step {env.common_step_counter}")

    elif stage == 1:
        stage2_steps = getattr(env.cfg, "curriculum_stage2_steps", 0)
        if stage2_steps > 0 and env.common_step_counter >= stage2_steps:
            env._curriculum_stage = 2
            print(
                f"[Curriculum] Advanced to stage 2 (recovery spawns, "
                f"prob={env.cfg.recovery_spawn_prob:.2f}) at step {env.common_step_counter}"
            )

    return {"Curriculum/stage": float(env._curriculum_stage)}


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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination conditions for the racing quadcopter.

Each function returns a boolean tensor of shape [N] indicating which environments
should reset this step.  They are registered in :class:`TerminationsCfg` and
evaluated by :class:`~isaaclab.managers.TerminationManager` every step.

The per-step crash buffer refresh that feeds into ``collision_termination`` lives in
:mod:`.gate_utils` (``refresh_crash_buffers_for_step``), since it is a state update
called from the env's step loop rather than a termination condition itself.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.terminations import *


def upside_down(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when the drone's body +Z points downward (gravity positive in body Z). Shape [N].

    Active only during curriculum stage 0 (early stabilisation).
    """
    if getattr(env, "_curriculum_stage", 0) == 0:
        return env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def below_height_threshold(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when the drone drops below cfg.min_height in world frame Z. Shape [N]."""
    return env._robot.data.root_pos_w[:, 2] < env.cfg.min_height


def runaway_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when world-frame speed exceeds cfg.max_velocity. Shape [N]."""
    return torch.norm(env._robot.data.root_lin_vel_w, dim=1) > env.cfg.max_velocity


def collision_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True for contact-force collisions that pass the curriculum gate. Shape [N].

    Gated by cfg.crash_termination (master switch).  The per-env curriculum gate
    (_collision_detection_enabled) is refreshed every step by
    gate_utils.refresh_crash_buffers_for_step, which is called from
    update_racing_state before the TerminationManager evaluates this function.
    """
    if not getattr(env.cfg, "crash_termination", True):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._collision_termination_buf


def lap_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when the env has passed every gate at least once (eval mode only). Shape [N].

    Disabled during training (returns all-False) to let episodes run until timeout or
    crash; the gate-passage reward already incentivises completing laps.
    """
    if not env.cfg.eval_mode:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Compare against per-environment gate counts (supports padded gate arrays).
    return env._gates_passed >= env._gate_counts

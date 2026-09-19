# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward / penalty callables for the hovering / waypoint-following env.

Mirrors the direct-env ``HoveringQuadcopterEnv.RewardManager`` (see the old
``isaaclab_tasks/direct/hovering_quadcopter/reward_manager.py``) one-to-one,
plus two new terms for the waypoint-task extension:

  - ``reward_waypoint_collected`` — sparse +1 on the step the active target
    advances. Drives the new collection mechanic.
  - ``reward_holding`` — +1 while the drone is inside the capture radius;
    purely additive and intended to be commentable-out without breaking
    learning of the other terms.

The Isaac Lab ``RewardManager`` multiplies every term by the environment time
step.  Consequently, the functions in this module return *unscaled* factors;
applying ``env.step_dt`` here as well would accidentally scale continuous
rewards by ``dt²``.

Each function returns the **unweighted** factor for that term — shape [N].
``RewardManager`` multiplies by the configured weight from
:class:`WaypointRewardWeights`. Each weight in the cfg is the original
direct-env weight × the old env's ``reward_scale=0.2``; the
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.rewards import *  # noqa: F401, F403


# ---------------------------------------------------------------------------
# Dense waypoint shaping
# ---------------------------------------------------------------------------

def reward_progress_to_waypoint(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bounded reward for reducing distance to the active waypoint.

    The state updater preserves both distances with respect to the target
    that was active during the just-completed control step.  This remains
    correct on a collection step, where the public ``current waypoint`` has
    already advanced for the next observation.
    """
    scale = float(env.cfg.progress_to_waypoint_scale_m)
    delta = env._reward_previous_dist_to_waypoint - env._reward_dist_to_waypoint
    return torch.tanh(delta / scale)


def reward_waypoint_proximity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Broad bounded reward for being near the waypoint."""
    scale = float(env.cfg.waypoint_proximity_scale_m)
    return 1.0 - torch.tanh(env._reward_dist_to_waypoint / scale)


def reward_waypoint_proximity_fine(env: ManagerBasedRLEnv) -> torch.Tensor:
    """High-resolution positive reward around the waypoint centre."""
    scale = float(env.cfg.waypoint_proximity_fine_scale_m)
    return 1.0 - torch.tanh(env._reward_dist_to_waypoint / scale)


def penalty_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``‖v_b‖²`` — discourages drifting / unnecessary motion."""
    return env._robot.data.root_lin_vel_b.square().sum(dim=1)


def penalty_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``‖ω_b‖²`` — without this the drone learns to circle the goal."""
    return env._robot.data.root_ang_vel_b.square().sum(dim=1)


def penalty_near_waypoint_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Extra braking cost that smoothly activates near the active target."""
    radius = float(env.cfg.near_waypoint_velocity_radius_m)
    proximity = torch.exp(-(env._reward_dist_to_waypoint / radius).square())
    return env._robot.data.root_lin_vel_b.square().sum(dim=1) * proximity


def penalty_near_waypoint_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Extra angular-rate cost near the active target."""
    radius = float(env.cfg.near_waypoint_velocity_radius_m)
    proximity = torch.exp(-(env._reward_dist_to_waypoint / radius).square())
    return env._robot.data.root_ang_vel_b.square().sum(dim=1) * proximity


def penalty_action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``‖a_t - a_{t-1}‖²`` — discourages jitter."""
    diff = env._actions_buf - env._prev_actions
    return diff.square().sum(dim=1)


def penalty_crash(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 on crash frames (z out of bounds OR upside-down). Shape [N]."""
    z = env._robot.data.root_pos_w[:, 2]
    out_of_bounds = (z < env.cfg.min_height) | (z > env.cfg.max_height)
    upside = env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold
    return (out_of_bounds | upside).float()


# ---------------------------------------------------------------------------
# Waypoint-task extensions (additive on top of the mirrored hovering terms)
# ---------------------------------------------------------------------------

def reward_waypoint_collected(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse +1 on the step the current waypoint is collected. Shape [N]."""
    return env._collected_step_flag


def reward_holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """+1 while inside the capture radius. Shape [N].

    Optional — designed to also work when the corresponding term is
    commented out of :class:`RewardsCfg`. The other terms (especially the
    cubic distance shaping) provide the necessary gradient on their own.
    """
    return env._holding.float()

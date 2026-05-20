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

Continuous-quantity terms (``lin_vel``, ``ang_vel``, ``distance_to_waypoint``)
multiply by ``env.step_dt`` so they integrate to a per-second magnitude
regardless of policy frequency, matching the original direct env.

Each function returns the **unweighted** factor for that term — shape [N].
``RewardManager`` multiplies by the configured weight from
:class:`WaypointRewardWeights`. Each weight in the cfg is the original
direct-env weight × the old env's ``reward_scale=0.2``; the
``@scale_dataclass_attributes(100.0)`` decorator scales by 100 so SKRL's
``rewards_shaper_scale=0.01`` produces the original effective magnitudes.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.rewards import *  # noqa: F401, F403


# ---------------------------------------------------------------------------
# Mirror of the old direct-env terms (continuous → multiplied by step_dt)
# ---------------------------------------------------------------------------

def reward_distance_to_waypoint(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``(d + 2)^3 * step_dt`` for the current target. Matches the old env.

    Cubic-shifted form gives a strong gradient even at large distances; the
    +2 shift keeps the derivative finite at d=0 and is more aggressive
    than a quadratic at every range. Multiplying by ``step_dt`` converts
    the per-step contribution into a per-second magnitude so the
    integrated penalty over an episode is independent of dt.
    """
    return torch.exp(-0.5*env._curr_dist_to_waypoint**2) * env.step_dt


def reward_hovering_to_waypoint(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``(d + 2)^3 * step_dt`` for the current target. Matches the old env.

    Cubic-shifted form gives a strong gradient even at large distances; the
    +2 shift keeps the derivative finite at d=0 and is more aggressive
    than a quadratic at every range. Multiplying by ``step_dt`` converts
    the per-step contribution into a per-second magnitude so the
    integrated penalty over an episode is independent of dt.
    """
    return torch.exp(-40*env._curr_dist_to_waypoint**2) * env.step_dt



def penalty_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``‖v_b‖² * step_dt`` — discourages drifting / unnecessary motion.

    Linear velocity is taken from sim (not from obs); the policy doesn't
    see it but the critic does via the reward.
    """
    return env._robot.data.root_lin_vel_b.square().sum(dim=1) * env.step_dt


def penalty_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``‖ω_b‖² * step_dt`` — without this the drone learns to circle the goal.

    Comment from the direct-env reward manager: "It learns to circle around
    when this is too low".
    """
    return env._robot.data.root_ang_vel_b.square().sum(dim=1) * env.step_dt


def penalty_action_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``‖a‖²`` — mild regulariser. No dt scaling (event-like)."""
    return env._actions_buf.square().sum(dim=1)


def penalty_oversaturation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``‖a_raw - a_clamped‖²`` — penalises the policy for asking for actions
    outside ``[-1, 1]`` that the action term then clips.

    Reads ``env._unclamped_actions_buf``, which the env stores at the top of
    each ``step`` before the action term clamps & applies noise/lag.
    """
    raw = env._unclamped_actions_buf
    clamped = raw.clamp(-1.0, 1.0)
    return (raw - clamped).square().sum(dim=1)


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

def penalty_max_velocity(env: ManagerBasedRLEnv, threshold: float = 2.0) -> torch.Tensor:
    """``max(0, ‖v_b‖ - threshold)² * step_dt`` — penalises speed above *threshold* (m/s).

    Zero below the cap so the drone can move freely at low speed; the critic
    still sees body-frame velocity via this term when the limit is exceeded.
    """
    speed = torch.norm(env._robot.data.root_lin_vel_b, dim=1)
    speed_excess = (speed - threshold).clamp(min=0.0)
    return speed_excess.square() * env.step_dt


def penalty_high_thrust(env: ManagerBasedRLEnv, threshold: float = 0.7) -> torch.Tensor:
    """``max(0, thrust_scaled - threshold)² * step_dt`` — penalises excess collective thrust.

    ``thrust_scaled = (action[0] + 1) / 2`` in [0, 1], matching
    ``QuadcopterRatesAction`` / Betaflight thrust mapping on ``_actions_buf``.
    """
    thrust_scaled = (env._actions_buf[:, 0] + 1.0) * 0.5
    thrust_excess = (thrust_scaled - threshold).clamp(min=0.0)
    return thrust_excess.square() * env.step_dt
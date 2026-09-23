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
direct-env weight x the old env's ``reward_scale=0.2``; the
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
    return torch.pow(env._curr_dist_to_waypoint + 2.0, 3.0) * env.step_dt


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
    """``‖a_rates‖²`` — mild regulariser on the 3 rate channels only.

    Deliberately excludes the thrust channel (index 0): penalizing it toward
    ``a=0`` implicitly biases the policy toward whatever thrust ``a=0``
    happens to command, which is only the hover point by coincidence of a
    particular ``thrust_to_weight``. See ``penalty_thrust_offset_from_hover``
    for the term that correctly targets the T2W-dependent hover action.
    """
    return env._actions_buf[:, 1:].square().sum(dim=1)


def penalty_thrust_offset_from_hover(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``(a_thrust - a_hover)²`` — pulls the thrust action toward the hover point.

    ``a_thrust=0`` maps to ``thrust_to_weight/2`` g of thrust (see
    ``QuadcopterRatesAction.process_actions``), which only coincides with the
    1g hover point when ``thrust_to_weight == 2``. At the racing profile's
    T2W=4.1, ``a=0`` commands ~2g (net climb) and the true hover action is
    ``2/T2W - 1 ≈ -0.51`` — far from the policy's initial mean (~0, with
    ``init_noise_std=1.0``), so an untrained policy is as likely to climb
    into the ceiling termination as to find the hover reward basin by chance.
    This term gives a dense gradient toward the correct equilibrium from the
    first rollout, independent of whatever T2W the drone profile uses.
    """
    t2w = env.cfg.actions.rates.thrust_to_weight
    hover_action = 2.0 / t2w - 1.0
    return (env._actions_buf[:, 0] - hover_action).square()


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
    """+1 while in the holding state. Shape [N].

    ``env._holding`` is set True on entering ``r_capture`` and stays True
    until the drone drifts back out past the looser ``r_hold`` radius (see
    ``waypoint_utils.update_waypoint_state``) — so this reward covers the
    whole holding zone, not just the tighter capture radius.

    Optional — designed to also work when the corresponding term is
    commented out of :class:`RewardsCfg`. The other terms (especially the
    cubic distance shaping) provide the necessary gradient on their own.
    """
    return env._holding.float()

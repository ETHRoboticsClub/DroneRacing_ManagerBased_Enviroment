# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter reward and penalty functions (manager-based).

Each public function returns the **unweighted** factor for that term.
:class:`~isaaclab.managers.RewardManager` multiplies by the configured ``weight``
(here ``MANAGER_SCALE * direct_weight``, i.e. 100 × each direct-env default weight).
SKRL's ``rewards_shaper_scale=0.01`` then divides by 100, so the final effective
reward per term matches the direct-env magnitudes: factor × weight × 0.01 = factor × direct_weight.

Gate geometry and per-step state updates live in :mod:`.gate_utils`.

See module-level ``WEIGHT_*`` constants for use in :class:`RewardsCfg`.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.rewards import *

from isaaclab.utils.math import quat_apply


# ---------------------------------------------------------------------------
# Reward / penalty functions
# Each returns an unweighted scalar per env [N]. The RewardManager multiplies
# by the corresponding WEIGHT_* constant configured in RewardsCfg.
# ---------------------------------------------------------------------------

def reward_gate_passage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary 1.0 on gate passage frames; 0.0 otherwise. Shape [N]."""
    return env._gate_passed_buf


def reward_gate_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance reduction toward current gate this step, clamped to [-1, 1] m."""
    drone_pos = env._robot.data.root_pos_w
    gate_pos = env._curr_gate_pos
    dist_to_gate = torch.norm(gate_pos - drone_pos, dim=1)
    prev_dist = torch.norm(env._prev_drone_pos - gate_pos, dim=1)
    # Positive when drone moves closer; clamped to prevent large single-step spikes.
    return (prev_dist - dist_to_gate).clamp(-1.0, 1.0)


def reward_gate_centering(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Exponential alignment reward: peaks when drone is centred on gate axis.

    Only active on the approach side (dist_to_plane <= 0) to avoid rewarding
    alignment from behind the gate after it has been passed.
    Formula: exp(-3 * lateral / size) * exp(-0.5 * |plane_dist|)
    """
    gate_size = env._gate_sizes  # [N] per-env aperture in metres
    centering_value = torch.exp(-3.0 * env._curr_dist_to_gate_lateral / gate_size)   # falls off with lateral offset
    proximity_factor = torch.exp(-0.5 * env._curr_dist_to_gate_forward.abs())         # falls off with distance to plane
    return centering_value * proximity_factor * env._on_approach_gate_side.float()


def reward_velocity_toward_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dot product of world-frame velocity with normalized gate direction, approach side only.

    Clamped to >= 0 so backing away from the gate gives zero reward, not negative.
    """
    drone_vel = env._robot.data.root_lin_vel_w
    vel_toward_gate = (drone_vel * env._curr_dist_to_gate_vector_normalized).sum(dim=1).clamp(min=0.0)
    return vel_toward_gate * env._on_approach_gate_side.float()


def penalty_velocity_limit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Quadratic penalty for exceeding the per-env velocity cap (m/s)."""
    drone_vel = env._robot.data.root_lin_vel_w
    speed = torch.norm(drone_vel, dim=1)
    speed_excess = (speed - env._velocity_limit).clamp(min=0.0)  # zero below limit
    return speed_excess**2


def penalty_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared angular deviation from upright in body frame.

    projected_gravity_b points in the direction gravity acts in the body frame.
    For a perfectly upright drone, gravity is along -Z in body frame → [0,0,-1].
    The dot product with [0,0,-1] gives cos(tilt_angle); acos gives the tilt.
    Clamp to (-0.98, 0.98) to keep acos well-conditioned near 0° and 180°.
    """
    projected_gravity = env._robot.data.projected_gravity_b  # body frame [N, 3]
    upright = torch.tensor([0.0, 0.0, -1.0], device=projected_gravity.device)
    orientation_error_angle = torch.acos(torch.clamp((projected_gravity * upright).sum(dim=1), -0.98, 0.98))
    return torch.square(orientation_error_angle)


def penalty_angular_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared body-frame angular velocities (rad/s)^2. Discourages spinning."""
    drone_angvel = env._robot.data.root_ang_vel_b  # body frame [N, 3]
    return drone_angvel.square().sum(dim=1)


def penalty_action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared action deltas between consecutive steps. Discourages jitter."""
    action_diff = env._actions_buf - env._prev_actions  # [N, 4]
    return action_diff.square().sum(dim=1)


def penalty_action_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared action values. Mild regulariser to prevent saturating actuators."""
    return env._actions_buf.square().sum(dim=1)


def penalty_crash(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary 1.0 on crash frames (contact force OR upside-down). Shape [N]."""
    return env._crashed_buf.float()


def penalty_wrong_direction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary 1.0 on frames where the drone crossed the current gate from behind. Shape [N].

    Paired with the ``wrong_direction`` termination: provides a sharp local penalty so the
    gradient does not have to flow through the value function to teach gate orientation.
    """
    return env._wrong_crossing_buf.float()


def penalty_wrong_side(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise being on the exit side of the gate without having passed it.

    Suppressed on gate-passage frames (_gate_passed_buf=1) to avoid a spurious
    penalty on the step the drone legitimately crosses through.
    """
    return (~env._on_approach_gate_side).float() * (1.0 - env._gate_passed_buf)


####################
# MONORACE REWARDS #
####################


def penalty_perception(
    env: ManagerBasedRLEnv, theta_threshold: float = math.pi / 6.0
) -> torch.Tensor:
    """The perception penalty encourages the drone to keep the next gate within the forward-facing
       camera's view:
    """
    gate_pos = env._curr_gate_pos  # [N, 3]
    kp_idx: int = env._camera_keypoint_body_idx
    root_quat = env._robot.data.root_quat_w

    anchor_pos = env._robot.data.body_pos_w[:, kp_idx]
    cam_axis = env._x_local.expand(env.num_envs, -1)
    cam_quat = root_quat

    to_gate = gate_pos - anchor_pos
    dist = torch.norm(to_gate, dim=1)
    to_gate_norm = to_gate / (dist.unsqueeze(-1) + 1e-8)

    cam_forward = quat_apply(cam_quat, cam_axis)
    cam_forward_norm = cam_forward / (torch.norm(cam_forward, dim=1, keepdim=True) + 1e-8)

    cos_theta = torch.sum(cam_forward_norm * to_gate_norm, dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    active = env._on_approach_gate_side & (dist > 0.1)
    penalty = theta * active.float()
    return torch.where(theta > theta_threshold, penalty, torch.zeros_like(theta))

def penalty_action_rate_thresh(
    env: ManagerBasedRLEnv, delta_omega_norm_thresh: float = 0.05
) -> torch.Tensor:
    """Penalise large per-step motor speed jumps (|Δω|/ω_max minus a threshold, summed over rotors)."""
    omega_max = env.cfg.motor_omega_max_rad_s
    diff = (env._motor_speeds - env._prev_motor_speeds).abs() / omega_max
    return (diff - delta_omega_norm_thresh).clamp(min=0.0).sum(dim=1)


def penalty_low_action(env: ManagerBasedRLEnv, low_floor_norm: float = 0.5) -> torch.Tensor:
    """This penalty was introduced after observing that an early policy exhibited highly aggressive bang–
        bang control, switching almost exclusively between minimum and maximum motor commands,
        which led to motor overheating. The low action penalty discourages low actions:"""
    omega_max = env.cfg.motor_omega_max_rad_s
    w = env._motor_speeds / omega_max
    return (low_floor_norm - w).clamp(min=0.0).sum(dim=1)


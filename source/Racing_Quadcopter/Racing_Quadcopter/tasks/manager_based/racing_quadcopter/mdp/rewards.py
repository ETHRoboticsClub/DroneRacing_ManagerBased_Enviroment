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

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .gate_utils import gather_current_gate, gate_forward, gate_right, gate_up


# ---------------------------------------------------------------------------
# Reward weight scaling
#
# The manager framework computes: reward = term_output * weight.
# The direct env applied weights internally, so to match direct-env magnitudes
# we set weight = MANAGER_SCALE * direct_weight.
# SKRL additionally applies rewards_shaper_scale=0.01 on top.
# ---------------------------------------------------------------------------

MANAGER_SCALE = 100.0

# Direct-env default weights (from RewardWeights in reward_manager.py).
DIRECT_GATE_PASSAGE = 250.0
DIRECT_GATE_PROGRESS = 20.0
DIRECT_GATE_CENTERING = 0.5
DIRECT_VELOCITY_TOWARD_GATE = 2.0
DIRECT_VELOCITY_LIMIT = -0.05
DIRECT_ORIENTATION = -0.005
DIRECT_ANGULAR_VELOCITY = -0.001
DIRECT_ACTION_SMOOTHNESS = -0.0001
DIRECT_ACTION_MAGNITUDE = -0.001
DIRECT_CRASH = -50.0
DIRECT_WRONG_SIDE = -0.1

# Use these in RewTerm(weight=...) — effective reward matches direct * MANAGER_SCALE per term.
WEIGHT_GATE_PASSAGE = MANAGER_SCALE * DIRECT_GATE_PASSAGE
WEIGHT_GATE_PROGRESS = MANAGER_SCALE * DIRECT_GATE_PROGRESS
WEIGHT_GATE_CENTERING = MANAGER_SCALE * DIRECT_GATE_CENTERING
WEIGHT_VELOCITY_TOWARD_GATE = MANAGER_SCALE * DIRECT_VELOCITY_TOWARD_GATE
WEIGHT_VELOCITY_LIMIT = MANAGER_SCALE * DIRECT_VELOCITY_LIMIT
WEIGHT_PERCEPTION = MANAGER_SCALE * -0.01
WEIGHT_ORIENTATION = MANAGER_SCALE * DIRECT_ORIENTATION
WEIGHT_ANGULAR_VELOCITY = MANAGER_SCALE * DIRECT_ANGULAR_VELOCITY
WEIGHT_ACTION_SMOOTHNESS = MANAGER_SCALE * DIRECT_ACTION_SMOOTHNESS
WEIGHT_ACTION_MAGNITUDE = MANAGER_SCALE * DIRECT_ACTION_MAGNITUDE
WEIGHT_CRASH = MANAGER_SCALE * DIRECT_CRASH
WEIGHT_WRONG_SIDE = MANAGER_SCALE * DIRECT_WRONG_SIDE


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
    gate_pos, _ = gather_current_gate(env)
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
    drone_pos = env._robot.data.root_pos_w
    gate_pos, gate_ori = gather_current_gate(env)
    gate_size = env._gate_sizes  # [N] per-env aperture in metres

    to_gate = gate_pos - drone_pos
    gate_fwd = gate_forward(env, gate_ori)
    g_right = gate_right(env, gate_ori)
    g_up = gate_up(env, gate_ori)

    # Signed distance to gate plane: negative = approach side.
    dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)
    on_approach_side = dist_to_plane <= 0.0  # only reward while approaching

    lateral_right = ((-to_gate) * g_right).sum(dim=1)
    lateral_up = ((-to_gate) * g_up).sum(dim=1)
    lateral_dist = torch.sqrt(lateral_right**2 + lateral_up**2)

    centering_value = torch.exp(-3.0 * lateral_dist / gate_size)   # falls off with lateral offset
    proximity_factor = torch.exp(-0.5 * dist_to_plane.abs())        # falls off with distance to plane
    return centering_value * proximity_factor * on_approach_side.float()


def penalty_perception(env: ManagerBasedRLEnv, theta_threshold: float = torch.pi / 3.0) -> torch.Tensor:
    """Penalty when next gate leaves the forward camera cone.

    p_perc = lambda_perc * theta_cam if theta_cam > theta_threshold else 0
    """
    from isaaclab.utils.math import quat_apply

    drone_pos = env._robot.data.root_pos_w
    drone_quat = env._robot.data.root_quat_w
    gate_pos, gate_ori = gather_current_gate(env)

    to_gate = gate_pos - drone_pos
    to_gate_norm = to_gate / (torch.norm(to_gate, dim=1, keepdim=True) + 1e-8)

    # Drone +Y is forward (Crazyflie body convention).
    drone_forward = quat_apply(drone_quat, env._y_local.expand(env.num_envs, -1))
    drone_forward_norm = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)

    gate_fwd = gate_forward(env, gate_ori)
    dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)
    on_approach_side = dist_to_plane <= 0.0

    cos_theta = torch.sum(drone_forward_norm * to_gate_norm, dim=1).clamp(-1.0, 1.0)
    theta_cam = torch.acos(cos_theta)  # angle between drone forward and gate direction [N]

    penalty = theta_cam * on_approach_side.float()
    return torch.where(theta_cam > theta_threshold, penalty, torch.zeros_like(theta_cam))


def reward_velocity_toward_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dot product of world-frame velocity with normalized gate direction, approach side only.

    Clamped to >= 0 so backing away from the gate gives zero reward, not negative.
    """
    drone_pos = env._robot.data.root_pos_w
    drone_vel = env._robot.data.root_lin_vel_w
    gate_pos, gate_ori = gather_current_gate(env)

    to_gate = gate_pos - drone_pos
    dist_to_gate = torch.norm(to_gate, dim=1)
    to_gate_normalized = to_gate / (dist_to_gate.unsqueeze(-1) + 1e-8)

    gate_fwd = gate_forward(env, gate_ori)
    dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)
    on_approach_side = dist_to_plane <= 0.0

    vel_toward_gate = (drone_vel * to_gate_normalized).sum(dim=1).clamp(min=0.0)
    return vel_toward_gate * on_approach_side.float()


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


def penalty_wrong_side(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise being on the exit side of the gate without having passed it.

    Suppressed on gate-passage frames (_gate_passed_buf=1) to avoid a spurious
    penalty on the step the drone legitimately crosses through.
    """
    drone_pos = env._robot.data.root_pos_w
    gate_pos, gate_ori = gather_current_gate(env)
    to_gate = gate_pos - drone_pos
    gate_fwd = gate_forward(env, gate_ori)
    dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)
    on_approach_side = dist_to_plane <= 0.0
    return (~on_approach_side).float() * (1.0 - env._gate_passed_buf)

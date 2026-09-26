# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter reward and penalty functions (gate racing, then waypoint following).

Each public function returns the **unweighted** factor for that term.
:class:`~isaaclab.managers.RewardManager` multiplies it by the configured ``weight`` and by
``step_dt``; weights live in each task's cfg.

Gate geometry and per-step state updates live in :mod:`.gate_utils`.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.rewards import *

from isaaclab.utils.math import quat_apply
from .shared_features import (
    penalty_angular_velocity_limit,
    penalty_velocity_limit,
)


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


def reward_gate_centering(env: ManagerBasedRLEnv, reference_speed: float = 15.0) -> torch.Tensor:
    """Reward centred *forward motion* through the current gate.

    The alignment potential is exponential in lateral and plane distance, but
    it is gated by positive velocity along the gate's passage direction.  This
    makes the time integral approximately depend on distance travelled through
    the well-centred region, rather than on dwell time: a stationary drone earns
    zero and a slower drone cannot collect extra reward just by taking more
    simulation steps near a gate.

    ``reference_speed`` is the velocity at which the motion factor reaches one;
    faster motion is clipped so this dense shaping term cannot grow unbounded.
    """
    if reference_speed <= 0.0:
        raise ValueError(f"reference_speed must be positive, got {reference_speed}.")
    gate_size = env._gate_sizes  # [N] per-env aperture in metres
    centering_value = torch.exp(-3.0 * env._curr_dist_to_gate_lateral / gate_size)   # falls off with lateral offset
    proximity_factor = torch.exp(-0.5 * env._curr_dist_to_gate_forward.abs())         # falls off with distance to plane
    forward_speed = (env._robot.data.root_lin_vel_w * env._curr_gate_forward).sum(dim=1)
    motion_factor = (forward_speed / reference_speed).clamp(min=0.0, max=1.0)
    return centering_value * proximity_factor * motion_factor * env._on_approach_gate_side.float()


def reward_velocity_toward_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dot product of world-frame velocity with normalized gate direction, approach side only.

    Clamped to >= 0 so backing away from the gate gives zero reward, not negative.
    """
    drone_vel = env._robot.data.root_lin_vel_w
    vel_toward_gate = (drone_vel * env._curr_dist_to_gate_vector_normalized).sum(dim=1).clamp(min=0.0)
    return vel_toward_gate * env._on_approach_gate_side.float()


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


def penalty_action_smoothness(env: ManagerBasedRLEnv, skip_first_step: bool = True) -> torch.Tensor:
    """Sum of squared action deltas between consecutive steps. Discourages jitter.

    With ``skip_first_step`` the first step after reset is excluded to avoid comparing
    against a reset/placeholder previous action value.
    """
    action_diff = env._actions_buf - env._prev_actions  # [N, 4]
    penalty = action_diff.square().sum(dim=1)
    if not skip_first_step:
        return penalty
    valid = (env.episode_length_buf > 1).to(penalty.dtype)
    return penalty * valid


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
    env: ManagerBasedRLEnv,
    theta_threshold: float = math.radians(15.0),
    max_penalty: float = 10.0,
) -> torch.Tensor:
    """Perception penalty: keep the next gate inside the FPV camera frustum, with a steep ramp
    once the line-of-sight angle approaches/leaves the field of view.

    Shape: ``max(0, tan(θ) − tan(threshold))``, clamped to ``max_penalty``. The tangent form is
    the natural pinhole projection — it equals the normalized pixel offset of the gate from the
    image center, so the penalty is "how far off-screen the gate is" in image-plane units.
    Compared with a linear-in-θ form, it ramps far faster as θ exceeds the threshold and
    diverges as the gate goes lateral (θ → 90°), driving tight optical-axis alignment.

    Default ``theta_threshold = 15°`` is well inside the camera's horizontal half-FOV
    (≈32° for focal=2.0, aperture=2.5) so the penalty starts pushing before the gate
    actually leaves the frame.

    Uses the ``camera_keypoint`` body's own world pose (position AND orientation), so the
    camera's URDF-defined tilt relative to ``base_link`` is respected. The optical axis is the
    camera-frame ``+X``; rotating it by the keypoint's world quaternion yields the true
    line-of-sight in world frame.
    """
    gate_pos = env._curr_gate_pos  # [N, 3]
    kp_idx: int = env._camera_keypoint_body_idx

    anchor_pos = env._robot.data.body_pos_w[:, kp_idx]
    cam_quat = env._robot.data.body_quat_w[:, kp_idx]
    cam_axis = env._x_local.expand(env.num_envs, -1)

    to_gate = gate_pos - anchor_pos
    dist = torch.norm(to_gate, dim=1)
    to_gate_norm = to_gate / (dist.unsqueeze(-1) + 1e-8)

    cam_forward = quat_apply(cam_quat, cam_axis)
    cam_forward_norm = cam_forward / (torch.norm(cam_forward, dim=1, keepdim=True) + 1e-8)

    cos_theta = torch.sum(cam_forward_norm * to_gate_norm, dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    # Clamp below 90° before applying tan: gate behind the camera (θ > 90°) saturates
    # to the ceiling rather than wrapping through tan's discontinuity.
    theta_safe = theta.clamp(max=math.pi / 2 - 1e-3)
    tan_thresh = math.tan(theta_threshold)
    penalty = (torch.tan(theta_safe) - tan_thresh).clamp(min=0.0, max=max_penalty)

    active = env._on_approach_gate_side & (dist > 0.1)
    return penalty * active.float()

def penalty_action_rate_thresh(
    env: ManagerBasedRLEnv, delta_omega_norm_thresh: float = 0.05
) -> torch.Tensor:
    """Penalise large per-step motor speed jumps (|Δω|/ω_max minus a threshold, summed over rotors).

    UNUSED by every task (no RewTerm references it). Racing-only: needs ``env._motor_speeds`` /
    ``_prev_motor_speeds`` and ``cfg.motor_omega_max_rad_s`` (hovering has neither). The cfg field
    ``motor_speed_delta_thresh_rad_s`` is NOT read; the threshold here is normalized.
    """
    omega_max = env.cfg.motor_omega_max_rad_s
    diff = (env._motor_speeds - env._prev_motor_speeds).abs() / omega_max
    return (diff - delta_omega_norm_thresh).clamp(min=0.0).sum(dim=1)


def penalty_low_action(env: ManagerBasedRLEnv, low_floor_norm: float = 0.5) -> torch.Tensor:
    """This penalty was introduced after observing that an early policy exhibited highly aggressive bang–
        bang control, switching almost exclusively between minimum and maximum motor commands,
        which led to motor overheating. The low action penalty discourages low actions.

    UNUSED by every task (no RewTerm references it). Racing-only: needs ``env._motor_speeds`` and
    ``cfg.motor_omega_max_rad_s``.
    """
    omega_max = env.cfg.motor_omega_max_rad_s
    w = env._motor_speeds / omega_max
    return (low_floor_norm - w).clamp(min=0.0).sum(dim=1)


# ---------------------------------------------------------------------------
# Waypoint following (hovering task)
#
# Mirrors the old direct hovering env's reward terms plus the waypoint-collection
# extensions. Each function returns the unweighted factor [N]; the RewardManager
# multiplies by the configured weight and by ``step_dt`` (do not apply dt here).
# ``penalty_action_smoothness`` above is shared (hovering passes ``skip_first_step=False``).
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


def penalty_out_of_bounds_or_upside_down(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 on frames where z leaves [min_height, max_height] or the drone is upside down. Shape [N]."""
    z = env._robot.data.root_pos_w[:, 2]
    out_of_bounds = (z < env.cfg.min_height) | (z > env.cfg.max_height)
    upside = env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold
    return (out_of_bounds | upside).float()


def reward_waypoint_collected(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse +1 on the step the current waypoint is collected. Shape [N]."""
    return env._collected_step_flag


def reward_holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """+1 while the waypoint "holding" state is active (capture radius with r_capture/r_hold
    hysteresis and the collection stability gates). Shape [N].

    Optional — designed to also work when the corresponding term is
    commented out of the hovering ``RewardsCfg``.
    """
    return env._holding.float()

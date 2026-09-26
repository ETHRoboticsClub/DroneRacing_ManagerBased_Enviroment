# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter observations (gate racing, then waypoint following).

All gate-related math lives in this module. The env only exposes state
(robot pose, gate arrays, indices); see :class:`RacingQuadcopterRLEnv`.
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import *
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul
from .shared_features import speed_limits_obs

# Divide world-frame gate offsets (metres) by this before feeding to the network.
# Keeps inputs in roughly [-1, 1] for gates up to ~10 m away.
_REL_POSITION_SCALE: float = 10.0
_REL_VELOCITY_SCALE: float = 4.0 
_REL_ATTITUDE_SCALE: float = math.pi  # radians → roughly [-1, 1] for ``φ, θ, ψ``

def compute_next_gates_rel_and_fwd_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Next-K gates observation: body-frame rel-position + forward + half-width per gate.

    Gate forward is +Y in gate local frame, mapped to world then to body. ``forward_b`` is
    the gate-plane normal expressed in the drone body frame. Gates are square and uniform,
    so rotation about the normal is geometrically inert — ``(forward, half_width)`` fully
    specifies the passable aperture under arbitrary SO(3) gate rotations.
    The per-gate half-width is the effective aperture seen by the policy:
    ``half_eff = (gate_size / 2) * (1 - 2 * gate_size_obs_buffer_frac)``.

    Returns:
        Concatenated ``[N, K*7]``:
        ``[rel_x, rel_y, rel_z, fwd_x, fwd_y, fwd_z, half_width]`` per gate.
    """
    k = env.cfg.observation.num_next_gates  # lookahead count
    n = env.num_envs

    drone_pos_w = env._robot.data.root_pos_w   # world frame [N, 3]
    drone_quat_w = env._robot.data.root_quat_w  # wxyz [N, 4]
    y_local = env._y_local  # gate +Y = forward in gate local frame

    # Build [N, K] gate index table: current_gate + {0,1,...,K-1} mod gate_count.
    gate_idx_raw = env._current_gate_idx.unsqueeze(1) + env._gate_step_indices  # [N, K]
    gate_counts = env._gate_counts.unsqueeze(1)  # [N, 1]
    safe_counts = gate_counts.clone()
    safe_counts[safe_counts == 0] = 1  # guard: envs with 0 gates would cause div-by-zero
    gate_idx = gate_idx_raw % safe_counts  # [N, K] wrapped indices, never reads padding

    # Gather world-frame gate positions and orientations for the K lookahead gates.
    idx3 = gate_idx.unsqueeze(-1).expand(-1, -1, 3)  # [N, K, 3] for position gather
    idx4 = gate_idx.unsqueeze(-1).expand(-1, -1, 4)  # [N, K, 4] for orientation gather
    g_pos_w = torch.gather(env._gate_positions, dim=1, index=idx3)    # [N, K, 3] world
    g_ori_w = torch.gather(env._gate_orientations, dim=1, index=idx4)  # [N, K, 4] wxyz

    # Gate offset in world frame, then rotated into drone body frame.
    offset_w = g_pos_w - drone_pos_w.unsqueeze(1)  # [N, K, 3] world frame
    q_drone = drone_quat_w.unsqueeze(1).expand(-1, k, -1)  # [N, K, 4]
    offset_b = quat_apply_inverse(q_drone, offset_w)  # [N, K, 3] body frame
    offset_b_scaled = offset_b / _REL_POSITION_SCALE   # normalise to ~[-1, 1]

    # Gate forward (+Y) direction: gate local → world → drone body frame.
    y_exp = y_local.view(1, 1, 3).expand(n, k, 3)  # [N, K, 3]
    forward_w = quat_apply(g_ori_w, y_exp)
    forward_b = quat_apply_inverse(q_drone, forward_w)  # [N, K, 3]

    # Effective half-width per gate (broadcast: per-cell sizes are uniform across K).
    buf_frac = float(env.cfg.observation.gate_size_obs_buffer_frac)
    shrink = max(1.0 - 2.0 * buf_frac, 0.0)
    half_eff = (env._gate_sizes * 0.5 * shrink).unsqueeze(1).expand(n, k).unsqueeze(-1)  # [N, K, 1]

    gate_obs = torch.cat([offset_b_scaled, forward_b, half_eff], dim=-1)  # [N, K, 7]
    gate_obs = gate_obs.reshape(n, k * 7)
    return gate_obs


def mono_race_gate_position_velocity_gate_frame(env: ManagerBasedRLEnv) -> torch.Tensor:

    k = env.cfg.observation.mono_race_num_next_gates
    n = env.num_envs

    drone_pos_w = env._robot.data.root_pos_w  # [N, 3]

    # Make sure step indices match k
    gate_step_indices = torch.arange(
        k, device=env.device
    ).unsqueeze(0)  # [1, K]

    # [N, K]
    gate_idx_raw = env._current_gate_idx.unsqueeze(1) + gate_step_indices
    gate_counts = env._gate_counts.unsqueeze(1)  # [N, 1]

    safe_counts = gate_counts.clone()
    safe_counts[safe_counts == 0] = 1

    gate_idx = gate_idx_raw % safe_counts  # [N, K]

    # Gather gate poses
    idx3 = gate_idx.unsqueeze(-1).expand(-1, -1, 3)
    idx4 = gate_idx.unsqueeze(-1).expand(-1, -1, 4)

    g_pos_w = torch.gather(env._gate_positions, 1, idx3)       # [N, K, 3]
    g_ori_w = torch.gather(env._gate_orientations, 1, idx4)    # [N, K, 4]

    # Relative position in gate frame: drone − gate (MonoRace convention, see
    # quad_race_env.py::update_states_gate; matches OBSERVATION_SPACE.md row 0..2).
    offset_w = drone_pos_w.unsqueeze(1) - g_pos_w              # [N, K, 3]

    pos_g = quat_apply_inverse(
        g_ori_w.reshape(-1, 4),
        offset_w.reshape(-1, 3),
    ).reshape(n, k, 3) / _REL_POSITION_SCALE

    # Velocity in gate frame
    drone_lin_vel_w = env._robot.data.root_lin_vel_w           # [N, 3]

    vel_w = drone_lin_vel_w.unsqueeze(1).expand(-1, k, -1)

    vel_g = quat_apply_inverse(
        g_ori_w.reshape(-1, 4),
        vel_w.reshape(-1, 3),
    ).reshape(n, k, 3) / _REL_VELOCITY_SCALE

    # Final observation
    gate_obs = torch.cat([pos_g, vel_g], dim=-1)               # [N, K, 6]

    return gate_obs.reshape(n, k * 6)

def mono_race_future_gates_pose_in_current_gate_frame(
    env: ManagerBasedRLEnv) -> torch.Tensor:
    """For each of the next ``k`` gates: offset from current gate centre to that gate's centre in ``G_i``, plus full relative attitude ``(φ, θ, ψ)``.
    ``k`` from ``cfg.observation.mono_race_future_gates_pose_ahead``; returns ``[N, 6·k]`` as ``[xyz, φ, θ, ψ]`` per future gate, scaled.
    Roll/pitch are nonzero only for tilted gates loaded from track YAMLs (``fixed_gates.py``); procedural gates remain z-axis only."""
    k_cfg = int(getattr(env.cfg.observation, "mono_race_future_gates_pose_ahead", 2))
    k = max(1, k_cfg)
    n = env.num_envs
    dev = env.device
    safe_counts = env._gate_counts.clamp(min=1)
    gi = env._current_gate_idx % safe_counts  # [N]

    offsets = torch.arange(1, k + 1, device=dev, dtype=torch.long).view(1, k).expand(n, -1)
    gate_idx_future = (gi.unsqueeze(1) + offsets) % safe_counts.unsqueeze(1)

    idx_i3 = gi.view(n, 1, 1).expand(n, 1, 3)
    idx_i4 = gi.view(n, 1, 1).expand(n, 1, 4)
    pos_i = torch.gather(env._gate_positions, 1, idx_i3)
    ori_i = torch.gather(env._gate_orientations, 1, idx_i4)

    idx_f3 = gate_idx_future.unsqueeze(-1).expand(-1, -1, 3)
    idx_f4 = gate_idx_future.unsqueeze(-1).expand(-1, -1, 4)
    pos_f = torch.gather(env._gate_positions, 1, idx_f3)
    ori_f = torch.gather(env._gate_orientations, 1, idx_f4)

    delta_w = pos_f - pos_i
    ori_i_k = ori_i.expand(-1, k, -1)
    p_gi = quat_apply_inverse(ori_i_k, delta_w) / _REL_POSITION_SCALE

    q_i = ori_i.squeeze(1).unsqueeze(1).expand(-1, k, -1).reshape(n * k, 4)
    q_f = ori_f.reshape(n * k, 4)
    q_rel = quat_mul(quat_inv(q_i), q_f)
    phi, theta, psi = euler_xyz_from_quat(q_rel, wrap_to_2pi=False)
    rpy_g = torch.stack([phi, theta, psi], dim=-1).view(n, k, 3) / _REL_ATTITUDE_SCALE

    block = torch.cat([p_gi, rpy_g], dim=-1)
    return block.reshape(n, 6 * k)


def mono_race_gate_orientations(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body roll, pitch, yaw (ZYX) relative to the active target gate ``G_i`` (``OBSERVATION_SPACE.md`` §2, rows 6–8).
    Returns ``[N, 3]``, scaled by ``_REL_ATTITUDE_SCALE``."""
    n = env.num_envs
    cur = env._current_gate_idx.reshape(n)
    counts = env._gate_counts.reshape(n).clamp(min=1)
    gi = (cur % counts).unsqueeze(1)
    # ``expand(n,1,4)`` on ``[n,1]`` is invalid (dims align from the right); need a true ``[n,1,4]`` index.
    idx4 = gi.unsqueeze(-1).expand(n, 1, 4)
    q_g0 = torch.gather(env._gate_orientations, 1, idx4).squeeze(1)
    q_bw = env._robot.data.root_quat_w
    q_bg = quat_mul(quat_inv(q_g0), q_bw)
    phi, theta, psi = euler_xyz_from_quat(q_bg, wrap_to_2pi=False)
    return torch.stack([phi, theta, psi], dim=-1) / _REL_ATTITUDE_SCALE


def motor_speeds_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Commanded rotor speeds ``[N, 4]`` normalized by ``cfg.motor_omega_max_rad_s``.

    Racing-only (``env._motor_speeds``). Semantics depend on the action term: mixer-inverse speeds
    for rates/moments, normalized sqrt(thrust) in SITL mode, and the collective speed repeated 4x
    for the second-order rates action.

    """
    omega_max = float(getattr(env.cfg, "motor_omega_max_rad_s"))
    scale = max(omega_max, 1e-6)
    return env._motor_speeds / scale


# ---------------------------------------------------------------------------
# Waypoint following (hovering task)
#
# Deployment constraint: mocap provides position + orientation only. Linear velocity is not
# measured on the real drone (state estimation is not in place yet); gyro (body angular
# velocity) is available from the on-board IMU. Body velocities and projected gravity use the
# stock Isaac Lab terms (``base_lin_vel``, ``base_ang_vel``, ``projected_gravity``).
# ---------------------------------------------------------------------------

# Scale used to bring world-frame relative positions roughly into [-1, 1]
# (largest cell is ~15 m diagonal; 10 m keeps obs near unit magnitude).
_WAYPOINT_REL_POSITION_SCALE: float = 10.0


def mocap_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Filtered body-frame velocity estimated from consecutive mocap poses.

    Returning zero columns cleanly removes this term from both actor and
    critic vectors without mutating shared observation configuration objects.
    """
    if not env.cfg.mocap_velocity_obs_enabled:
        return torch.empty((env.num_envs, 0), device=env.device)
    return env._mocap_lin_vel_b


def next_waypoints_rel_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative positions of the next two waypoints in body frame. Shape [N, 6].

    The lookahead wraps modulo per-env waypoint count, so even short tracks
    produce a valid 6-vector. Output is scaled by ``_WAYPOINT_REL_POSITION_SCALE``.
    """
    n = env.num_envs
    k = 2
    drone_pos_w = env._robot.data.root_pos_w  # [N, 3]
    drone_quat_w = env._robot.data.root_quat_w  # [N, 4]

    counts = env._waypoint_counts.clamp(min=1).unsqueeze(1)  # [N, 1]
    step = torch.arange(k, device=env.device, dtype=torch.long).unsqueeze(0).expand(n, -1)
    wp_idx = (env._current_wp_idx.unsqueeze(1) + step) % counts  # [N, K]

    idx3 = wp_idx.unsqueeze(-1).expand(-1, -1, 3)  # [N, K, 3]
    wp_pos_w = torch.gather(env._waypoint_positions, dim=1, index=idx3)  # [N, K, 3]

    offset_w = wp_pos_w - drone_pos_w.unsqueeze(1)  # [N, K, 3]
    q = drone_quat_w.unsqueeze(1).expand(-1, k, -1)  # [N, K, 4]
    offset_b = quat_apply_inverse(q, offset_w) / _WAYPOINT_REL_POSITION_SCALE

    return offset_b.reshape(n, k * 3)


def collection_time_const(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env constant ``T_c`` (seconds). Shape [N, 1].

    Lets the policy condition on whether the task wants fly-through (T_c≈0)
    versus dwell behaviour.
    """
    return env._collection_time.unsqueeze(-1)


def collection_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dwell-fraction ``timer / T_c`` ∈ [0, 1]. Shape [N, 1].

    Returns 0 when ``T_c == 0`` (the policy receives the constant as a
    separate signal; fraction is undefined here).
    """
    tc = env._collection_time
    progress = torch.where(
        tc > 1e-6,
        (env._collection_timer / tc.clamp(min=1e-6)).clamp(0.0, 1.0),
        torch.zeros_like(tc),
    )
    return progress.unsqueeze(-1)


def last_applied_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last *applied* 4-dim action (after clamp, action noise, LPF and delay). Shape [N, 4].

    Unlike Isaac Lab's ``last_action`` (the raw policy output), this is the command the
    actuator actually received.
    """
    return env._actions_buf

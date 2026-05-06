# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for racing quadcopter.

All gate-related math lives in this module. The env only exposes state
(robot pose, gate arrays, indices); see :class:`RacingQuadcopterRLEnv`.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import *
from isaaclab.utils.math import quat_apply, quat_apply_inverse

# Divide world-frame gate offsets (metres) by this before feeding to the network.
# Keeps inputs in roughly [-1, 1] for gates up to ~10 m away.
_REL_POSITION_SCALE: float = 10.0

def compute_next_gates_rel_and_fwd_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Next-K gates observation: body-frame rel-position + forward + half-width per gate.

    Gate forward is +Y in gate local frame, mapped to world then to body.
    The per-gate half-width is the effective aperture seen by the policy:
    ``half_eff = (gate_size / 2) * (1 - 2 * gate_size_obs_buffer_frac)``.
    Set ``gate_size_obs_buffer_frac=0`` (default) to report the true aperture.
    Fully vectorized over environments and lookahead (no Python loop over K).

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

    # Gate forward direction: local +Y → world → drone body frame.
    y_exp = y_local.view(1, 1, 3).expand(n, k, 3)  # [N, K, 3]
    forward_w = quat_apply(g_ori_w, y_exp)           # [N, K, 3] world frame
    forward_b = quat_apply_inverse(q_drone, forward_w)  # [N, K, 3] body frame

    # Effective half-width per gate (broadcast: per-cell sizes are uniform across K).
    buf_frac = float(env.cfg.observation.gate_size_obs_buffer_frac)
    shrink = max(1.0 - 2.0 * buf_frac, 0.0)
    half_eff = (env._gate_sizes * 0.5 * shrink).unsqueeze(1).expand(n, k).unsqueeze(-1)  # [N, K, 1]

    gate_obs = torch.cat([offset_b_scaled, forward_b, half_eff], dim=-1)  # [N, K, 7]
    gate_obs = gate_obs.reshape(n, k * 7)
    return gate_obs


def velocity_limit_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env velocity limit ``[N, 1]``.

    Exposes the per-env speed cap to the policy so it can adapt its behaviour
    when the cap is randomised across episodes (curriculum).
    """
    return env._velocity_limit.unsqueeze(-1)


def motor_speeds_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Commanded rotor speeds ``[N, 4]`` (rad/s), scaled by ``cfg.motor_omega_max_rad_s``.

    """
    omega_max = float(getattr(env.cfg, "motor_omega_max_rad_s", 1000.0))
    scale = max(omega_max, 1e-6)
    return env._motor_speeds / scale
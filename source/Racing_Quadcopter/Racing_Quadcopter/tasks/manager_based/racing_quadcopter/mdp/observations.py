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
from isaaclab.utils.math import quat_apply, quat_apply_inverse

# Divide world-frame gate offsets (metres) by this before feeding to the network.
# Keeps inputs in roughly [-1, 1] for gates up to ~10 m away.
_REL_POSITION_SCALE: float = 10.0


def _compute_next_gates_rel_and_fwd_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Next-K gates observation: body-frame relative position + forward per gate.

    Gate forward is +Y in gate local frame, mapped to world then to body.
    Fully vectorized over environments and lookahead (no Python loop over K).

    Returns:
        Concatenated ``[N, K*6]`` in direct-env order:
        ``[rel_x, rel_y, rel_z, fwd_x, fwd_y, fwd_z]`` for each gate.
    """
    k = env.cfg.observation.num_next_gates  # lookahead count
    n = env.num_envs
    device = env.device

    drone_pos_w = env._robot.data.root_pos_w   # world frame [N, 3]
    drone_quat_w = env._robot.data.root_quat_w  # wxyz [N, 4]
    y_local = env._y_local  # gate +Y = forward in gate local frame

    # Build [N, K] gate index table: current_gate + {0,1,...,K-1} mod gate_count.
    step = torch.arange(k, device=device, dtype=torch.long).unsqueeze(0).expand(n, -1)  # [N, K]
    gate_idx_raw = env._current_gate_idx.unsqueeze(1) + step  # [N, K]
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

    gate_obs = torch.cat([offset_b_scaled, forward_b], dim=-1)  # [N, K, 6]
    return gate_obs.reshape(n, k * 6)


def velocity_limit_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env velocity limit ``[N, 1]``.

    Exposes the per-env speed cap to the policy so it can adapt its behaviour
    when the cap is randomised across episodes (curriculum).
    """
    return env._velocity_limit.unsqueeze(-1)

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

# Match previous single-vector scaling: world offset (m) -> network input.
_REL_POSITION_SCALE: float = 10.0


def _compute_next_gates_rel_and_fwd_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Next-K gates observation: body-frame relative position + forward per gate.

    Gate forward is +Y in gate local frame, mapped to world then to body.
    Fully vectorized over environments and lookahead (no Python loop over K).

    Returns:
        Concatenated ``[N, K*6]`` in direct-env order:
        ``[rel_x, rel_y, rel_z, fwd_x, fwd_y, fwd_z]`` for each gate.
    """
    k = env.cfg.observation.num_next_gates
    num_gates = env._num_gates
    n = env.num_envs
    device = env.device

    drone_pos_w = env._robot.data.root_pos_w
    drone_quat_w = env._robot.data.root_quat_w
    y_local = env._y_local

    # [N, K] indices of the next K gates per env
    step = torch.arange(k, device=device, dtype=torch.long).unsqueeze(0).expand(n, -1)
    gate_idx = (env._current_gate_idx.unsqueeze(1) + step) % num_gates

    idx3 = gate_idx.unsqueeze(-1).expand(-1, -1, 3)
    idx4 = gate_idx.unsqueeze(-1).expand(-1, -1, 4)
    g_pos_w = torch.gather(env._gate_positions, dim=1, index=idx3)
    g_ori_w = torch.gather(env._gate_orientations, dim=1, index=idx4)

    offset_w = g_pos_w - drone_pos_w.unsqueeze(1)

    q_drone = drone_quat_w.unsqueeze(1).expand(-1, k, -1)
    offset_b = quat_apply_inverse(q_drone, offset_w)
    offset_b_scaled = offset_b / _REL_POSITION_SCALE

    y_exp = y_local.view(1, 1, 3).expand(n, k, 3)
    forward_w = quat_apply(g_ori_w, y_exp)
    forward_b = quat_apply_inverse(q_drone, forward_w)

    gate_obs = torch.cat([offset_b_scaled, forward_b], dim=-1)  # [N, K, 6]
    return gate_obs.reshape(n, k * 6)


def velocity_limit_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env velocity limit ``[N, 1]``."""
    return env._velocity_limit.unsqueeze(-1)


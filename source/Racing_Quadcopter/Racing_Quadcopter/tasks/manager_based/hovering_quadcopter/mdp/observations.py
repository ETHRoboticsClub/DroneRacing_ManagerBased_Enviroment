# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation callables for the hovering / waypoint-following env.

Deployment constraint: mocap provides position + orientation only. Linear
velocity is intentionally absent (state estimation is not in place yet);
gyro (body angular velocity) is available from the on-board IMU.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import *  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


# Scale used to bring world-frame relative positions roughly into [-1, 1]
# (largest cell is ~15 m diagonal; 10 m keeps obs near unit magnitude).
_REL_POSITION_SCALE: float = 10.0


def projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body-frame projection of gravity. Shape [N, 3]."""
    return env.scene[asset_cfg.name].data.projected_gravity_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body-frame angular velocity (gyro). Shape [N, 3]."""
    return env.scene[asset_cfg.name].data.root_ang_vel_b


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body-frame linear velocity. Shape [N, 3].

    Mirrors the old direct-env obs. NOT available from mocap in deployment —
    requires state estimation. Keep it in training; replace / disable at
    deploy time once the SE pipeline is wired up.
    """
    return env.scene[asset_cfg.name].data.root_lin_vel_b


def next_waypoints_rel_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative positions of the next two waypoints in body frame. Shape [N, 6].

    The lookahead wraps modulo per-env waypoint count, so even short tracks
    produce a valid 6-vector. Output is scaled by ``_REL_POSITION_SCALE``.
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
    offset_b = quat_apply_inverse(q, offset_w) / _REL_POSITION_SCALE

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


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last-applied 4-dim action. Shape [N, 4]."""
    return env._actions_buf
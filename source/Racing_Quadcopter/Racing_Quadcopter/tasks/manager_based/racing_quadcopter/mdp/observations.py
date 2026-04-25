# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for racing quadcopter.

All gate-related math lives in this module. The env only exposes state
(robot pose, gate arrays, indices); see :class:`RacingQuadcopterRLEnv`.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

# Lightweight wrappers which add configurable observation corruption (state-estimation noise)
# to the ground-truth observation terms. The configuration lives in
# RacingQuadcopterEnvCfg.observation_noise (defined in the env cfg file).

# Divide world-frame gate offsets (metres) by this before feeding to the network.
# Keeps inputs in roughly [-1, 1] for gates up to ~10 m away.
_REL_POSITION_SCALE: float = 10.0


def compute_next_gates_rel_and_fwd_body(env: ManagerBasedRLEnv) -> torch.Tensor:
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
    gate_obs = gate_obs.reshape(n, k * 6)
    return _apply_obs_noise(env, gate_obs, "obs_gates")


def velocity_limit_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env velocity limit ``[N, 1]``.

    Exposes the per-env speed cap to the policy so it can adapt its behaviour
    when the cap is randomised across episodes (curriculum).
    """
    return _apply_obs_noise(env, env._velocity_limit.unsqueeze(-1), "velocity_limit")


# ---------------------------- Observation corruption ----------------------------
def _apply_obs_noise(env: ManagerBasedRLEnv, tensor: torch.Tensor, term_key: str) -> torch.Tensor:
    """Apply configurable state-estimation corruption to a term tensor.

    Supported corruptions (controlled by ``env.cfg.observation_noise``):
    - per-step additive Gaussian noise (term-specific std)
    - optional per-episode bias (sampled and stored on the env)
    - dropout (zero-out entire term with probability p per-step)

    The function is defensive: if the configuration is not present or disabled
    it returns the tensor unchanged.
    """
    cfg = getattr(env.cfg, "observation_noise", None)
    if cfg is None or not getattr(cfg, "enable", False):
        return tensor

    device = tensor.device

    # Add stored per-episode bias if present on env
    bias_attr_map = {
        "base_lin_vel": "_obs_noise_bias_base_lin_vel",
        "base_ang_vel": "_obs_noise_bias_base_ang_vel",
        "projected_gravity": "_obs_noise_bias_projected_gravity",
        "obs_gates": "_obs_noise_bias_obs_gates",
        "velocity_limit": "_obs_noise_bias_velocity_limit",
        "last_action": "_obs_noise_bias_last_action",
    }
    bias_attr = bias_attr_map.get(term_key)
    if bias_attr and hasattr(env, bias_attr):
        b = getattr(env, bias_attr)
        try:
            tensor = tensor + b
        except Exception:
            # shape mismatch → ignore per-episode bias for safety
            pass

    # Per-step Gaussian noise
    std_name_map = {
        "base_lin_vel": "gaussian_std_base_lin_vel",
        "base_ang_vel": "gaussian_std_base_ang_vel",
        "projected_gravity": "gaussian_std_projected_gravity",
        "obs_gates": "gaussian_std_obs_gates",
        "velocity_limit": "gaussian_std_velocity_limit",
        "last_action": "gaussian_std_last_action",
    }
    std = getattr(cfg, std_name_map.get(term_key, "gaussian_std_default"), getattr(cfg, "gaussian_std_default", 0.0))
    if std and float(std) > 0.0:
        tensor = tensor + torch.randn_like(tensor, device=device) * float(std)

    # Dropout (per-env, per-step)
    dp = getattr(cfg, "dropout_prob", 0.0)
    if dp and float(dp) > 0.0:
        mask_shape = [tensor.shape[0]] + [1] * (tensor.dim() - 1)
        mask = (torch.rand(tensor.shape[0], device=device) > float(dp)).view(*mask_shape)
        tensor = tensor * mask

    return tensor


# ---------------------------- Wrappers for external terms ----------------------------
def _lookup_mdp_term(name: str):
    """Late import helper: get term from isaaclab.envs.mdp or None."""
    try:
        import importlib

        mdp = importlib.import_module("isaaclab.envs.mdp")
        return getattr(mdp, name)
    except Exception:
        return None


def wrap_base_lin_vel(env: ManagerBasedRLEnv, asset_cfg=None) -> torch.Tensor:
    fn = _lookup_mdp_term("base_lin_vel")
    if fn is None:
        raise RuntimeError("isaaclab.envs.mdp.base_lin_vel not available")
    if asset_cfg is None:
        out = fn(env)
    else:
        try:
            out = fn(env, asset_cfg=asset_cfg)
        except TypeError:
            out = fn(env, asset_cfg)
    return _apply_obs_noise(env, out, "base_lin_vel")


def wrap_base_ang_vel(env: ManagerBasedRLEnv, asset_cfg=None) -> torch.Tensor:
    fn = _lookup_mdp_term("base_ang_vel")
    if fn is None:
        raise RuntimeError("isaaclab.envs.mdp.base_ang_vel not available")
    if asset_cfg is None:
        out = fn(env)
    else:
        try:
            out = fn(env, asset_cfg=asset_cfg)
        except TypeError:
            out = fn(env, asset_cfg)
    return _apply_obs_noise(env, out, "base_ang_vel")


def wrap_projected_gravity(env: ManagerBasedRLEnv, asset_cfg=None) -> torch.Tensor:
    fn = _lookup_mdp_term("projected_gravity")
    if fn is None:
        raise RuntimeError("isaaclab.envs.mdp.projected_gravity not available")
    if asset_cfg is None:
        out = fn(env)
    else:
        try:
            out = fn(env, asset_cfg=asset_cfg)
        except TypeError:
            out = fn(env, asset_cfg)
    return _apply_obs_noise(env, out, "projected_gravity")


def wrap_last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    fn = _lookup_mdp_term("last_action")
    if fn is None:
        raise RuntimeError("isaaclab.envs.mdp.last_action not available")
    out = fn(env)
    return _apply_obs_noise(env, out, "last_action")


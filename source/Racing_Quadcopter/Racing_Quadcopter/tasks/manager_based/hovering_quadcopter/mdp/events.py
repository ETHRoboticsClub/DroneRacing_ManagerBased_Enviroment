# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event callables for the hovering / waypoint-following env.

Re-exports the stock IsaacLab event terms (mass / CoM / actuator-gain
randomisation, external force/torque, reset_scene_to_default) and adds the
hovering-specific spawn term.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.events import *  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg


# PhysX inertia layout: diagonal entries ixx, iyy, izz at indices 0, 4, 8.
_INERTIA_DIAG_INDICES = (0, 4, 8)


def randomize_rigid_body_inertia_diagonal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float] = (0.9, 1.1),
) -> None:
    """Scale body inertia diagonal (Ixx, Iyy, Izz) uniformly per axis on startup.

    Each diagonal entry is multiplied by an independent sample in
    ``inertia_distribution_params`` (default ±10 % → 0.9–1.1). Off-diagonal
    terms are left at their default values. Always resets from
    ``asset.data.default_inertia`` so repeated calls do not compound.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    low, high = inertia_distribution_params
    num_envs = env_ids.shape[0]
    num_bodies = body_ids.shape[0]

    inertias = asset.root_physx_view.get_inertias()
    default = asset.data.default_inertia[env_ids[:, None], body_ids].clone()
    inertias[env_ids[:, None], body_ids] = default

    scales = torch.empty(num_envs, num_bodies, 3, device="cpu").uniform_(low, high)
    scale_9 = torch.ones(num_envs, num_bodies, 9, device="cpu")
    for k, idx in enumerate(_INERTIA_DIAG_INDICES):
        scale_9[..., idx] = scales[..., k]
    inertias[env_ids[:, None], body_ids] = default * scale_9
    asset.root_physx_view.set_inertias(inertias, env_ids)


def collect_episode_metrics_before_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> dict:
    """Snapshot episode metrics for resetting envs, then zero the accumulators."""
    if env_ids.numel() == 0:
        return {}
    out = {
        "Metrics/waypoints_collected": env._episode_collected_count[env_ids].float().mean().item(),
        "Metrics/episode_length": env.episode_length_buf[env_ids].float().mean().item(),
    }
    env._episode_collected_count[env_ids] = 0.0
    return out


def reset_waypoint_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Spawn near waypoint 0 of each env's track with small horizontal jitter.

    Resets dwell-state buffers and per-env action history so the next episode
    starts clean.
    """
    if env_ids.numel() == 0:
        return
    n = env_ids.shape[0]

    env._current_wp_idx[env_ids] = 0
    env._holding[env_ids] = False
    env._collection_timer[env_ids] = 0.0
    env._all_collected_buf[env_ids] = False
    env._gates_passed[env_ids] = 0
    env._collected_step_flag[env_ids] = 0.0

    # Spawn pose: small jitter above the first waypoint to avoid all drones
    # stacking on the exact point.
    first_wp = env._waypoint_positions[env_ids, 0]  # [n, 3]
    xy_jitter = float(getattr(env.cfg, "spawn_xy_jitter", 0.3))
    z_low, z_high = getattr(env.cfg, "spawn_z_jitter", (-0.1, 0.1))
    jitter = torch.zeros(n, 3, device=env.device)
    jitter[:, 0].uniform_(-xy_jitter, xy_jitter)
    jitter[:, 1].uniform_(-xy_jitter, xy_jitter)
    jitter[:, 2].uniform_(float(z_low), float(z_high))
    start_pos = first_wp + jitter
    start_pos[:, 2] = start_pos[:, 2].clamp(min=env.cfg.min_height + 0.1)

    # Identity quaternion (level, facing world +X).
    start_quat = torch.zeros(n, 4, device=env.device)
    start_quat[:, 0] = 1.0
    root_pose = torch.cat([start_pos, start_quat], dim=-1)

    root_vel = torch.zeros(n, 6, device=env.device)
    joint_pos = env._robot.data.default_joint_pos[env_ids]
    joint_vel = env._robot.data.default_joint_vel[env_ids]

    env._robot.write_root_pose_to_sim(root_pose, env_ids)
    env._robot.write_root_velocity_to_sim(root_vel, env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    env._actions_buf[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0
    env._prev_prev_actions[env_ids] = 0.0
    env._unclamped_actions_buf[env_ids] = 0.0
    env._prev_distance[env_ids] = env._curr_dist_to_waypoint[env_ids]

    # Per-episode action-noise bias (zeroed unless DR enabled & per_episode_bias).
    action_noise_cfg = getattr(env.cfg, "action_noise", None)
    if (
        action_noise_cfg is not None
        and getattr(action_noise_cfg, "enable", False)
        and getattr(action_noise_cfg, "per_episode_bias", False)
    ):
        a, b = action_noise_cfg.bias_range
        env._action_noise_bias[env_ids] = torch.rand(n, 4, device=env.device) * (b - a) + a
    else:
        env._action_noise_bias[env_ids] = 0.0


def reset_cardinal_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Deterministic cardinal-direction spawn for evaluation.

    Mirrors ``test_controller_offline.py``: drone is placed 1 m from a fixed
    goal at [0, 0, 1] (cell-local) in one of four cardinal directions, with
    identity orientation, zero body rates, and a small upward vz seed.

    The per-env spawn offset (E/W/N/S) and the world-frame goal are pre-
    computed once in ``HoveringQuadcopterRLEnv._setup_cardinal_eval_layout``
    and read here from ``env._eval_spawn_offset`` / ``env._waypoint_positions``.

    Rationale for the upward vz: at exact rest a hover policy reads the state
    as "stationary and needs full active support against gravity" and tends to
    spike thrust on the first tick. A small upward velocity simulates the
    drone already being supported, so the resulting commands reflect what the
    policy would command in mid-flight rather than at the instant of liftoff.
    """
    if env_ids.numel() == 0:
        return
    n = env_ids.shape[0]

    env._current_wp_idx[env_ids] = 0
    env._holding[env_ids] = False
    env._collection_timer[env_ids] = 0.0
    env._all_collected_buf[env_ids] = False
    env._gates_passed[env_ids] = 0
    env._collected_step_flag[env_ids] = 0.0

    goal_pos = env._waypoint_positions[env_ids, 0]            # [n, 3]
    spawn_offset = env._eval_spawn_offset[env_ids]            # [n, 3]
    start_pos = goal_pos + spawn_offset

    # Identity quaternion (w, x, y, z) — body frame == world frame.
    start_quat = torch.zeros(n, 4, device=env.device)
    start_quat[:, 0] = 1.0
    root_pose = torch.cat([start_pos, start_quat], dim=-1)

    # Linear (vx, vy, vz, wx, wy, wz). Upward vz seed = cfg.eval_initial_vz.
    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 2] = float(env.cfg.eval_initial_vz)

    joint_pos = env._robot.data.default_joint_pos[env_ids]
    joint_vel = env._robot.data.default_joint_vel[env_ids]

    env._robot.write_root_pose_to_sim(root_pose, env_ids)
    env._robot.write_root_velocity_to_sim(root_vel, env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    env._actions_buf[env_ids] = 0.0
    env._prev_actions[env_ids] = 0.0
    env._prev_prev_actions[env_ids] = 0.0
    env._unclamped_actions_buf[env_ids] = 0.0
    env._action_noise_bias[env_ids] = 0.0
    env._prev_distance[env_ids] = env._curr_dist_to_waypoint[env_ids]

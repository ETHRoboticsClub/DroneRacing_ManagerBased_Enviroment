# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter rewards (manager-based).

Logic matches ``isaaclab_tasks/direct/racing_quadcopter/reward_manager.GateRewardManager.compute_rewards``:
each public function returns the **unweighted** factor for that term (the value multiplied by the
corresponding entry in ``RewardWeights`` in the direct env). :class:`~isaaclab.managers.RewardManager`
applies the configured ``weight`` (here ``0.01 *`` each direct default weight).

Gate frame helpers, passage detection, and :func:`update_racing_state` live here (shared with the env).

See module-level ``WEIGHT_*`` constants for use in :class:`RewardsCfg`.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from .terminations import refresh_crash_buffers_for_step


def gather_current_gate(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    idx = env._current_gate_idx
    env_arange = torch.arange(env.num_envs, device=env.device)
    pos = env._gate_positions[env_arange, idx]
    ori = env._gate_orientations[env_arange, idx]
    return pos, ori


def gate_forward(env: ManagerBasedRLEnv, gate_ori: torch.Tensor) -> torch.Tensor:
    return quat_apply(gate_ori, env._y_local.expand(gate_ori.shape[0], -1))


def gate_right(env: ManagerBasedRLEnv, gate_ori: torch.Tensor) -> torch.Tensor:
    return quat_apply(gate_ori, env._x_local.expand(gate_ori.shape[0], -1))


def gate_up(env: ManagerBasedRLEnv, gate_ori: torch.Tensor) -> torch.Tensor:
    return quat_apply(gate_ori, env._z_local.expand(gate_ori.shape[0], -1))


def check_gate_passages(env: ManagerBasedRLEnv, drone_pos: torch.Tensor) -> torch.Tensor:
    gate_pos, gate_ori = gather_current_gate(env)
    fwd = gate_forward(env, gate_ori)

    to_prev = env._prev_drone_pos - gate_pos
    to_curr = drone_pos - gate_pos

    dist_prev = (to_prev * fwd).sum(dim=1)
    dist_curr = (to_curr * fwd).sum(dim=1)

    crossed = (dist_prev <= 0) & (dist_curr > 0)

    t = (-dist_prev / (dist_curr - dist_prev + 1e-8)).clamp(0.0, 1.0)
    crossing = env._prev_drone_pos + t.unsqueeze(-1) * (drone_pos - env._prev_drone_pos)
    to_cross = crossing - gate_pos

    right = gate_right(env, gate_ori)
    up = gate_up(env, gate_ori)
    lat = (to_cross * right).sum(dim=1).abs()
    vert = (to_cross * up).sum(dim=1).abs()

    half = env._gate_size / 2.0
    within = (lat < half) & (vert < half)

    passed = crossed & within

    if passed.any():
        env._gates_passed[passed] += 1
        # Advance per-env gate index using the actual per-env gate counts
        counts = env._gate_counts[passed]
        new_idx = env._current_gate_idx[passed] + 1
        # Safe modulo: avoid division by zero for any stray zero-count envs
        mask_nz = counts > 0
        res = torch.zeros_like(new_idx)
        if mask_nz.any():
            res[mask_nz] = new_idx[mask_nz] % counts[mask_nz]
        env._current_gate_idx[passed] = res

    return passed.float()


def update_racing_state(env: ManagerBasedRLEnv) -> None:
    """After physics: gate passages, crash buffers, episode speed stats."""
    drone_pos = env._robot.data.root_pos_w
    drone_vel = env._robot.data.root_lin_vel_w

    gate_pos, gate_ori = gather_current_gate(env)
    to_gate = gate_pos - drone_pos
    g_right = gate_right(env, gate_ori)
    g_up = gate_up(env, gate_ori)
    lateral_right = ((-to_gate) * g_right).sum(dim=1)
    lateral_up = ((-to_gate) * g_up).sum(dim=1)
    lateral_dist = torch.sqrt(lateral_right**2 + lateral_up**2)
    speed = torch.norm(drone_vel, dim=1)

    env._gate_passed_buf = check_gate_passages(env, drone_pos)
    passed_mask = env._gate_passed_buf > 0.5
    if passed_mask.any():
        env._passage_velocity_sum[passed_mask] += speed[passed_mask]
        env._passage_centering_sum[passed_mask] += lateral_dist[passed_mask]
        env._passage_count[passed_mask] += 1.0

    refresh_crash_buffers_for_step(env)

    env._episode_velocity_sum += speed
    env._episode_step_count += 1.0


# ---------------------------------------------------------------------------
# Scaling: manager applies an extra global factor on top of direct weights
# ---------------------------------------------------------------------------

MANAGER_SCALE = 100.0

# Defaults from ``RewardWeights`` in direct ``reward_manager.py`` (reward_scale=1.0).
DIRECT_GATE_PASSAGE = 250.0
DIRECT_GATE_PROGRESS = 20.0
DIRECT_GATE_CENTERING = 0.5
DIRECT_VELOCITY_TOWARD_GATE = 0.1
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


def reward_gate_passage(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._gate_passed_buf


def reward_gate_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    drone_pos = env._robot.data.root_pos_w
    gate_pos, _ = gather_current_gate(env)
    dist_to_gate = torch.norm(gate_pos - drone_pos, dim=1)
    prev_dist = torch.norm(env._prev_drone_pos - gate_pos, dim=1)
    return (prev_dist - dist_to_gate).clamp(-1.0, 1.0)


def reward_gate_centering(env: ManagerBasedRLEnv) -> torch.Tensor:
    drone_pos = env._robot.data.root_pos_w
    gate_pos, gate_ori = gather_current_gate(env)
    gate_size = float(env._gate_size)

    to_gate = gate_pos - drone_pos
    gate_fwd = gate_forward(env, gate_ori)
    g_right = gate_right(env, gate_ori)
    g_up = gate_up(env, gate_ori)

    dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)
    on_approach_side = dist_to_plane <= 0.0

    lateral_right = ((-to_gate) * g_right).sum(dim=1)
    lateral_up = ((-to_gate) * g_up).sum(dim=1)
    lateral_dist = torch.sqrt(lateral_right**2 + lateral_up**2)

    centering_value = torch.exp(-3.0 * lateral_dist / gate_size)
    proximity_factor = torch.exp(-0.5 * dist_to_plane.abs())
    return centering_value * proximity_factor * on_approach_side.float()

def penalty_perception(env: ManagerBasedRLEnv, theta_threshold: float = torch.pi / 3.0) -> torch.Tensor:
    """Penalty when next gate leaves the forward camera cone.

    p_perc = lambda_perc * theta_cam if theta_cam > theta_threshold else 0
    """
    drone_pos = env._robot.data.root_pos_w
    drone_quat = env._robot.data.root_quat_w
    gate_pos, gate_ori = gather_current_gate(env)

    to_gate = gate_pos - drone_pos
    to_gate_norm = to_gate / (torch.norm(to_gate, dim=1, keepdim=True) + 1e-8)

    drone_forward = quat_apply(drone_quat, env._y_local.expand(env.num_envs, -1))
    drone_forward_norm = drone_forward / (torch.norm(drone_forward, dim=1, keepdim=True) + 1e-8)

    gate_fwd = gate_forward(env, gate_ori)
    dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)
    on_approach_side = dist_to_plane <= 0.0

    cos_theta = torch.sum(drone_forward_norm * to_gate_norm, dim=1).clamp(-1.0, 1.0)
    theta_cam = torch.acos(cos_theta)

    penalty = theta_cam * on_approach_side.float()
    return torch.where(theta_cam > theta_threshold, penalty, torch.zeros_like(theta_cam))

def reward_velocity_toward_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
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
    drone_vel = env._robot.data.root_lin_vel_w
    speed = torch.norm(drone_vel, dim=1)
    speed_excess = (speed - env._velocity_limit).clamp(min=0.0)
    return speed_excess**2


def penalty_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    projected_gravity = env._robot.data.projected_gravity_b
    upright = torch.tensor([0.0, 0.0, -1.0], device=projected_gravity.device)
    orientation_error_angle = torch.acos(torch.clamp(torch.norm((projected_gravity * upright), dim=1), -0.98, 0.98))
    return torch.square(orientation_error_angle)


def penalty_angular_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    drone_angvel = env._robot.data.root_ang_vel_b
    return drone_angvel.square().sum(dim=1)


def penalty_action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    action_diff = env._actions_buf - env._prev_actions
    return action_diff.square().sum(dim=1)


def penalty_action_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._actions_buf.square().sum(dim=1)


def penalty_crash(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._crashed_buf.float()


def penalty_wrong_side(env: ManagerBasedRLEnv) -> torch.Tensor:
    drone_pos = env._robot.data.root_pos_w
    gate_pos, gate_ori = gather_current_gate(env)
    to_gate = gate_pos - drone_pos
    gate_fwd = gate_forward(env, gate_ori)
    dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)
    on_approach_side = dist_to_plane <= 0.0
    return (~on_approach_side).float() * (1.0 - env._gate_passed_buf)

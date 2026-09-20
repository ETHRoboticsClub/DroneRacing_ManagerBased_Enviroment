# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter: episode metrics at reset and racing spawn (event manager reset mode)."""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.events import *
from isaaclab.utils.math import quat_apply

from Racing_Quadcopter.tasks.manager_based.route_progress import snapshot_episode_route_progress

from .stats import accumulate_episode_stats


def _emit_episode_metrics(env: ManagerBasedRLEnv, ids: torch.Tensor, prefix: str, gate_log: dict) -> None:
    """Write gates_passed / mean_velocity_kmh / lap_completion_pct / mean_lap_time_s to gate_log.

    Logging contract: ``gates_passed`` and ``mean_velocity_kmh`` are always
    emitted; ``lap_completion_pct`` is emitted whenever the subset has at least
    one countable run (a finished lap or a terminated partial run);
    ``mean_lap_time_s`` is emitted only when at least one env completed a full lap.

    ``mean_velocity_kmh`` is the per-step world-frame speed averaged over each
    env's episode (in m/s), then averaged across envs in the subset, then
    converted to km/h (×3.6) to match the console "Speed" stat.

    ``lap_completion_pct`` mirrors the console "Lap%" formula: each completed lap
    contributes 1.0, each terminated (not timed-out) episode contributes a
    partial-run value of ``(gates_passed % gate_count) / gate_count``, and the
    metric is the mean over those run entries.
    """
    if ids.numel() == 0:
        return

    gate_log[f"{prefix}/gates_passed"] = env._gates_passed[ids].float().mean().item()

    step_counts = env._episode_step_count[ids]
    valid_steps = step_counts > 0
    if valid_steps.any():
        mean_vel_ms = (
            env._episode_velocity_sum[ids][valid_steps] / step_counts[valid_steps]
        ).mean().item()
        gate_log[f"{prefix}/mean_velocity_kmh"] = mean_vel_ms * 3.6
    else:
        gate_log[f"{prefix}/mean_velocity_kmh"] = 0.0

    lap_counts = env._episode_lap_count[ids].long()
    gates_passed = env._gates_passed[ids].long()
    gate_counts = env._gate_counts[ids].long()
    reset_terminated = getattr(env, "reset_terminated", None)
    if reset_terminated is not None:
        terminated = reset_terminated[ids].bool()
    else:
        terminated = torch.zeros_like(lap_counts, dtype=torch.bool)

    has_laps = lap_counts > 0
    if has_laps.any():
        mean_lap_time = (
            env._episode_lap_time_sum[ids][has_laps] / lap_counts[has_laps]
        ).mean().item()
        gate_log[f"{prefix}/mean_lap_time_s"] = mean_lap_time

    # Lap-completion percentage, matching the sliding-window console "Lap%".
    partial_mask = terminated & (gate_counts > 0)
    runs_count = int(lap_counts.sum().item()) + int(partial_mask.sum().item())
    if runs_count > 0:
        completed_value = float(lap_counts.sum().item())
        if partial_mask.any():
            gp_p = gates_passed[partial_mask].float()
            gc_p = gate_counts[partial_mask].float()
            partial_value = float(((gp_p % gc_p) / gc_p).sum().item())
        else:
            partial_value = 0.0
        gate_log[f"{prefix}/lap_completion_pct"] = (completed_value + partial_value) / runs_count * 100.0


def _sample_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(low, high)


def randomize_rigid_body_inertia(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg,
    inertia_distribution_params: tuple[float, float] = (0.8, 1.2),
    operation: str = "scale",
    distribution: str = "uniform",
) -> None:
    """Randomize rigid-body inertias if the Isaac/PhysX runtime exposes inertia tensors.

    Isaac Lab provides a stock mass randomizer, but not every installed version exposes a
    matching inertia event. This helper mirrors the mass-randomization pattern while staying
    defensive: if the current runtime lacks ``get_inertias``/``set_inertias``, the event
    prints one warning and otherwise leaves the simulation unchanged.
    """
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if distribution != "uniform":
        raise ValueError(f"Unsupported inertia distribution {distribution!r}; only 'uniform' is implemented.")
    if operation != "scale":
        raise ValueError(f"Unsupported inertia operation {operation!r}; only 'scale' is implemented.")

    env_ids_cpu = env_ids.to(device="cpu")

    asset = env.scene[asset_cfg.name]
    physx_view = asset.root_physx_view
    if not hasattr(physx_view, "get_inertias") or not hasattr(physx_view, "set_inertias"):
        if not getattr(env, "_warned_missing_inertia_randomization", False):
            print("[WARN] Inertia randomization requested, but this Isaac runtime has no PhysX inertia API.")
            env._warned_missing_inertia_randomization = True
        return

    cache_name = f"_default_inertias_{asset_cfg.name}"
    default_inertias = getattr(env, cache_name, None)
    if default_inertias is None:
        default_inertias = physx_view.get_inertias().clone()
        setattr(env, cache_name, default_inertias)

    inertias = default_inertias.clone()
    selected = inertias[env_ids_cpu]
    inertia_device = selected.device
    body_ids = getattr(asset_cfg, "body_ids", slice(None))
    if body_ids is None:
        body_ids = slice(None)
    if isinstance(body_ids, int):
        body_ids = [body_ids]

    body_selected = selected[:, body_ids]
    low, high = inertia_distribution_params
    multiplier_shape = body_selected.shape[:2] + (1,) * max(body_selected.ndim - 2, 0)
    multipliers = _sample_uniform(multiplier_shape, float(low), float(high), inertia_device).to(
        body_selected.dtype
    )
    selected[:, body_ids] = body_selected * multipliers

    try:
        physx_view.set_inertias(selected, indices=env_ids_cpu)
    except (TypeError, RuntimeError):
        inertias[env_ids_cpu] = selected
        try:
            physx_view.set_inertias(inertias, indices=env_ids_cpu)
        except TypeError:
            physx_view.set_inertias(inertias, env_ids_cpu)


def collect_episode_metrics_before_reset(env: ManagerBasedRLEnv, env_ids_tensor: torch.Tensor) -> dict:
    """Snapshot episode stats into _last_episode_* and zero accumulators for the next episode.

    Must be called BEFORE super()._reset_idx so the accumulators still hold the
    completed episode's data. After super()._reset_idx the EventManager will call
    reset_racing_spawn which zeros _current_gate_idx and _gates_passed.

    When the env exposes ``_env_col_idx`` (skill-evaluation env), per-terrain
    metrics are emitted under ``terrain_{i}/*`` in addition to the global
    ``Episode_Metrics/*`` keys. Each terrain gets its own top-level
    tensorboard panel; the global aggregate stays under ``Episode_Metrics/``.
    """
    gate_log: dict = {}
    if env_ids_tensor.numel() == 0:
        return gate_log

    # This task-independent score is additional to all existing lap/gate
    # statistics. Curriculum controllers consume it on the ensuing reset.
    gate_log.update(snapshot_episode_route_progress(env, env_ids_tensor))

    # Restrict all stats to drones that spawned at the offset origin (when configured). The
    # curriculum snapshots and buffer zeroing further below still apply to ALL resetting envs.
    if bool(getattr(env.cfg, "stats_origin_only", False)) and hasattr(env, "_spawned_at_origin"):
        stats_ids = env_ids_tensor[env._spawned_at_origin[env_ids_tensor]]
    else:
        stats_ids = env_ids_tensor

    # Overall (run-wide) peak speed over any completed run; folded origin-only in gate_utils.
    gate_log["Episode_Metrics/top_speed"] = env._top_speed.item()

    if stats_ids.numel() > 0:
        gate_log["Episode_Metrics/gates_passed"] = env._gates_passed[stats_ids].float().mean().item()

        step_counts = env._episode_step_count[stats_ids]
        valid_steps = step_counts > 0
        if valid_steps.any():
            gate_log["Episode_Metrics/mean_velocity"] = (
                env._episode_velocity_sum[stats_ids][valid_steps] / step_counts[valid_steps]
            ).mean().item()
        else:
            gate_log["Episode_Metrics/mean_velocity"] = 0.0

        has_passages_stats = env._passage_count[stats_ids] > 0
        if has_passages_stats.any():
            gate_log["Episode_Metrics/mean_passage_center_offset"] = (
                env._passage_centering_sum[stats_ids][has_passages_stats]
                / env._passage_count[stats_ids][has_passages_stats]
            ).mean().item()
            gate_log["Episode_Metrics/mean_passage_velocity"] = (
                env._passage_velocity_sum[stats_ids][has_passages_stats]
                / env._passage_count[stats_ids][has_passages_stats]
            ).mean().item()
        else:
            gate_log["Episode_Metrics/mean_passage_center_offset"] = 0.0
            gate_log["Episode_Metrics/mean_passage_velocity"] = 0.0

        gate_log["Episode_Metrics/collision_count"] = env._episode_collision_count[stats_ids].mean().item()

        lap_counts = env._episode_lap_count[stats_ids]
        has_laps = lap_counts > 0
        if has_laps.any():
            gate_log["Episode_Metrics/mean_lap_time_s"] = (
                env._episode_lap_time_sum[stats_ids][has_laps] / lap_counts[has_laps]
            ).mean().item()
            env._total_lap_time_sum += float(env._episode_lap_time_sum[stats_ids][has_laps].sum().item())
            env._total_lap_count += int(lap_counts[has_laps].sum().item())

        step_counts_all = env._episode_step_count[stats_ids]
        episode_mean_speed = torch.where(
            step_counts_all > 0,
            env._episode_velocity_sum[stats_ids] / step_counts_all.clamp(min=1),
            torch.zeros_like(step_counts_all),
        )
        accumulate_episode_stats(
            env,
            stats_ids,
            env._gates_passed[stats_ids],
            lap_counts,
            env._gate_counts[stats_ids],
            episode_mean_speed,
        )

    # Per-terrain breakdown when running on the skill-evaluation env. Deliberately
    # keyed off all resetting envs rather than stats_ids: each terrain column is its
    # own population, so origin-only filtering does not apply here.
    env_col_idx = getattr(env, "_env_col_idx", None)
    if env_col_idx is not None:
        cols = env_col_idx[env_ids_tensor]
        for col in cols.unique().tolist():
            ids = env_ids_tensor[cols == col]
            _emit_episode_metrics(env, ids, f"terrain_{int(col)}", gate_log)

    # Save passage-quality averages for the curriculum check in the NEXT episode.
    # _last_episode_* persist across resets; the live accumulators are zeroed below.
    env._last_episode_gates_passed[env_ids_tensor] = env._gates_passed[env_ids_tensor].float()
    has_passages = env._passage_count[env_ids_tensor] > 0
    env._last_episode_mean_velocity[env_ids_tensor] = torch.where(
        has_passages,
        env._passage_velocity_sum[env_ids_tensor] / env._passage_count[env_ids_tensor],
        torch.zeros_like(env._passage_velocity_sum[env_ids_tensor]),
    )
    env._last_episode_center_offset[env_ids_tensor] = torch.where(
        has_passages,
        env._passage_centering_sum[env_ids_tensor] / env._passage_count[env_ids_tensor],
        torch.zeros_like(env._passage_centering_sum[env_ids_tensor]),
    )

    # Zero all live accumulators so the next episode starts clean.
    env._episode_velocity_sum[env_ids_tensor] = 0.0
    env._episode_step_count[env_ids_tensor] = 0.0
    env._episode_collision_count[env_ids_tensor] = 0.0
    env._passage_velocity_sum[env_ids_tensor] = 0.0
    env._passage_centering_sum[env_ids_tensor] = 0.0
    env._passage_count[env_ids_tensor] = 0.0
    env._episode_lap_time_sum[env_ids_tensor] = 0.0
    env._episode_lap_count[env_ids_tensor] = 0.0
    env._last_lap_step[env_ids_tensor] = 0.0
    env._run_max_speed[env_ids_tensor] = 0.0  # discard peak speed of the (now-ended) run

    return gate_log


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion from XYZ Euler angles in wxyz order."""
    hr = 0.5 * roll
    hp = 0.5 * pitch
    hy = 0.5 * yaw
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.stack([qw, qx, qy, qz], dim=-1)


def _write_spawn_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    start_pos: torch.Tensor,
    start_quat: torch.Tensor,
    root_vel: torch.Tensor,
    init_actions: torch.Tensor | None = None,
) -> None:
    """Write root pose, velocity, and joint state to simulation and reset racing buffers."""
    root_poses = torch.zeros(len(env_ids), 7, device=env.device)
    root_poses[:, :3] = start_pos
    root_poses[:, 3:7] = start_quat

    joint_pos = env._robot.data.default_joint_pos[env_ids]
    joint_vel = env._robot.data.default_joint_vel[env_ids]
    env._robot.write_root_pose_to_sim(root_poses, env_ids)
    env._robot.write_root_velocity_to_sim(root_vel, env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    env._route_episode_start_position[env_ids] = start_pos

    # Sync _prev_drone_pos to the new spawn position so the first step's gate-progress
    # reward computes a valid (zero) distance reduction rather than a garbage delta.
    env._prev_drone_pos[env_ids] = start_pos
    if init_actions is None:
        env._actions_buf[env_ids] = 0.0
        env._prev_actions[env_ids] = 0.0
    else:
        # Randomised initial actions prevent the action-smoothness penalty from
        # producing a large spike on the very first step after reset.
        env._actions_buf[env_ids] = init_actions
        env._prev_actions[env_ids] = init_actions

    if hasattr(env, "_motor_speeds"):
        env._motor_speeds[env_ids] = 0.0
    if hasattr(env, "_prev_motor_speeds"):
        env._prev_motor_speeds[env_ids] = 0.0


def _sample_spawn_gate(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a gate index per env and return (spawn_idx, gate_pos, gate_ori).

    The distribution over gates is controlled by ``cfg.spawn_gate_bias`` (a power-law
    exponent on each gate's rank from the end of the track)::

        weight(gate i) ∝ (gate_count - i) ** spawn_gate_bias

    - ``bias = 0`` → uniform over [0, gate_count-1] (default; original behaviour).
    - ``bias = 1`` → linear, favouring earlier gates; the first:last spawn ratio is
      ``gate_count : 1`` (e.g. 6:1 on a 6-gate track → gate 0 ≈ 28.6 %, last ≈ 4.8 %).
    - higher values steepen the bias toward the first gate.

    Earlier gates are favoured because a drone spawned at gate i also flies the
    downstream gates once competent, so late gates accrue pass-through coverage;
    the non-zero tail still gives the final segments direct spawns early in training.

    Each env draws independently so multi-difficulty grids (different gate counts
    per row) are handled correctly.
    """
    n = len(env_ids)
    gate_counts = env._gate_counts[env_ids].float()  # [N]
    bias = float(getattr(env.cfg, "spawn_gate_bias", 0.0))

    if bias == 0.0:
        spawn_idx = (torch.rand(n, device=env.device) * gate_counts).long()  # [N] in [0, count-1]
    else:
        max_count = int(env._gate_counts[env_ids].max().item())
        idx = torch.arange(max_count, device=env.device).unsqueeze(0)  # [1, max_count]
        rank_from_end = gate_counts.unsqueeze(1) - idx                  # [N, max_count]; >0 for valid gates
        weights = torch.where(
            rank_from_end > 0, rank_from_end.pow(bias), torch.zeros_like(rank_from_end)
        )
        spawn_idx = torch.multinomial(weights, num_samples=1).squeeze(1)  # [N] in [0, count-1]

    idx_pos = spawn_idx.view(n, 1, 1).expand(n, 1, 3)
    gate_pos = env._gate_positions[env_ids].gather(1, idx_pos).squeeze(1)  # [N, 3]
    idx_ori = spawn_idx.view(n, 1, 1).expand(n, 1, 4)
    gate_ori = env._gate_orientations[env_ids].gather(1, idx_ori).squeeze(1)  # [N, 4]
    return spawn_idx, gate_pos, gate_ori


def _apply_fixed_initialization(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Spawn all drones at a deterministic 8-lane starting grid behind gate 0.

    Envs that share the same terrain cell get unique positions so they don't
    stack on top of each other. Lanes are 0.6 m wide; rows are 0.6 m deep,
    starting 2 m behind the gate.

    Race-scenario hook: when ``env.cfg.opponent_spawn_gate_idx`` is set (an int
    in [0, gate_count-1]), env_id == 1 (the "opponent") spawns at that gate
    index instead of gate 0 — giving it a head start along the track. env_0
    remains the main drone, always starting at gate 0.
    """
    n = len(env_ids)
    spawn_idx = torch.zeros(n, dtype=torch.long, device=env.device)

    opp_offset = getattr(env.cfg, "opponent_spawn_gate_idx", None)
    if opp_offset is not None:
        opp_mask = env_ids == 1
        if opp_mask.any():
            opp_local_idx = opp_mask.nonzero(as_tuple=False).squeeze(-1)
            gc_opp = env._gate_counts[env_ids[opp_local_idx]].clamp(min=1)
            spawn_idx[opp_local_idx] = torch.minimum(
                torch.full_like(gc_opp, int(opp_offset)),
                gc_opp - 1,
            )

    idx_pos = spawn_idx.view(n, 1, 1).expand(n, 1, 3)
    gate_pos = env._gate_positions[env_ids].gather(1, idx_pos).squeeze(1)  # [N, 3]
    idx_ori = spawn_idx.view(n, 1, 1).expand(n, 1, 4)
    gate_ori = env._gate_orientations[env_ids].gather(1, idx_ori).squeeze(1)  # [N, 4]

    env._spawn_gate_idx[env_ids] = spawn_idx
    env._current_gate_idx[env_ids] = spawn_idx
    env._gates_passed[env_ids] = 0

    fwd   = quat_apply(gate_ori, env._y_local.expand(n, -1))
    right = quat_apply(gate_ori, env._x_local.expand(n, -1))

    n_lanes = 8
    rank    = torch.arange(n, device=env.device)
    lane    = (rank % n_lanes).float() - (n_lanes - 1) / 2.0  # -3.5 … +3.5
    row     = (rank // n_lanes).float()

    start_pos = (
        gate_pos
        - (2.0 + row * 0.6).unsqueeze(-1) * fwd
        + (lane * 0.6).unsqueeze(-1) * right
    )
    start_pos[:, 2] = gate_pos[:, 2]
    root_vel = torch.zeros(n, 6, device=env.device)

    # Deterministic eval pose: aim the camera optical axis at gate 0 (no random noise).
    # Same geometry as the uniform spawn — see _apply_uniform_initialization for the
    # derivation. With lane lateral offsets, yaw is no longer just along the gate's
    # forward axis; computing it from the actual spawn→gate vector keeps every lane on-axis.
    CAM_TILT = math.radians(30.0)
    to_gate = gate_pos - start_pos
    d_horiz = torch.linalg.norm(to_gate[:, :2], dim=1).clamp(min=1e-3)
    yaw   = torch.atan2(to_gate[:, 1], to_gate[:, 0])
    pitch = (CAM_TILT - torch.atan2(to_gate[:, 2], d_horiz)).clamp(-math.pi / 3.0, math.pi / 3.0)
    roll  = torch.zeros(n, device=env.device)
    start_quat = _quat_from_euler_xyz(roll, pitch, yaw)

    _write_spawn_state(env, env_ids, start_pos, start_quat, root_vel)


def _apply_ground_initialization(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Start 2 m behind a randomly chosen gate, aligned with gate orientation."""
    n = len(env_ids)
    spawn_idx, gate_pos, gate_ori = _sample_spawn_gate(env, env_ids)
    env._spawn_gate_idx[env_ids] = spawn_idx
    env._current_gate_idx[env_ids] = spawn_idx
    env._gates_passed[env_ids] = 0
    fwd = quat_apply(gate_ori, env._y_local.expand(n, -1))

    start_pos = gate_pos - 2.0 * fwd  # 2 m behind gate along gate forward axis
    start_pos[:, 2] = gate_pos[:, 2]  # same height as gate centre
    start_quat = gate_ori              # drone faces the gate directly
    root_vel = torch.zeros(n, 6, device=env.device)

    _write_spawn_state(env, env_ids, start_pos, start_quat, root_vel)


def _apply_uniform_initialization(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Uniform spawn in a small box on the approach side of a randomly chosen gate.

    Box is defined in the gate's own frame so it is always before the gate
    regardless of world orientation:
      - depth  [1.0, 4.0] m behind the gate (negative forward axis)
      - right  ±1.5 m along gate right axis
      - up     ±0.75 m around gate center height
    """
    n = len(env_ids)
    spawn_idx, gate_pos, gate_ori = _sample_spawn_gate(env, env_ids)
    env._spawn_gate_idx[env_ids] = spawn_idx
    env._current_gate_idx[env_ids] = spawn_idx
    env._gates_passed[env_ids] = 0

    # World-frame gate axes for placing the spawn box in gate-local coordinates.
    fwd   = quat_apply(gate_ori, env._y_local.expand(n, -1))  # passage direction
    right = quat_apply(gate_ori, env._x_local.expand(n, -1))
    up    = quat_apply(gate_ori, env._z_local.expand(n, -1))

    depth   = 1.0 + torch.rand(n, device=env.device) * 3.0          # [1, 4] m before gate
    lateral = 1.5 * (2.0 * torch.rand(n, device=env.device) - 1.0)  # ±1.5 m right/left
    vert    = 0.75 * (2.0 * torch.rand(n, device=env.device) - 1.0) # ±0.75 m up/down

    start_pos = (
        gate_pos
        - depth.unsqueeze(-1)   * fwd    # behind gate (approach side)
        + lateral.unsqueeze(-1) * right
        + vert.unsqueeze(-1)    * up
    )
    # Never go below ground. Clamp above min_height (0.2 m) to avoid instant termination.
    start_pos[:, 2] = start_pos[:, 2].clamp(min=0.3)

    # Aim the camera optical axis at the next gate, then sprinkle ±20° noise on each Euler angle.
    # Geometry: camera_keypoint is rotated -30° about Y wrt base_link (URDF), so the
    # camera's body-frame forward is R_y(-30°)·[1,0,0] = (cos30°, 0, sin30°) — 30° above
    # body +X. Drone Euler is R_world_body = R_z(yaw) R_y(pitch) R_x(roll). With roll=0,
    # the camera-world direction in the (x,z) pitch-plane lies at elevation (CAM_TILT − pitch).
    # To make that match the elevation-to-gate, set:    pitch = CAM_TILT − elevation_to_gate.
    # Yaw is set so the body-forward horizontal projection points at the gate.
    CAM_TILT = math.radians(30.0)
    to_gate = gate_pos - start_pos                                        # [N, 3]
    d_horiz = torch.linalg.norm(to_gate[:, :2], dim=1).clamp(min=1e-3)    # [N]
    yaw_to_gate       = torch.atan2(to_gate[:, 1], to_gate[:, 0])         # [N]
    elevation_to_gate = torch.atan2(to_gate[:, 2], d_horiz)               # [N]
    desired_pitch     = (CAM_TILT - elevation_to_gate).clamp(-math.pi / 3.0, math.pi / 3.0)

    # Wider noise here trains robustness to mid-flight misalignment encountered later in
    # episodes — too tight a cone caused the policy to overfit to near-aligned starts.
    yaw_half   = math.pi / 3.0   # ±60° around gate direction
    pitch_half = math.pi / 4.0   # ±45° around camera-aimed pitch
    roll_half  = math.pi / 4.0   # ±45° roll
    roll  = (2.0 * torch.rand(n, device=env.device) - 1.0) * roll_half
    pitch = desired_pitch + (2.0 * torch.rand(n, device=env.device) - 1.0) * pitch_half
    yaw   = yaw_to_gate   + (2.0 * torch.rand(n, device=env.device) - 1.0) * yaw_half
    start_quat = _quat_from_euler_xyz(roll, pitch, yaw)

    # Small random initial velocity and angular rate to improve robustness to
    # mid-flight perturbations during training.
    lin_vel = -0.5 + torch.rand(n, 3, device=env.device) * 1.0   # ±0.5 m/s per axis
    ang_vel = -0.1 + torch.rand(n, 3, device=env.device) * 0.2   # ±0.1 rad/s per axis
    root_vel = torch.cat([lin_vel, ang_vel], dim=-1)

    init_actions = -1.0 + torch.rand(n, 4, device=env.device) * 2.0  # uniform in [-1, 1]
    _write_spawn_state(env, env_ids, start_pos, start_quat, root_vel, init_actions=init_actions)


def _apply_recovery_override(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Overwrite orientation and velocity with fully random values for recovery training.

    Position is kept from the preceding normal spawn (read from _prev_drone_pos).
    Orientation is a uniformly random unit quaternion (any attitude, including inverted).
    Linear velocity is 2–6 m/s in a random direction; angular velocity is ±3 rad/s.
    """
    n = len(env_ids)

    # Position from the normal spawn (already written via _write_spawn_state).
    start_pos = env._prev_drone_pos[env_ids].clone()

    # Uniformly random unit quaternion via Gaussian normalisation.
    quat = torch.randn(n, 4, device=env.device)
    quat = quat / quat.norm(dim=-1, keepdim=True)

    # Random linear velocity: random direction, speed in [2, 6] m/s.
    speed     = 2.0 + torch.rand(n, device=env.device) * 4.0
    direction = torch.randn(n, 3, device=env.device)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    lin_vel   = direction * speed.unsqueeze(-1)

    # Random angular velocity up to ±3 rad/s per axis.
    ang_vel  = (torch.rand(n, 3, device=env.device) * 2.0 - 1.0) * 3.0
    root_vel = torch.cat([lin_vel, ang_vel], dim=-1)

    init_actions = -1.0 + torch.rand(n, 4, device=env.device) * 2.0
    _write_spawn_state(env, env_ids, start_pos, quat, root_vel, init_actions=init_actions)


def reset_racing_spawn(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None, method: str = "mixed"
) -> None:
    """Reset gate progress and sample initialization method.

    Args:
        method: One of ``"ground"``, ``"uniform"``, or ``"mixed"``.
    """
    n = len(env_ids)
    if n == 0:
        return

    if getattr(env.cfg, "fixed_spawn", False):
        _apply_fixed_initialization(env, env_ids)
    elif method == "ground":
        _apply_ground_initialization(env, env_ids)
    elif method == "uniform":
        _apply_uniform_initialization(env, env_ids)
    elif method == "mixed":
        # Stochastically choose per-env whether to use ground or uniform spawn.
        # ground_init_prob=0.5 gives equal chance; tune to bias toward harder starts.
        ground_prob = float(getattr(env.cfg, "ground_init_prob", 0.5))
        use_ground = torch.rand(n, device=env.device) < ground_prob
        ground_ids = env_ids[use_ground]
        uniform_ids = env_ids[~use_ground]
        if len(ground_ids) > 0:
            _apply_ground_initialization(env, ground_ids)
        if len(uniform_ids) > 0:
            _apply_uniform_initialization(env, uniform_ids)
    else:
        raise ValueError(f"Unknown reset method '{method}'. Use 'ground', 'uniform', or 'mixed'.")

    # Recovery spawns active only in curriculum stage 3+.
    recovery_prob = float(getattr(env.cfg, "recovery_spawn_prob", 0.0))
    if recovery_prob > 0.0 and getattr(env, "_curriculum_stage", 0) >= 2 and not getattr(env.cfg, "eval_mode", False):
        mask = torch.rand(n, device=env.device) < recovery_prob
        if mask.any():
            _apply_recovery_override(env, env_ids[mask])


def _apply_upside_down_override(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Re-write drone orientation to 180° around X (upside-down), keeping spawn position.

    Called after a standard spawn has already set ``_prev_drone_pos``. Reads the
    position from there, writes back with a flipped quaternion.  Gate indices and
    the action/motor buffers are reset by ``_write_spawn_state`` as a side-effect.

    wxyz (0, 1, 0, 0) encodes a 180° rotation around the world X axis, which
    flips the drone so its +Z (body up) points toward -Z (world down).
    """
    n = len(env_ids)
    pos = env._prev_drone_pos[env_ids].clone()
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 1] = 1.0  # wxyz (0, 1, 0, 0) = 180° around X = upside-down
    _write_spawn_state(env, env_ids, pos, quat, torch.zeros(n, 6, device=env.device))


def reset_skill_eval_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Per-terrain spawn dispatcher for skill evaluation terrains.

    Iterates over ``env.cfg.skill_terrains`` (one entry per terrain) and
    applies each terrain's spawn logic to the subset of ``env_ids`` assigned to
    that terrain (determined via ``env._env_col_idx``, stored by
    :class:`RacingQuadcopterSkillEvalRLEnv`).

    Currently supported per-terrain flags (extend :class:`SkillTerrainCfg` and
    add branches here to expose new spawn behaviours):
        spawn_upside_down : spawn behind gate 0, then flip orientation 180° around X.
        spawn_recovery    : spawn behind gate 0, then apply random attitude + velocity
                            (delegates to ``_apply_recovery_override``).
    """
    if env_ids is None or len(env_ids) == 0:
        return

    skill_terrains = env.cfg.skill_terrains
    for col_idx, terrain_cfg in enumerate(skill_terrains):
        mask = env._env_col_idx[env_ids] == col_idx
        col_ids = env_ids[mask]
        if len(col_ids) == 0:
            continue

        # Base spawn: deterministic grid behind gate 0 (same as eval/fixed_spawn mode).
        _apply_fixed_initialization(env, col_ids)

        # Per-terrain overrides applied on top of the base spawn.
        if getattr(terrain_cfg, "spawn_upside_down", False):
            _apply_upside_down_override(env, col_ids)
        elif getattr(terrain_cfg, "spawn_recovery", False):
            _apply_recovery_override(env, col_ids)

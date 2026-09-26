# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate frame helpers, plane-crossing passage detection, and the per-step racing state update.

Functions here are called directly from the environment's step loop (``update_racing_state``)
and from reward/observation terms that need gate geometry.  They are separate from
``rewards.py`` so reward functions remain pure read-only computations, and the single
state-update entry-point has a dedicated home.

Gate local coordinate system
-----------------------------
  +X  right   (along gate bar width)
  +Y  forward (passage direction — drone should fly through in +Y)
  +Z  up      (gates are always upright, world Z-up)

Quaternions are in (w, x, y, z) order (IsaacLab convention).
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from .curriculums import check_collision_curriculum_enabled


# ---------------------------------------------------------------------------
# Gate frame helpers
# ---------------------------------------------------------------------------

def gather_current_gate(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame position [N,3] and orientation [N,4] of each env's target gate."""
    idx = env._current_gate_idx  # [N] long
    env_arange = torch.arange(env.num_envs, device=env.device)
    pos = env._gate_positions[env_arange, idx]     # [N, 3] world frame
    ori = env._gate_orientations[env_arange, idx]  # [N, 4] wxyz
    return pos, ori


def gate_forward(env: ManagerBasedRLEnv, gate_ori: torch.Tensor) -> torch.Tensor:
    """Gate +Y axis in world frame (passage direction). Shape [N, 3]."""
    return quat_apply(gate_ori, env._y_local.expand(gate_ori.shape[0], -1))


def gate_right(env: ManagerBasedRLEnv, gate_ori: torch.Tensor) -> torch.Tensor:
    """Gate +X axis in world frame (lateral). Shape [N, 3]."""
    return quat_apply(gate_ori, env._x_local.expand(gate_ori.shape[0], -1))


def gate_up(env: ManagerBasedRLEnv, gate_ori: torch.Tensor) -> torch.Tensor:
    """Gate +Z axis in world frame (vertical). Shape [N, 3]."""
    return quat_apply(gate_ori, env._z_local.expand(gate_ori.shape[0], -1))


# ---------------------------------------------------------------------------
# Gate passage detection
# ---------------------------------------------------------------------------

def check_gate_passages(env: ManagerBasedRLEnv, drone_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect line-crossing through the current gate plane and update gate indices.

    Uses the signed distance from the gate plane: a passage is registered when the
    drone transitions from the approach side (dist <= 0) to the exit side (dist > 0)
    AND the interpolated crossing point lies within the gate aperture (square test).

    Also detects wrong-direction crossings: drone enters from the exit side (dist > 0)
    through the aperture — the gate was crossed backwards.

    Returns:
        passed_float: Float [N] tensor: 1.0 where a valid passage occurred this step.
        wrong_cross:  Bool  [N] tensor: True where a wrong-direction crossing occurred.
    """
    gate_pos = env._curr_gate_pos
    fwd = env._curr_gate_forward
    right = env._curr_gate_right
    up = env._curr_gate_up
    half = env._gate_sizes / 2.0  # [N]

    # Signed distance along gate forward axis: negative = approach side, positive = exit.
    to_prev = env._prev_drone_pos - gate_pos  # [N, 3]
    to_curr = drone_pos - gate_pos             # [N, 3]
    dist_prev = (to_prev * fwd).sum(dim=1)  # [N]
    dist_curr = (to_curr * fwd).sum(dim=1)  # [N]

    def _aperture_check(t: torch.Tensor) -> torch.Tensor:
        crossing = env._prev_drone_pos + t.unsqueeze(-1) * (drone_pos - env._prev_drone_pos)
        to_cross = crossing - gate_pos
        lat  = (to_cross * right).sum(dim=1).abs()
        vert = (to_cross * up).sum(dim=1).abs()
        return (lat < half) & (vert < half)

    # --- Correct-direction crossing: approach side → exit side ---
    correct_crossed = (dist_prev <= 0) & (dist_curr > 0)
    t_correct = (-dist_prev / (dist_curr - dist_prev + 1e-8)).clamp(0.0, 1.0)
    passed = correct_crossed & _aperture_check(t_correct)

    # --- Wrong-direction crossing: exit side → approach side (backwards through aperture) ---
    wrong_crossed = (dist_prev > 0) & (dist_curr <= 0)
    t_wrong = (dist_prev / (dist_prev - dist_curr + 1e-8)).clamp(0.0, 1.0)
    wrong_cross = wrong_crossed & _aperture_check(t_wrong)

    return passed, wrong_cross


# ---------------------------------------------------------------------------
# Crash buffer refresh (called once per step, before termination evaluation)
# ---------------------------------------------------------------------------

def refresh_crash_buffers_for_step(env: ManagerBasedRLEnv) -> None:
    """Update _crashed_buf and _collision_termination_buf after each physics step.

    Two separate buffers are maintained intentionally:
      _crashed_buf               = contact force > threshold  OR  upside-down
      _collision_termination_buf = contact force > threshold ONLY

    The crash PENALTY (_crashed_buf) always fires for both conditions so the agent
    learns to avoid poor orientations even before the curriculum enables termination.
    The termination (_collision_termination_buf) only fires for hard contacts and is
    additionally gated by _collision_detection_enabled (the curriculum).  This means
    upside-down drones are penalised but not terminated until they are competent
    enough that early termination provides a useful training signal.

    Called from update_racing_state() after sensors are refreshed.
    """
    # Re-evaluate the curriculum gate using last-episode metrics (not current-episode,
    # because those are mid-episode and would give noisy signals). Skipped when
    # force_collision_termination or eval_mode is set — eval setups need collisions
    # to terminate from step 0 regardless of training-curriculum state.
    if getattr(env.cfg, "force_collision_termination", False) or getattr(env.cfg, "eval_mode", False):
        env._collision_detection_enabled = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    else:
        env._collision_detection_enabled = check_collision_curriculum_enabled(
            env=env,
            gates_passed_per_env=env._last_episode_gates_passed,
            passage_velocity_per_env=env._last_episode_mean_velocity,
            passage_centering_per_env=env._last_episode_center_offset,
            # passage_count_per_env is passed as ones so the curriculum function receives
            # the already-averaged _last_episode_mean_velocity / _last_episode_center_offset
            # directly (no further averaging needed inside check_collision_curriculum_enabled).
            passage_count_per_env=torch.ones_like(env._last_episode_gates_passed),
            min_gates_passed=env.cfg.curriculum_min_gates_passed,
            min_mean_velocity=env.cfg.curriculum_min_mean_velocity,
            max_center_offset=env.cfg.curriculum_max_center_offset,
        )

    # net_forces_w_history: [N, history_steps, num_bodies, 3] world-frame contact forces.
    net_contact_forces = env._contact_sensor.data.net_forces_w_history
    # For each env: take the norm over the 3 force components, max over history steps,
    # then check whether any body exceeds the threshold.
    collision_detected = torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, env._undesired_contact_body_ids], dim=-1),
            dim=1,
        )[0]
        > env.cfg.collision_force_threshold,
        dim=1,
    )

    upside_down = env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold

    env._collision_termination_buf = collision_detected & env._collision_detection_enabled
    env._crashed_buf = collision_detected | upside_down


# ---------------------------------------------------------------------------
# Per-step racing state update (called from env.step())
# ---------------------------------------------------------------------------

def update_racing_state(env: ManagerBasedRLEnv) -> None:
    """After physics: gate passages, crash buffers, episode speed stats.

    Called once per RL step after the final decimation sub-step so all sensor
    data (contact forces, root state) is fully refreshed before reads.
    """
    drone_pos = env._robot.data.root_pos_w    # world frame [N, 3]
    drone_vel = env._robot.data.root_lin_vel_w  # world frame [N, 3]

    def _refresh_current_gate_cache() -> None:
        gate_pos, gate_ori = gather_current_gate(env)
        env._curr_gate_pos = gate_pos
        env._curr_gate_ori = gate_ori
        env._curr_gate_forward = gate_forward(env, gate_ori)
        env._curr_gate_right = gate_right(env, gate_ori)
        env._curr_gate_up = gate_up(env, gate_ori)

        env._curr_dist_to_gate_vector = gate_pos - drone_pos
        env._curr_dist_to_gate_vector_normalized = env._curr_dist_to_gate_vector / (
            torch.norm(env._curr_dist_to_gate_vector, dim=1, keepdim=True) + 1e-8
        )
        env._curr_dist_to_gate_right = ((-env._curr_dist_to_gate_vector) * env._curr_gate_right).sum(dim=1)
        env._curr_dist_to_gate_up = ((-env._curr_dist_to_gate_vector) * env._curr_gate_up).sum(dim=1)
        env._curr_dist_to_gate_lateral = torch.sqrt(env._curr_dist_to_gate_right**2 + env._curr_dist_to_gate_up**2)
        env._curr_dist_to_gate_forward = -(env._curr_dist_to_gate_vector * env._curr_gate_forward).sum(dim=1)
        env._on_approach_gate_side = env._curr_dist_to_gate_forward <= 0.0

    _refresh_current_gate_cache()
    speed = torch.norm(drone_vel, dim=1)  # [N] m/s
    env._run_max_speed = torch.maximum(env._run_max_speed, speed)  # peak speed since spawn

    # Detect gate crossings
    passed, env._wrong_crossing_buf = check_gate_passages(env, drone_pos)
    
    env._gate_passed_buf = passed.float()

    passed_mask = env._gate_passed_buf > 0.5
    if passed_mask.any():
        env._gates_passed[passed] += 1
        env._route_confirmed_segments[passed] += 1
        # Accumulate passage-quality stats (velocity + lateral offset) for curriculum.
        env._passage_velocity_sum[passed_mask] += speed[passed_mask]
        env._passage_centering_sum[passed_mask] += env._curr_dist_to_gate_lateral[passed_mask]
        env._passage_count[passed_mask] += 1.0

        # Detect lap completions: gates_passed divisible by gate_counts means a full circuit done.
        counts = env._gate_counts[passed_mask].clamp(min=1)
        total = env._gates_passed[passed_mask]
        lap_done = (total % counts) == 0
        if lap_done.any():
            lap_envs = passed_mask.nonzero(as_tuple=False).squeeze(-1)[lap_done]
            # Delta from last lap boundary (or episode start) — not cumulative from episode start.
            lap_time = (env._episode_step_count[lap_envs] - env._last_lap_step[lap_envs]) * env.step_dt
            env._episode_lap_time_sum[lap_envs] += lap_time
            env._episode_lap_count[lap_envs] += 1.0
            env._last_lap_step[lap_envs] = env._episode_step_count[lap_envs]
            # Lap completed → this run's peak speed is now valid; fold into the global max.
            # Restrict to origin-spawned drones when configured (stats_origin_only).
            fold_envs = lap_envs
            if bool(getattr(env.cfg, "stats_origin_only", False)):
                origin = getattr(env, "_spawned_at_origin", None)
                if origin is not None:
                    fold_envs = lap_envs[origin[lap_envs]]
            if fold_envs.numel() > 0:
                env._top_speed = torch.maximum(env._top_speed, env._run_max_speed[fold_envs].max())

    # now advance gate indexes if needed
    if passed.any():
        counts = env._gate_counts[passed]
        new_idx = env._current_gate_idx[passed] + 1
        mask_nz = counts > 0
        res = torch.zeros_like(new_idx)
        if mask_nz.any():
            res[mask_nz] = new_idx[mask_nz] % counts[mask_nz]
        env._current_gate_idx[passed] = res
        
    _refresh_current_gate_cache()
        
    refresh_crash_buffers_for_step(env)  # updates _crashed_buf and _collision_termination_buf

    env._episode_velocity_sum += speed
    env._episode_step_count += 1.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-independent geometric route progress for curriculum decisions.

A task supplies an ordered polyline at reset (gate centres for racing, target
poses for hovering).  The curriculum only consumes the resulting normalized
score and therefore never needs to know what a gate or waypoint is.
"""

from __future__ import annotations

import torch


def initialize_route_progress(env, max_route_points: int) -> None:
    """Allocate the common per-environment route buffers."""
    env._route_points = torch.zeros(env.num_envs, max_route_points, 3, device=env.device)
    env._route_point_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._route_point_dwell_s = torch.zeros(env.num_envs, max_route_points, device=env.device)
    env._route_episode_start_position = torch.zeros(env.num_envs, 3, device=env.device)
    env._route_episode_start_arc = torch.zeros(env.num_envs, device=env.device)
    env._route_confirmed_segments = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._last_episode_route_distance = torch.zeros(env.num_envs, device=env.device)
    env._last_episode_reachable_distance = torch.zeros(env.num_envs, device=env.device)
    env._last_episode_route_progress_score = torch.zeros(env.num_envs, device=env.device)


def _set_routes(env, env_ids: torch.Tensor, task_points: torch.Tensor, counts: torch.Tensor) -> None:
    """Prepend a route origin, set ordered task points, and project the spawn."""
    if env_ids.numel() == 0:
        return
    env._route_points[env_ids] = 0.0
    max_task_points = task_points.shape[1]
    env._route_points[env_ids, 0] = env._route_episode_start_position[env_ids]
    for offset in range(max_task_points):
        env._route_points[env_ids, offset + 1] = task_points[:, offset]
    env._route_point_counts[env_ids] = counts + 1
    env._route_confirmed_segments[env_ids] = 0
    start_arc, _ = project_onto_route(
        env._route_points[env_ids],
        env._route_point_counts[env_ids],
        env._route_episode_start_position[env_ids],
    )
    env._route_episode_start_arc[env_ids] = start_arc


def begin_racing_routes(env, env_ids: torch.Tensor) -> None:
    """Create ordered gate-center polylines for the new episodes."""
    if env_ids.numel() == 0:
        return
    counts = env._gate_counts[env_ids]
    max_gates = env._gate_positions.shape[1]
    offsets = torch.arange(max_gates, device=env.device).unsqueeze(0)
    # Routes always rotate the polyline to the drone's active gate for one lap. (An "official
    # route prefix" starting before the active gate used to be supported here for a since-removed
    # user; the AI-GP experimental snapshot carries its own copy of this module with that feature.)
    start_idx = env._current_gate_idx[env_ids]
    indices = (start_idx.unsqueeze(1) + offsets) % counts.unsqueeze(1).clamp(min=1)
    ordered = torch.gather(env._gate_positions[env_ids], 1, indices.unsqueeze(-1).expand(-1, -1, 3))
    valid = offsets < counts.unsqueeze(1)
    ordered = torch.where(valid.unsqueeze(-1), ordered, torch.zeros_like(ordered))
    _set_routes(env, env_ids, ordered, counts)
    # Rotated base-racing routes and spawn-to-waypoint routes always begin at segment zero.
    env._route_confirmed_segments[env_ids] = torch.zeros_like(env._current_gate_idx[env_ids])
    env._route_point_dwell_s[env_ids] = 0.0


def begin_hovering_routes(env, env_ids: torch.Tensor) -> None:
    """Create spawn-to-ordered-target-pose polylines for hovering episodes."""
    if env_ids.numel() == 0:
        return
    _set_routes(env, env_ids, env._waypoint_positions[env_ids], env._waypoint_counts[env_ids])
    env._route_point_dwell_s[env_ids] = 0.0
    point_ids = torch.arange(env._route_points.shape[1] - 1, device=env.device).unsqueeze(0)
    valid = point_ids < env._waypoint_counts[env_ids].unsqueeze(1)
    waypoint_dwell = torch.where(
        valid,
        env._collection_time[env_ids].unsqueeze(1),
        torch.zeros_like(valid, dtype=torch.float32),
    )
    env._route_point_dwell_s[env_ids, 1:] = waypoint_dwell


def project_onto_route(
    route_points: torch.Tensor,
    point_counts: torch.Tensor,
    query_points: torch.Tensor,
    max_segment_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return closest-point arc coordinate, optionally bounded to confirmed segments."""
    segments = route_points[:, 1:] - route_points[:, :-1]
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    segment_ids = torch.arange(segments.shape[1], device=route_points.device).unsqueeze(0)
    valid = segment_ids < (point_counts - 1).clamp(min=0).unsqueeze(1)
    if max_segment_indices is not None:
        valid &= segment_ids <= max_segment_indices.unsqueeze(1)

    rel = query_points.unsqueeze(1) - route_points[:, :-1]
    denom = lengths.square().clamp(min=1.0e-12)
    t = (rel * segments).sum(dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    projected = route_points[:, :-1] + t.unsqueeze(-1) * segments
    distance_sq = (query_points.unsqueeze(1) - projected).square().sum(dim=-1)
    distance_sq = torch.where(valid, distance_sq, torch.full_like(distance_sq, torch.inf))
    closest_segment = distance_sq.argmin(dim=1)

    cumulative = torch.cumsum(torch.where(valid, lengths, torch.zeros_like(lengths)), dim=1)
    before = cumulative - torch.where(valid, lengths, torch.zeros_like(lengths))
    row = torch.arange(route_points.shape[0], device=route_points.device)
    arc = before[row, closest_segment] + t[row, closest_segment] * lengths[row, closest_segment]
    total = cumulative[:, -1]
    return arc, total


def _reachable_arc_distance(
    route_points: torch.Tensor,
    point_counts: torch.Tensor,
    start_arc: torch.Tensor,
    speeds: torch.Tensor,
    episode_time_s: float,
    point_dwell_s: torch.Tensor,
) -> torch.Tensor:
    """Maximum forward arc distance after travel and ordered pose dwell times."""
    segment_lengths = torch.linalg.vector_norm(route_points[:, 1:] - route_points[:, :-1], dim=-1)
    vertex_arcs = torch.cat(
        (torch.zeros(route_points.shape[0], 1, device=route_points.device), torch.cumsum(segment_lengths, dim=1)),
        dim=1,
    )
    arc = start_arc.clone()
    time_left = torch.full_like(start_arc, float(episode_time_s))
    safe_speed = speeds.clamp(min=1.0e-6)
    for point_idx in range(route_points.shape[1]):
        valid = point_idx < point_counts
        target_arc = vertex_arcs[:, point_idx]
        distance = (target_arc - arc).clamp(min=0.0)
        travel_time = distance / safe_speed
        reached = valid & (travel_time <= time_left)
        partial = valid & (~reached) & (time_left > 0.0)
        arc = torch.where(reached, torch.maximum(arc, target_arc), arc)
        arc = torch.where(partial, arc + safe_speed * time_left, arc)
        time_left = torch.where(reached, time_left - travel_time, torch.where(partial, torch.zeros_like(time_left), time_left))

        # Holding the final pose cannot unlock any more geometric progress.
        has_later_point = valid & (point_idx + 1 < point_counts)
        time_left = torch.where(
            reached & has_later_point,
            (time_left - point_dwell_s[:, point_idx]).clamp(min=0.0),
            time_left,
        )
    return (arc - start_arc).clamp(min=0.0)


def snapshot_episode_route_progress(env, env_ids: torch.Tensor) -> dict[str, float]:
    """Store progress relative to what the episode's speed limit made reachable."""
    if env_ids.numel() == 0:
        return {}
    points = env._route_points[env_ids]
    counts = env._route_point_counts[env_ids]
    # Gate/waypoint completion is the task-owned proof of progress.  Restrict
    # projection to the current route segment to prevent self-intersections or
    # shortcuts from receiving credit for future route sections.
    max_segments = torch.minimum(
        env._route_confirmed_segments[env_ids].clamp(min=0), (counts - 2).clamp(min=0)
    )
    arc, total = project_onto_route(
        points, counts, env._robot.data.root_pos_w[env_ids], max_segment_indices=max_segments
    )
    start_arc = env._route_episode_start_arc[env_ids]
    travelled = (arc - start_arc).clamp(min=0.0)
    valid_route = counts >= 2

    episode_time = float(env.cfg.episode_length_s)
    speed_limits_cfg = getattr(env.cfg, "speed_limits", None)
    if speed_limits_cfg is not None and speed_limits_cfg.enabled:
        reference_speed = env._velocity_limit[env_ids]
    else:
        reference_speed = torch.full_like(
            start_arc, float(getattr(env.cfg, "curriculum_reference_speed", 15.0))
        )
    remaining = (total - start_arc).clamp(min=0.0)
    reachable = torch.minimum(
        remaining,
        _reachable_arc_distance(
            points,
            counts,
            start_arc,
            reference_speed,
            episode_time,
            env._route_point_dwell_s[env_ids],
        ),
    )
    score = torch.where(
        valid_route & (reachable > 1.0e-6),
        (travelled / reachable).clamp(0.0, 1.0),
        torch.zeros_like(travelled),
    )
    env._last_episode_route_distance[env_ids] = travelled
    env._last_episode_reachable_distance[env_ids] = reachable
    env._last_episode_route_progress_score[env_ids] = score
    return {
        "Curriculum/route_distance_m": float(travelled.mean().item()),
        "Curriculum/reachable_route_distance_m": float(reachable.mean().item()),
        "Curriculum/normalized_route_progress": float(score.mean().item()),
    }

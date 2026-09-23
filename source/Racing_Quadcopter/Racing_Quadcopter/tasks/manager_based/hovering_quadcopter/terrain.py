# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generator for the hovering / waypoint-following task.

Produces an 8x8 grid of flat plane sub-terrains. Waypoint tracks are NOT
generated inside the sub-terrain callback (TerrainGenerator differentiates
the seed per cell but does not expose row/col), but rather later in the
env, where we have explicit control over per-cell parameters and can force
the first ``num_zero_tc_cells`` to have ``T_c = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass


def _flat_terrain(difficulty: float, cfg: "MeshFlatTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Return a single flat ground plane mesh + the cell's corner-origin centre."""
    ground = make_plane(cfg.size, height=0.0, center_zero=False)
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return [ground], origin


@configclass
class MeshFlatTerrainCfg(SubTerrainBaseCfg):
    """Minimal flat-ground sub-terrain. No obstacles, no markers — those are
    handled separately by the env.
    """

    function = _flat_terrain
    seed: int | None = None


def create_waypoint_terrain(
    seed: int = 42,
    num_rows: int = 8,
    num_cols: int = 8,
    cell_size: tuple[float, float] = (15.0, 15.0),
) -> TerrainGeneratorCfg:
    """Return a TerrainGeneratorCfg that tiles an ``num_rows x num_cols`` grid
    of flat plane cells. The env stores ``terrain_origins`` and uses one
    pre-generated waypoint track per cell.
    """
    return TerrainGeneratorCfg(
        seed=seed,
        size=cell_size,
        difficulty_range=(0.0, 0.0),
        border_width=0.5,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "flat": MeshFlatTerrainCfg(proportion=1.0, seed=seed),
        },
    )


# ---------------------------------------------------------------------------
# Per-cell waypoint generation (run by the env, not the TerrainGenerator)
# ---------------------------------------------------------------------------

@dataclass
class CellTrack:
    """One waypoint track in the cell's local frame (origin at cell corner)."""

    waypoints: np.ndarray  # [G, 3] local-frame positions
    collection_time_s: float


def _sample_track(
    rng: np.random.Generator,
    cell_size: tuple[float, float],
    num_waypoints: int,
    xy_half_range: float,
    z_range: tuple[float, float],
    min_spacing: float,
) -> np.ndarray:
    """Rejection-sample ``num_waypoints`` 3D positions inside a cell-centred box.

    Sampling box is centred on the cell centre, half-width ``xy_half_range``
    in X and Y (clamped to stay inside the cell), Z uniform in ``z_range``.
    """
    cx, cy = cell_size[0] / 2.0, cell_size[1] / 2.0
    margin = 1.0
    half_x = min(xy_half_range, cx - margin)
    half_y = min(xy_half_range, cy - margin)

    pts: list[np.ndarray] = []
    attempts = 0
    max_attempts = num_waypoints * 200
    while len(pts) < num_waypoints and attempts < max_attempts:
        attempts += 1
        candidate = np.array(
            [
                cx + rng.uniform(-half_x, half_x),
                cy + rng.uniform(-half_y, half_y),
                rng.uniform(z_range[0], z_range[1]),
            ]
        )
        if all(np.linalg.norm(candidate - p) >= min_spacing for p in pts):
            pts.append(candidate)
    # Backfill if we hit the attempt cap (e.g., too-tight spacing for the box).
    while len(pts) < num_waypoints:
        pts.append(
            np.array(
                [
                    cx + rng.uniform(-half_x, half_x),
                    cy + rng.uniform(-half_y, half_y),
                    rng.uniform(z_range[0], z_range[1]),
                ]
            )
        )
    return np.stack(pts, axis=0)


def generate_random_tracks(
    base_seed: int,
    num_cells: int,
    num_waypoints_range: tuple[int, int] = (4, 8),
    waypoint_xy_range: float = 3.0,
    waypoint_z_range: tuple[float, float] = (0.5, 2.5),
    min_waypoint_spacing: float = 0.8,
    cell_size: tuple[float, float] = (15.0, 15.0),
    num_zero_tc_cells: int = 16,
    collection_time_range_s: tuple[float, float] = (0.2, 2.0),
) -> list[CellTrack]:
    """Generate ``num_cells`` deterministic random tracks.

    The first ``num_zero_tc_cells`` cells are forced to ``T_c = 0`` so the
    fly-through behaviour is always represented in the training distribution.
    Remaining cells sample T_c uniformly in ``collection_time_range_s``.
    """
    tracks: list[CellTrack] = []
    n_zero = max(0, min(int(num_zero_tc_cells), int(num_cells)))
    for cell_idx in range(num_cells):
        # Independent RNGs for the count, the waypoints, and T_c — keeps each
        # cell reproducible from ``base_seed`` alone and decorrelates the draws.
        rng = np.random.default_rng(base_seed + cell_idx * 1009 + 7)
        n_waypoints = int(rng.integers(num_waypoints_range[0], num_waypoints_range[1] + 1))
        waypoints = _sample_track(
            rng=rng,
            cell_size=cell_size,
            num_waypoints=n_waypoints,
            xy_half_range=waypoint_xy_range,
            z_range=waypoint_z_range,
            min_spacing=min_waypoint_spacing,
        )
        if cell_idx < n_zero:
            tc = 0.0
        else:
            tc = float(rng.uniform(collection_time_range_s[0], collection_time_range_s[1]))
        tracks.append(CellTrack(waypoints=waypoints, collection_time_s=tc))
    return tracks


def broadcast_yaml_track(
    waypoints_local: np.ndarray,
    collection_time_s: float,
    num_cells: int,
) -> list[CellTrack]:
    """Reuse the same fixed track across all cells when WAYPOINTS=... is set."""
    return [CellTrack(waypoints=waypoints_local.copy(), collection_time_s=collection_time_s)
            for _ in range(num_cells)]
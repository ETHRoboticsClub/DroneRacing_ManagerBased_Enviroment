# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate racing gate terrains using the ``trimesh`` library.

Overview
--------
``racing_gates_terrain`` is the entry point called by IsaacLab's TerrainGenerator
once per terrain cell.  It builds:
  1. A flat ground-plane mesh.
  2. A procedural closed-loop trajectory (figure-8, oval, or random spline).
  3. Gate frames placed at arc-length intervals along that trajectory.

Registry pattern
----------------
TerrainGenerator **copies** the cfg dataclass before calling the terrain function,
so any attribute written to cfg inside the function is silently discarded. Gate data
is stored in ``gate_base.gate_registry``, keyed by ``(round(difficulty, 6), seed)``,
and retrieved by the env after all terrain cells have been generated.

Seed per cell
-------------
TerrainGenerator differentiates the seed for every cell before invoking the terrain
function.  In a 4-row × 4-col grid this yields 16 unique (difficulty, seed) registry
keys — every cell has a distinct random gate layout.
"""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_plane

from .gate_base import GateData, gate_registry, create_gate_frame

if TYPE_CHECKING:
    from .racing_gates_cfg import MeshRacingGatesTerrainCfg


def racing_gates_terrain(
    difficulty: float, cfg: "MeshRacingGatesTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat terrain with racing gates placed along a generated trajectory.

    Called by TerrainGenerator once per terrain cell with the cell's difficulty
    value.  Returns trimesh objects for the cell and its local origin so the
    generator can compose the full scene.

    Args:
        difficulty: 0-1 float assigned by TerrainGenerator (increases with row).
                    Higher difficulty → smaller gates, more complex trajectories.
        cfg: Sub-terrain configuration (seed already overwritten by generator).

    Returns:
        meshes:  Ground plane + gate bars + label segments + direction indicators.
        origin:  Centre of the ground plane at Z=0 — TerrainGenerator uses this to
                 position the cell in the world grid.
    """
    meshes_list = []

    # Ground plane at Z=0, corner at (0,0,0), extending to (size_x, size_y, 0).
    # center_zero=False keeps origin at the corner, matching the gate placement
    # coordinate system.  TerrainGenerator later applies its own centering.
    ground_mesh = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_mesh)

    # Both trajectory and gate placement use the same seed value so the track shape
    # and gate positions are reproducible.  Note: two separate RNG objects are created
    # from the same seed — this means the placement RNG starts in the same state as
    # the trajectory RNG, which can cause correlation between waypoint noise and
    # gate spacings.  This is intentional for reproducibility; if independent
    # randomness is desired, use seed+1 for one of them.
    rng_traj = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    waypoints, tangents = _generate_racing_trajectory(
        size=cfg.size,
        num_waypoints=cfg.num_waypoints,
        trajectory_type=cfg.trajectory_type,
        difficulty=difficulty,
        height_range=cfg.gate_height_range,
        rng=rng_traj,
    )

    # Gate size decreases linearly with difficulty:
    #   difficulty=0 → gate_size_range[1] (largest, easiest)
    #   difficulty=1 → gate_size_range[0] (smallest, hardest)
    gate_size = cfg.gate_size_range[1] - difficulty * (cfg.gate_size_range[1] - cfg.gate_size_range[0])

    placement_rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    gate_positions, gate_orientations = _place_gates_along_trajectory(
        waypoints=waypoints,
        tangents=tangents,
        gate_spacing_range=cfg.gate_spacing_range,
        size=cfg.size,
        rng=placement_rng,
    )

    # Write to module-level registry so the env can retrieve it after terrain gen.
    # round(difficulty, 6) guards against floating-point key collisions for cells
    # that are assigned the same nominal difficulty by the generator.
    gate_registry[(round(difficulty, 6), cfg.seed)] = GateData(
        positions=gate_positions.copy(),
        orientations=gate_orientations.copy(),
        gate_size=gate_size,
        bar_thickness=cfg.bar_thickness,
        waypoints=waypoints.copy(),
    )

    # Build trimesh geometry for every gate: frame bars only.
    # Labels and direction indicators are drawn as collision-free USD BasisCurves
    # from the env (see trajectory_vis.py).
    for pos, quat in zip(gate_positions, gate_orientations):
        meshes_list.extend(create_gate_frame(
            position=pos,
            orientation=quat,
            gate_size=gate_size,
            bar_thickness=cfg.bar_thickness,
            color=cfg.gate_color,
        ))

    # The origin is the centre of this cell's ground patch (corner-origin + size/2).
    # TerrainGenerator uses this to align the mesh in the world grid.
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return meshes_list, origin


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def _generate_racing_trajectory(
    size: tuple[float, float],
    num_waypoints: int,
    trajectory_type: str,
    difficulty: float,
    height_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate closed-loop trajectory generator.

    Returns:
        waypoints: [N, 3] positions of trajectory sample points.
        tangents:  [N, 3] unit tangent vectors at each waypoint (forward direction).
    """
    center_x = size[0] / 2.0
    center_y = size[1] / 2.0

    # Keep 80 % of the available radius to leave margin for the border + gate depth.
    margin = 1.0
    max_radius_x = (size[0] / 2.0 - margin) * 0.8
    max_radius_y = (size[1] / 2.0 - margin) * 0.8

    if trajectory_type == "figure8":
        return _generate_figure8(center_x, center_y, max_radius_x, max_radius_y,
                                 num_waypoints, height_range, difficulty, rng)
    elif trajectory_type == "oval":
        return _generate_oval(center_x, center_y, max_radius_x, max_radius_y,
                              num_waypoints, height_range, difficulty, rng)
    else:  # "random" — pick a sub-style per cell for visual variety
        style = int(rng.integers(0, 3))   # 0,1 → spline variants; 2 → chicane
        if style == 2:
            return _generate_chicane_track(center_x, center_y, max_radius_x, max_radius_y,
                                           num_waypoints, height_range, difficulty, rng)
        return _generate_random_track(center_x, center_y, max_radius_x, max_radius_y,
                                      num_waypoints, height_range, difficulty, rng)


def _generate_figure8(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Lemniscate-like figure-8 parametric trajectory.

    The XY path is deterministic given cx/cy/rx/ry, where c represents the center 
    and r the rays; only the Z variation uses rng.
    Tangents are computed as finite differences between consecutive waypoints and
    then normalised.  The loop closes because t is sampled on [0, 2π) endpoint=False.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    scale = 0.7  # keeps the figure-8 comfortably within the padded radius
    x = cx + rx * scale * np.sin(t)
    y = cy + ry * scale * np.sin(t) * np.cos(t)  # lemniscate Y: sin*cos gives two loops

    # Height oscillates twice per lap (sin(2t)) with amplitude scaled by difficulty.
    height_var = (height_range[1] - height_range[0]) * difficulty * 0.5
    base_height = (height_range[0] + height_range[1]) / 2.0
    z = base_height + height_var * np.sin(2 * t) + rng.uniform(-0.1, 0.1, n) * difficulty
    z = np.clip(z, height_range[0], height_range[1])

    waypoints = np.stack([x, y, z], axis=1)

    # Finite-difference tangents with wrap-around at the last waypoint.
    tangents = np.zeros_like(waypoints)
    for i in range(n):
        tangents[i] = waypoints[(i + 1) % n] - waypoints[i]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8

    return waypoints, tangents


def _generate_oval(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simple elliptical trajectory.C represents the center 
    and r the rays; Z varies in height_range.

    Straightforward parametric ellipse; height varies with sin(2t).
    Less complex than random but useful for debugging gate passage logic.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    x = cx + rx * 0.8 * np.cos(t)
    y = cy + ry * 0.8 * np.sin(t)

    height_var = (height_range[1] - height_range[0]) * difficulty * 0.3
    base_height = (height_range[0] + height_range[1]) / 2.0
    z = base_height + height_var * np.sin(2 * t)
    z = np.clip(z, height_range[0], height_range[1])

    waypoints = np.stack([x, y, z], axis=1)

    tangents = np.zeros_like(waypoints)
    for i in range(n):
        tangents[i] = waypoints[(i + 1) % n] - waypoints[i]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8

    return waypoints, tangents


def _spline_track(
    t_ctrl: np.ndarray,
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    ctrl_z: np.ndarray,
    n: int,
    height_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a periodic CubicSpline through control points and sample n waypoints.
    t_ctrl is an array of angular parameters, while ctrl_x,ctrl_y,ctrl_z 
    are the coordinates of the control points (one per each value of t_ctrl)"""
    from scipy.interpolate import CubicSpline
    t_ext = np.append(t_ctrl, 2 * np.pi)
    cs_x = CubicSpline(t_ext, np.append(ctrl_x, ctrl_x[0]), bc_type="periodic")
    cs_y = CubicSpline(t_ext, np.append(ctrl_y, ctrl_y[0]), bc_type="periodic")
    cs_z = CubicSpline(t_ext, np.append(ctrl_z, ctrl_z[0]), bc_type="periodic")
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    waypoints = np.stack([cs_x(t), cs_y(t), np.clip(cs_z(t), *height_range)], axis=1)
    tangents  = np.stack([cs_x(t, 1), cs_y(t, 1), cs_z(t, 1)], axis=1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    return waypoints, tangents


def _generate_random_track(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random closed-loop track — aggressive at high difficulty.

    Difficulty effects:
      - More control points → more corners and direction changes.
      - Larger radius perturbation amplitude → sharper turns.
      - Multiple stacked harmonic distortions for unique shapes per cell.
      - Height amplitude scales to almost the full height range at difficulty=1.
    """
    num_control = max(12, int(16 + difficulty * 14))  # 16 easy → 30 hard
    t_ctrl = np.linspace(0, 2 * np.pi, num_control, endpoint=False)

    # Base radius noise — large amplitude at high difficulty produces sharp corners.
    perturb_amp = 0.3 + difficulty * 0.65
    r_perturb = 1.0 + rng.uniform(-perturb_amp, perturb_amp, num_control)

    # Stack 2–4 independent harmonic distortions so every cell looks different.
    for _ in range(int(rng.integers(2, 5))):
        freq = rng.uniform(1.5, max(2.0, num_control / 3.0))
        amp  = rng.uniform(0.1, 0.35) * (0.3 + difficulty * 0.7)
        r_perturb *= 1.0 + amp * np.sin(freq * t_ctrl + rng.uniform(0, 2 * np.pi))

    r_perturb = np.clip(r_perturb, 0.25, 2.2)

    ctrl_x = cx + rx * 0.88 * np.cos(t_ctrl) * r_perturb
    ctrl_y = cy + ry * 0.88 * np.sin(t_ctrl) * r_perturb

    # Multiple independent spiral distortions for asymmetry.
    for _ in range(int(rng.integers(1, 4))):
        freq  = rng.uniform(2.0, 6.0)
        amp_x = rng.uniform(0.1, 0.4) * rx * (0.2 + difficulty * 0.8)
        amp_y = rng.uniform(0.1, 0.4) * ry * (0.2 + difficulty * 0.8)
        phase = rng.uniform(0, 2 * np.pi)
        ctrl_x += amp_x * np.sin(freq * t_ctrl + phase)
        ctrl_y += amp_y * np.cos(freq * t_ctrl + phase)

    # Clamp to tile interior so no control point sits outside the terrain.
    ctrl_x = np.clip(ctrl_x, 1.5, 2 * cx - 1.5)
    ctrl_y = np.clip(ctrl_y, 1.5, 2 * cy - 1.5)

    # Heights: 3–5 stacked sinusoids; at difficulty=1 amplitude spans ~90% of range.
    h_span = height_range[1] - height_range[0]
    ctrl_z = rng.uniform(height_range[0], height_range[1], num_control)
    for _ in range(int(rng.integers(3, 6))):
        freq = rng.uniform(0.5, num_control / 2.5)
        amp  = h_span * rng.uniform(0.1, 0.3) * (0.3 + difficulty * 0.7)
        ctrl_z += amp * np.sin(freq * t_ctrl + rng.uniform(0, 2 * np.pi))
    ctrl_z = np.clip(ctrl_z, height_range[0], height_range[1])

    return _spline_track(t_ctrl, ctrl_x, ctrl_y, ctrl_z, n, height_range)


def _generate_chicane_track(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop track built around alternating lateral offsets (chicane sections).

    Control points sit on an oval, then are displaced alternately inward/outward
    along the local normal.  This produces a track with many rapid direction
    reversals — at high difficulty the offsets are large and sections are short,
    creating extremely tight S-curves.
    """
    # More sections at higher difficulty → more frequent direction changes.
    num_sections = max(6, int(8 + difficulty * 10))  # 8 easy → 18 hard
    t_ctrl = np.linspace(0, 2 * np.pi, num_sections, endpoint=False)

    # Base oval, slightly inside the available radius.
    base_x = cx + rx * 0.70 * np.cos(t_ctrl)
    base_y = cy + ry * 0.70 * np.sin(t_ctrl)

    # Inward normal of the oval at each control point: (-cos, -sin).
    normal_x = -np.cos(t_ctrl)
    normal_y = -np.sin(t_ctrl)

    # Alternating sign creates chicane effect; amplitude grows with difficulty.
    chicane_amp = rx * (0.20 + difficulty * 0.45)
    signs = np.where(np.arange(num_sections) % 2 == 0, 1.0, -1.0)
    # Add per-section random variation so consecutive chicanes aren't identical.
    signs *= rng.uniform(0.6, 1.0, num_sections)

    ctrl_x = np.clip(base_x + normal_x * chicane_amp * signs, 1.5, 2 * cx - 1.5)
    ctrl_y = np.clip(base_y + normal_y * chicane_amp * signs, 1.5, 2 * cy - 1.5)

    # Heights alternate up/down with each chicane to add 3-D difficulty.
    h_span  = height_range[1] - height_range[0]
    h_mid   = (height_range[0] + height_range[1]) / 2.0
    h_amp   = h_span * (0.35 + difficulty * 0.55)
    ctrl_z  = np.clip(
        h_mid + h_amp * signs * rng.uniform(0.7, 1.0, num_sections)
        + rng.uniform(-0.2, 0.2, num_sections) * h_span * 0.15,
        height_range[0], height_range[1],
    )

    return _spline_track(t_ctrl, ctrl_x, ctrl_y, ctrl_z, n, height_range)


# ---------------------------------------------------------------------------
# Gate placement
# ---------------------------------------------------------------------------

def _place_gates_along_trajectory(
    waypoints: np.ndarray,
    tangents: np.ndarray,
    gate_spacing_range: tuple[float, float],
    size: tuple[float, float],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place gates at arc-length intervals along a closed-loop trajectory.

    Gate count is determined purely by how many fit in one lap at the average
    spacing: ``num_gates = int(total_arc / avg_spacing)``.  There is no minimum
    or maximum guard — if the circuit is short relative to spacing, few (or zero)
    gates are placed.  Gates that land outside the terrain bounds are skipped
    (advancing arc position by that spacing) which can reduce the final count.

    Gates do NOT wrap across laps: placement stops when arc position exceeds
    total_arc.

    Args:
        waypoints:          [N, 3] closed-loop trajectory waypoints.
        tangents:           [N, 3] unit tangents at each waypoint.
        gate_spacing_range: (min, max) metres between consecutive gates.
                            Use (n, n) for fixed spacing.
        size:               (width, height) of the terrain tile in metres.
        rng:                Random number generator; created if None.

    Returns:
        gate_positions:    [M, 3] gate centre positions in terrain local frame.
        gate_orientations: [M, 4] quaternions (w, x, y, z); always upright (Z=world up).
    """
    n = waypoints.shape[0]

    # Arc-length parameterisation: cumulative Euclidean distance between waypoints.
    # The last segment closes the loop back to waypoint 0.
    segment_lengths = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
    segment_lengths = np.append(segment_lengths, np.linalg.norm(waypoints[0] - waypoints[-1]))
    cumulative_arc = np.zeros(n + 1)
    cumulative_arc[1:] = np.cumsum(segment_lengths)
    total_arc = cumulative_arc[-1]

    avg_spacing = (gate_spacing_range[0] + gate_spacing_range[1]) / 2.0
    num_gates = int(total_arc / avg_spacing)  # upper bound; actual count may be lower

    if rng is None:
        rng = np.random.default_rng()

    # Sample all spacings up front; each gate consumes one value.
    spacings = rng.uniform(gate_spacing_range[0], gate_spacing_range[1], num_gates)

    gate_positions = []
    gate_orientations = []

    current_arc = 0.0
    for i in range(num_gates):
        if current_arc >= total_arc:
            break  # one lap completed; stop (no wrap-around)

        # Binary search for the waypoint segment that contains current_arc.
        idx = np.searchsorted(cumulative_arc[:-1], current_arc)
        idx = np.clip(idx, 1, n) - 1

        # Linear interpolation within the segment for position and tangent.
        if idx < n - 1:
            prev_arc  = cumulative_arc[idx]
            next_arc  = cumulative_arc[idx + 1]
            t = (current_arc - prev_arc) / (next_arc - prev_arc + 1e-8)
            pos     = (1 - t) * waypoints[idx]  + t * waypoints[idx + 1]
            tangent = (1 - t) * tangents[idx]   + t * tangents[idx + 1]
        else:
            pos     = waypoints[idx]
            tangent = tangents[idx]

        # Skip gates closer than 0.5 m to the tile boundary (avoids clipping).
        margin = 0.5
        if not (margin < pos[0] < size[0] - margin and margin < pos[1] < size[1] - margin):
            current_arc += spacings[i]
            continue

        # Gate orientation — always UPRIGHT:
        # +Y = horizontal projection of trajectory tangent (passage direction)
        # +Z = world up (gates never tilt even on steep trajectory segments)
        # +X = cross(tangent_h, +Z) = gate right
        tangent_h = tangent.copy()
        tangent_h[2] = 0.0
        tangent_h /= np.linalg.norm(tangent_h) + 1e-8

        z_axis = np.array([0.0, 0.0, 1.0])
        right  = np.cross(tangent_h, z_axis)
        right /= np.linalg.norm(right) + 1e-8

        # Rotation matrix columns = [right, forward, up] → maps local gate frame to world.
        rot_mat = np.stack([right, tangent_h, z_axis], axis=1)
        rot = tf.Rotation.from_matrix(rot_mat)
        # scipy returns (x, y, z, w); convert to IsaacLab's (w, x, y, z).
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        gate_positions.append(pos)
        gate_orientations.append(quat_wxyz)
        current_arc += spacings[i]

    if not gate_positions:
        return np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=float)
    return np.array(gate_positions), np.array(gate_orientations)




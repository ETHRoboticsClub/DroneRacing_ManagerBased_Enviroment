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
  4. Visual extras: a 7-segment gate-index label and a forward-direction indicator.

Registry pattern
----------------
TerrainGenerator **copies** the cfg dataclass before calling the terrain function,
so any attribute written to cfg inside the function is silently discarded. Gate data
(positions, orientations, size) therefore cannot be returned via cfg.  Instead they
are stored in the module-level ``_gate_registry`` dict, keyed by
``(round(difficulty, 6), seed)``, and retrieved by the env after all terrain cells
have been generated.

Seed per cell
-------------
TerrainGenerator differentiates the seed for every cell before invoking the terrain
function.  In a 4-row × 4-col grid this yields 16 unique (difficulty, seed) registry
keys — every cell has a distinct random gate layout.  Difficulty is assigned per row
(4 difficulty levels across 4 rows); columns within a row share the same difficulty
but have different seeds, so their circuits still differ.

Gate coordinate system
----------------------
  +X  right  (horizontal, along gate bar width)
  +Y  forward (passage direction — the drone should fly through in +Y)
  +Z  up     (vertical; gates are always upright, Z always = world up)

All quaternions are stored in (w, x, y, z) order to match IsaacLab convention.
scipy.spatial.transform uses (x, y, z, w); conversions are explicit at every
call site.
"""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import trimesh
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from .racing_gates_cfg import MeshRacingGatesTerrainCfg


@dataclass
class GateData:
    """Gate data produced by a single sub-terrain generation call.

    All arrays are in the sub-terrain's **local corner-origin frame** — i.e. the
    coordinate system where (0,0,0) is the corner of the terrain mesh as built by
    make_plane.  The environment must add the TerrainGenerator's centre-offset
    (−size/2 in X and Y) and then the per-cell terrain_origin to convert to world
    frame (see _setup_gate_tracking in racing_quadcopter_env.py).
    """
    positions: np.ndarray       # [num_gates, 3]  gate centre positions, local frame
    orientations: np.ndarray    # [num_gates, 4]  quaternions (w, x, y, z), local frame
    gate_size: float            # aperture side length in metres (square gate)
    start_position: np.ndarray  # [3]  first waypoint — informational, not currently used
    start_tangent: np.ndarray   # [3]  tangent at first waypoint — informational
    waypoints: np.ndarray       # [num_waypoints, 3]  full trajectory sample points, local frame


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------
# Dict from (difficulty, seed) → GateData for every cell generated in the
# current run.  The env reads this after TerrainGenerator finishes.
# Safe to clear between independent training runs if needed.
_gate_registry: dict[tuple[float, int | None], GateData] = {}


def get_gate_registry() -> dict[tuple[float, int | None], GateData]:
    """Return a reference to the module-level gate data registry."""
    return _gate_registry


def clear_gate_registry() -> None:
    """Clear all stored gate data (call between training runs if needed)."""
    _gate_registry.clear()


def racing_gates_terrain(
    difficulty: float, cfg: "MeshRacingGatesTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat terrain with racing gates placed along a generated trajectory.

    Called by TerrainGenerator once per terrain cell with the cell's difficulty
    value.  Returns trimesh objects for the cell and its local origin so the
    generator can compose the full scene.

    Args:
        difficulty: 0–1 float assigned by TerrainGenerator (increases with row).
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
    _gate_registry[(round(difficulty, 6), cfg.seed)] = GateData(
        positions=gate_positions.copy(),
        orientations=gate_orientations.copy(),
        gate_size=gate_size,
        start_position=waypoints[0].copy(),
        start_tangent=tangents[0].copy(),
        waypoints=waypoints.copy(),
    )

    # Build trimesh geometry for every gate: frame bars only.
    # Labels and direction indicators are drawn as collision-free USD BasisCurves
    # from the env (see trajectory_vis.py).
    for pos, quat in zip(gate_positions, gate_orientations):
        meshes_list.extend(_create_gate_frame(
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
    else:  # "random" — cubic-spline through randomised control points
        return _generate_random_track(center_x, center_y, max_radius_x, max_radius_y,
                                      num_waypoints, height_range, difficulty, rng)


def _generate_figure8(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Lemniscate-like figure-8 parametric trajectory.

    The XY path is deterministic given cx/cy/rx/ry; only the Z variation uses rng.
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
    """Simple elliptical trajectory.

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


def _generate_random_track(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random smooth closed-loop track via cubic-spline through randomised control points.

    The algorithm:
      1. Place num_control points evenly around a circle in parameter space (0–2π).
      2. Perturb their radial distance with layered noise (base + secondary oscillation).
      3. Add a random spiral offset in XY for extra 3D complexity.
      4. Assign heights from a uniform base + two superimposed sinusoids.
      5. Fit a periodic CubicSpline through the closed control polygon.
      6. Sample the spline at n evenly-spaced parameter values.
      7. Compute tangents analytically from the spline's first derivative (exact, no FD error).

    Difficulty effects:
      - num_control scales with difficulty (more control points = more corners/turns).
      - Radius perturbation amplitude grows with difficulty.
      - Height oscillation amplitude grows with difficulty.
    """
    # More control points at higher difficulty → more complex, tighter circuits.
    num_control = max(8, int(12 + difficulty * 8))

    t_ctrl = np.linspace(0, 2 * np.pi, num_control, endpoint=False)

    # Layered radius perturbation: base noise × secondary sinusoidal oscillation.
    r_perturb_base = 1.0 + rng.uniform(-0.6, 0.6, num_control) * (0.5 + difficulty * 0.5)
    secondary_freq = rng.uniform(1.5, 3.5)
    secondary_amp  = rng.uniform(0.2, 0.4)
    r_perturb = r_perturb_base * (1.0 + secondary_amp * np.sin(secondary_freq * t_ctrl))

    # Control point positions: 85 % of available radius × perturbation.
    ctrl_x = cx + rx * 0.85 * np.cos(t_ctrl) * r_perturb
    ctrl_y = cy + ry * 0.85 * np.sin(t_ctrl) * r_perturb

    # Random spiral adds non-circular XY distortion on top of the radius perturbation.
    spiral_strength = rng.uniform(0.1, 0.3)
    ctrl_x += spiral_strength * rx * np.sin(3 * t_ctrl)
    ctrl_y += spiral_strength * ry * np.cos(3 * t_ctrl)

    # Heights: random base values + two sinusoidal oscillations for roller-coaster effect.
    base_heights = rng.uniform(height_range[0], height_range[1], num_control)
    height_amp1 = (height_range[1] - height_range[0]) * 0.3
    height_amp2 = (height_range[1] - height_range[0]) * 0.15
    height_freq1 = rng.uniform(1.0, 3.0)
    height_freq2 = rng.uniform(4.0, 6.0)
    ctrl_z = np.clip(
        base_heights
        + height_amp1 * np.sin(height_freq1 * t_ctrl + rng.uniform(0, 2 * np.pi))
        + height_amp2 * np.sin(height_freq2 * t_ctrl + rng.uniform(0, 2 * np.pi)),
        height_range[0], height_range[1],
    )

    # Append first point to close the loop; bc_type='periodic' enforces C2 continuity
    # at the seam so the track is smooth where the start and end meet.
    from scipy.interpolate import CubicSpline
    t_ext   = np.append(t_ctrl, 2 * np.pi)
    ctrl_x  = np.append(ctrl_x,  ctrl_x[0])
    ctrl_y  = np.append(ctrl_y,  ctrl_y[0])
    ctrl_z  = np.append(ctrl_z,  ctrl_z[0])

    cs_x = CubicSpline(t_ext, ctrl_x, bc_type='periodic')
    cs_y = CubicSpline(t_ext, ctrl_y, bc_type='periodic')
    cs_z = CubicSpline(t_ext, ctrl_z, bc_type='periodic')

    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cs_x(t)
    y = cs_y(t)
    z = np.clip(cs_z(t), height_range[0], height_range[1])

    waypoints = np.stack([x, y, z], axis=1)

    # Analytical first derivative → exact tangent direction (no finite-difference error).
    tangents = np.stack([cs_x(t, 1), cs_y(t, 1), cs_z(t, 1)], axis=1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8

    return waypoints, tangents


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

    return np.array(gate_positions), np.array(gate_orientations)


# ---------------------------------------------------------------------------
# Gate mesh construction
# ---------------------------------------------------------------------------

def _create_gate_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    gate_size: float,
    bar_thickness: float,
    color: tuple[float, float, float] | None = None,
) -> list[trimesh.Trimesh]:
    """Build a square gate frame from 4 axis-aligned box bars.

    The gate opening is ``gate_size × gate_size`` in the gate's XZ plane (right × up).
    Bars are extended by bar_thickness at each end so adjacent bars overlap at corners,
    avoiding visible gaps.

    Color is set AFTER apply_transform.  Setting face_colors before transform can be
    silently ignored if trimesh is still using TextureVisuals internally.
    """
    half = gate_size / 2.0

    # Convert (w, x, y, z) → scipy's (x, y, z, w) for Rotation.
    quat_scipy = [orientation[1], orientation[2], orientation[3], orientation[0]]
    rot_mat = tf.Rotation.from_quat(quat_scipy).as_matrix()

    # Each bar: (local_centre_offset_in_gate_frame, box_dimensions_in_gate_frame)
    # Horizontal bars (top/bottom) are wider in X; vertical bars (left/right) are taller in Z.
    bar_specs = [
        ("top",    np.array([0,    0, +half]), np.array([gate_size + bar_thickness, bar_thickness, bar_thickness])),
        ("bottom", np.array([0,    0, -half]), np.array([gate_size + bar_thickness, bar_thickness, bar_thickness])),
        ("left",   np.array([-half, 0, 0]),   np.array([bar_thickness, bar_thickness, gate_size + bar_thickness])),
        ("right",  np.array([+half, 0, 0]),   np.array([bar_thickness, bar_thickness, gate_size + bar_thickness])),
    ]

    meshes = []
    for _name, local_offset, dims in bar_specs:
        bar = trimesh.creation.box(dims)
        world_pos = position + rot_mat @ local_offset
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = world_pos
        bar.apply_transform(transform)
        if color is not None:
            # Assign after transform so ColorVisuals is set on the final mesh state.
            bar.visual.face_colors = [int(c * 255) for c in color] + [255]
        meshes.append(bar)

    return meshes




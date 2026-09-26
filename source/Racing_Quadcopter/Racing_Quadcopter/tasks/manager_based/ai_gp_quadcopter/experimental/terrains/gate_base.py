# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared data structures and mesh utilities for racing gate terrains.

Both ``racing_gates`` (procedural) and ``fixed_gates`` (deterministic) are
peer terrain generators that share the same gate representation, registry, and
mesh-building helper defined here.

Gate coordinate system
----------------------
  +X  right  (horizontal, along gate bar width)
  +Y  forward (passage direction — the drone should fly through in +Y)
  +Z  up     (gates are always upright, Z always = world up)

All quaternions are stored in (w, x, y, z) order to match IsaacLab convention.
scipy.spatial.transform uses (x, y, z, w); conversions are explicit at every
call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.spatial.transform as tf
import trimesh


# ---------------------------------------------------------------------------
# Gate data
# ---------------------------------------------------------------------------

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
    bar_thickness: float        # frame bar thickness in metres
    start_position: np.ndarray  # [3]  first waypoint — informational
    start_tangent: np.ndarray   # [3]  tangent at first waypoint — informational
    waypoints: np.ndarray       # [num_waypoints, 3]  trajectory sample points, local frame


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------
# Dict from (difficulty, seed) → GateData for every cell generated in the
# current run.  The env reads this after TerrainGenerator finishes.

gate_registry: dict[tuple[float, int | None], GateData] = {}


def get_gate_registry() -> dict[tuple[float, int | None], GateData]:
    """Return a reference to the module-level gate data registry."""
    return gate_registry


def clear_gate_registry() -> None:
    """Clear all stored gate data (call between training runs if needed)."""
    gate_registry.clear()


# ---------------------------------------------------------------------------
# GateSpec → GateData conversion
# ---------------------------------------------------------------------------

def gate_specs_to_gate_data(
    specs: list,
    gate_size: float,
    bar_thickness: float,
) -> GateData:
    """Convert a list of :class:`GateSpec` into a :class:`GateData` instance.

    Shared by ``fixed_gates.fixed_gates_terrain`` (terrain generation) and the
    multi-track env's ``_load_cell_gate_data`` (runtime data loading).

    Yaw is negated when converting to a scipy ``Rotation`` because ``GateSpec``
    uses a CW-from-+Y convention while scipy uses CCW-around-Z.  Quaternions are
    stored as wxyz to match IsaacLab convention (scipy returns xyzw).
    """
    positions_list: list[np.ndarray] = []
    orientations_list: list[np.ndarray] = []
    for spec in specs:
        pos = np.array(spec.position, dtype=float)
        rot = tf.Rotation.from_euler(
            "xyz", [spec.rotation[0], spec.rotation[1], -spec.rotation[2]], degrees=True
        )
        q = rot.as_quat()  # xyzw
        positions_list.append(pos)
        orientations_list.append(np.array([q[3], q[0], q[1], q[2]]))  # → wxyz

    positions_arr = np.array(positions_list)
    orientations_arr = np.array(orientations_list)
    first_rot = tf.Rotation.from_quat(
        [orientations_arr[0, 1], orientations_arr[0, 2],
         orientations_arr[0, 3], orientations_arr[0, 0]]
    )
    return GateData(
        positions=positions_arr,
        orientations=orientations_arr,
        gate_size=gate_size,
        bar_thickness=bar_thickness,
        start_position=positions_arr[0].copy(),
        start_tangent=first_rot.apply([0.0, 1.0, 0.0]),
        waypoints=positions_arr.copy(),
    )


# ---------------------------------------------------------------------------
# Gate mesh construction
# ---------------------------------------------------------------------------

def create_gate_frame(
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
    bar_specs = [
        ("top",    np.array([0,    0, +half]), np.array([gate_size + bar_thickness, bar_thickness, bar_thickness])),
        ("bottom", np.array([0,    0, -half]), np.array([gate_size + bar_thickness, bar_thickness, bar_thickness])),
        ("left",   np.array([-half, 0, 0]),   np.array([bar_thickness, bar_thickness, gate_size + bar_thickness])),
        ("right",  np.array([+half, 0, 0]),   np.array([bar_thickness, bar_thickness, gate_size + bar_thickness])),
    ]

    meshes = []
    for _, local_offset, dims in bar_specs:
        bar = trimesh.creation.box(dims)
        world_pos = position + rot_mat @ local_offset
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = world_pos
        bar.apply_transform(transform)
        if color is not None:
            bar.visual.face_colors = [int(c * 255) for c in color] + [255]
        meshes.append(bar)

    return meshes

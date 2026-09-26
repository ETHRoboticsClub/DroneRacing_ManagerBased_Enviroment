# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-layout racing gate terrain — no randomisation.

Entry point: ``fixed_gates_terrain(difficulty, cfg)``
  Called by TerrainGenerator once per cell.  All cells in the grid receive the
  same layout because positions and headings are taken directly from ``cfg.gates``.
  The ``difficulty`` argument is accepted but ignored.

Gate coordinate conventions (shared with racing_gates.py)
----------------------------------------------------------
  +X  right  (horizontal, along gate bar width)
  +Y  forward (passage direction — drone should fly through in +Y)
  +Z  up

``GateSpec.yaw_deg`` follows a **clockwise-from-+Y** convention when viewed from
above: 0° → drone passes in the +Y direction; 90° → passes in +X.
All quaternions are (w, x, y, z) to match IsaacLab convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.spatial.transform as tf
import trimesh

from isaaclab.terrains.trimesh.utils import make_plane

from .gate_base import GateData, gate_registry, create_gate_frame

if TYPE_CHECKING:
    from .racing_gates_cfg import MeshFixedGatesTerrainCfg


def fixed_gates_terrain(
    difficulty: float, cfg: "MeshFixedGatesTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat terrain with gates at exactly the positions defined in cfg.gates.

    Args:
        difficulty: Ignored; accepted only because TerrainGenerator passes it to
                    every terrain function.
        cfg:        ``MeshFixedGatesTerrainCfg`` with a populated ``gates`` list.

    Returns:
        meshes:  Ground plane + gate bar meshes for this cell.
        origin:  Cell centre at Z=0 for TerrainGenerator world placement.
    """
    if not cfg.gates:
        raise ValueError("MeshFixedGatesTerrainCfg.gates must not be empty.")

    meshes_list: list[trimesh.Trimesh] = [make_plane(cfg.size, height=0.0, center_zero=False)]

    frame_fn = cfg.gate_frame_fn if cfg.gate_frame_fn is not None else create_gate_frame

    positions_list = []
    orientations_list = []

    for spec in cfg.gates:
        pos = np.array(spec.position, dtype=float)

        # Extrinsic XYZ (= RPY) Euler. Yaw is CW-from-+Y by convention (0°→+Y,
        # 90°→+X), while scipy uses CCW around Z — negate yaw to match.
        rot = tf.Rotation.from_euler(
            "xyz",
            [spec.roll_deg, spec.pitch_deg, -spec.yaw_deg],
            degrees=True,
        )
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        positions_list.append(pos)
        orientations_list.append(quat_wxyz)
        meshes_list.extend(
            frame_fn(pos, quat_wxyz, cfg.gate_size, cfg.bar_thickness, cfg.gate_color)
        )

    positions_arr = np.array(positions_list)
    orientations_arr = np.array(orientations_list)

    # Forward direction of gate 0 (gate-local +Y rotated to world frame).
    first_rot = tf.Rotation.from_quat(
        [orientations_arr[0, 1], orientations_arr[0, 2], orientations_arr[0, 3], orientations_arr[0, 0]]
    )
    start_tangent = first_rot.apply([0.0, 1.0, 0.0])

    gate_registry[(round(difficulty, 6), cfg.seed)] = GateData(
        positions=positions_arr,
        orientations=orientations_arr,
        gate_size=cfg.gate_size,
        bar_thickness=cfg.bar_thickness,
        start_position=positions_arr[0].copy(),
        start_tangent=start_tangent,
        waypoints=positions_arr.copy(),  # gate centres used as trajectory waypoints for vis
    )

    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return meshes_list, origin

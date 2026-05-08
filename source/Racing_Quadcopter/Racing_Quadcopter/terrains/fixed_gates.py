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
import trimesh

from isaaclab.terrains.trimesh.utils import make_plane

from .gate_base import create_gate_frame, gate_registry, gate_specs_to_gate_data

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

    gd = gate_specs_to_gate_data(cfg.gates, cfg.gate_size, cfg.bar_thickness)

    meshes_list: list[trimesh.Trimesh] = [make_plane(cfg.size, height=0.0, center_zero=False)]
    for pos, quat_wxyz in zip(gd.positions, gd.orientations):
        meshes_list.extend(create_gate_frame(pos, quat_wxyz, cfg.gate_size, cfg.bar_thickness, cfg.gate_color))

    gate_registry[(round(difficulty, 6), cfg.seed)] = gd

    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    return meshes_list, origin

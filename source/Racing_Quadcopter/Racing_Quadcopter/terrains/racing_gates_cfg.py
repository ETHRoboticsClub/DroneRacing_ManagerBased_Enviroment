# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for racing-gate mesh sub-terrains (extension-local; not in stock Isaac Lab)."""

from dataclasses import MISSING

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import racing_gates


@configclass
class MeshRacingGatesTerrainCfg(SubTerrainBaseCfg):
    """Flat ground with racing gates along a procedural trajectory."""

    function = racing_gates.racing_gates_terrain

    trajectory_type: str = "figure8"
    """``figure8``, ``oval``, or ``random``."""

    num_waypoints: int = 100

    seed: int | None = None

    gate_spacing: float = 4.0

    gate_spacing_range: tuple[float, float] | None = None

    min_gates: int = 3

    gate_size_range: tuple[float, float] = (0.4, 0.8)

    gate_height_range: tuple[float, float] = (1.0, 2.5)

    bar_thickness: float = 0.05

    gate_color: tuple[float, float, float] = (1.0, 0.3, 0.0)

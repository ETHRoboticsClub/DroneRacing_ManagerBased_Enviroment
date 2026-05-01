# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing-gate mesh terrain (vendored for stock Isaac Lab; upstream has no ``isaaclab.terrains.config.racing``)."""

from .gate_base import GateData, clear_gate_registry, get_gate_registry
from .racing_gates_cfg import GateSpec, MeshFixedGatesTerrainCfg, MeshRacingGatesTerrainCfg
from .racing_terrain import create_fixed_track_terrain, create_racing_gates_terrain
from .track_io import load_gates_from_yaml

__all__ = [
    "GateData",
    "GateSpec",
    "MeshFixedGatesTerrainCfg",
    "MeshRacingGatesTerrainCfg",
    "clear_gate_registry",
    "create_fixed_track_terrain",
    "create_racing_gates_terrain",
    "get_gate_registry",
    "load_gates_from_yaml",
]

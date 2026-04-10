# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing-gate mesh terrain (vendored for stock Isaac Lab; upstream has no ``isaaclab.terrains.config.racing``)."""

from .racing_gates import GateData, clear_gate_registry, get_gate_registry
from .racing_gates_cfg import MeshRacingGatesTerrainCfg
from .racing_terrain import create_racing_gates_terrain

__all__ = [
    "GateData",
    "MeshRacingGatesTerrainCfg",
    "clear_gate_registry",
    "create_racing_gates_terrain",
    "get_gate_registry",
]

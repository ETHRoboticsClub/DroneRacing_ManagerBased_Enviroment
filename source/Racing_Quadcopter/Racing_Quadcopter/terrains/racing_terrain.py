# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generator presets for racing gates."""

from isaaclab.terrains import TerrainGeneratorCfg

from .racing_gates_cfg import MeshRacingGatesTerrainCfg

__all__ = ["create_racing_gates_terrain"]


def create_racing_gates_terrain(seed: int = 42) -> TerrainGeneratorCfg:
    """Build a :class:`~isaaclab.terrains.TerrainGeneratorCfg` with random racing-gate tracks."""
    return TerrainGeneratorCfg(
        seed=seed,
        size=(15.0, 15.0),
        difficulty_range=(0.7, 1.0),
        border_width=0.5,
        num_rows=4,
        num_cols=4,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "racing_gates": MeshRacingGatesTerrainCfg(
                proportion=1.0,
                seed=seed,
                trajectory_type="random",
                num_waypoints=150,
                gate_spacing_range=(2.5, 10.0),
                gate_size_range=(0.7, 0.7001),
                gate_height_range=(0.8, 3.5),
                bar_thickness=0.05,
                gate_color=(1.0, 0.3, 0.0),
            ),
        },
    )

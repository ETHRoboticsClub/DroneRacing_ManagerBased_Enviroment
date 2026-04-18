# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generator presets for racing gates."""

from isaaclab.terrains import TerrainGeneratorCfg

from .racing_gates_cfg import MeshRacingGatesTerrainCfg

__all__ = ["create_racing_gates_terrain"]


def create_racing_gates_terrain(seed: int = 42) -> TerrainGeneratorCfg:
    """Return a TerrainGeneratorCfg that tiles 16 racing-gate circuits in a 4×4 grid.

    Called once during env setup (RacingQuadcopterEnvCfg.__post_init__).  The
    returned cfg is passed to TerrainImporterCfg, which creates one USD sub-terrain
    mesh per cell and stores their world-frame origins in terrain.terrain_origins.

    Grid layout:
        4 rows × 4 cols = 16 cells, each 15×15 m.
        Rows encode difficulty (0.7–1.0); all cells in a row share the same difficulty.
        With num_envs=1024, each cell hosts 64 simultaneous drone environments.

    Seed per cell:
        TerrainGenerator differentiates the seed for each cell before invoking the
        terrain function, so all 16 cells produce unique gate layouts and the registry
        contains 16 distinct entries.  Difficulty is assigned per row (4 levels across
        4 rows); columns within a row share the same difficulty but a different seed.

    curriculum=False:
        All 16 cells are available from episode 0. Isaac Lab's terrain curriculum
        (which progressively unlocks harder rows) is disabled; the env uses its own
        curriculum logic (collision gating) instead.
    """
    return TerrainGeneratorCfg(
        seed=seed,
        # Each cell occupies a 15×15 m patch in the world.
        size=(15.0, 15.0),
        # Difficulty 0.7–1.0: all cells are on the harder end (complex tracks,
        # smaller gates). Difficulty 0.7 → largest gates; 1.0 → smallest gates.
        difficulty_range=(0.7, 1.0),
        # 0.5 m border kept collision-free around each cell edge.
        border_width=0.5,
        # 4×4 grid = 16 terrain tiles. With num_envs=1024: 64 drones per tile.
        # If num_envs < 16 only the first num_envs cells receive drones (the rest
        # are static geometry with no agents).
        num_rows=4,
        num_cols=4,
        # Mesh resolution parameters; used for height-map generation (ground plane).
        # These have little effect on flat racing terrain but must be set for the
        # TerrainGenerator pipeline.
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        # Always regenerate — do not cache to disk. Caching would prevent changes
        # to trajectory/gate parameters from taking effect on the next run.
        use_cache=False,
        # See note above: curriculum is disabled; the env handles its own progression.
        curriculum=False,
        sub_terrains={
            "racing_gates": MeshRacingGatesTerrainCfg(
                proportion=1.0,         # 100 % of cells use this terrain type
                seed=seed,              # Overwritten to seed=42 by TerrainGenerator anyway
                trajectory_type="random",  # Cubic-spline randomised circuit
                num_waypoints=150,      # Fine arc-length resolution for 15×15 m tiles
                # Spacing 2.5–10 m → avg 6.25 m → ~8–12 gates per circuit on a
                # typical 15×15 m tile arc (~60–80 m total arc length).
                gate_spacing_range=(2.5, 10.0),
                gate_size_range=(0.7, 0.7001),
                # Gates can appear anywhere from 0.8 m (low slalom) to 3.5 m (high
                # climb). Difficulty controls oscillation amplitude within this range.
                gate_height_range=(0.8, 3.5),
                bar_thickness=0.05,
                gate_color=(1.0, 0.3, 0.0),  # Orange, high-contrast against sky/ground
            ),
        },
    )

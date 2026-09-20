# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generator presets for racing gates."""

from __future__ import annotations

from isaaclab.terrains import TerrainGeneratorCfg

from .racing_gates_cfg import GateSpec, MeshFixedGatesTerrainCfg, MeshRacingGatesTerrainCfg
from .track_io import load_gates_from_yaml

__all__ = [
    "create_fixed_track_terrain",
    "create_multi_track_terrain",
    "create_racing_gates_terrain",
]


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
                trajectory_type="random",  # randomly picks spline or chicane per cell
                num_waypoints=200,      # finer resolution for tighter curves
                # Spacing 3.0–7 m → avg 5.0 m → ~10–16 gates per circuit.
                gate_spacing_range=(7.5, 15.0),
                gate_size_range=(0.7, 1.5),
                # 0.7–4.5 m: meaningful height variation with dramatic climbs at high difficulty.
                gate_height_range=(0.7, 4.5),
                bar_thickness=0.05,
                gate_color=(1.0, 0.3, 0.0),  # Orange, high-contrast against sky/ground (DOESN'T WORK)
            ),
        },
    )


def create_fixed_track_terrain(
    gates: list[GateSpec],
    gate_size: float = 0.8,
    bar_thickness: float = 0.05,
    gate_color: tuple[float, float, float] = (1.0, 0.3, 0.0),
    terrain_size: tuple[float, float] = (15.0, 15.0),
    num_rows: int = 4,
    num_cols: int = 4,
    seed: int = 42,
) -> TerrainGeneratorCfg:
    """Return a TerrainGeneratorCfg that tiles the same fixed circuit in a grid.

    All cells in the grid receive an identical copy of the circuit — no
    randomisation is applied.  Gate positions are in the sub-terrain's
    corner-origin local frame (see ``GateSpec`` for full conventions).

    Example usage::

        from Racing_Quadcopter.terrains import GateSpec, create_fixed_track_terrain

        terrain = create_fixed_track_terrain(
            gates=[
                GateSpec(position=[7.5, 4.0, 1.5], rotation=[0.0, 0.0, 0.0]),
                GateSpec(position=[11.0, 7.5, 2.0], rotation=[0.0, 0.0, 90.0]),
                GateSpec(position=[7.5, 11.0, 1.5], rotation=[0.0, 0.0, 180.0]),
                GateSpec(position=[4.0, 7.5, 2.0], rotation=[0.0, 0.0, 270.0]),
            ],
            gate_size=0.8,
        )

    Args:
        gates:         Ordered list of ``GateSpec`` — one entry per gate on the
                       circuit.  The drone must pass them in order; the last gate
                       connects back to the first to close the loop.
        gate_size:     Square aperture side length in metres.
        bar_thickness: Gate bar diameter in metres (visual only).
        gate_color:    RGB in 0–1 range.
        terrain_size:  (width, height) of each terrain cell in metres.
        num_rows:      Terrain grid rows (all rows share the same fixed circuit).
        num_cols:      Terrain grid columns.
        seed:          Base seed passed to TerrainGenerator.

    Returns:
        ``TerrainGeneratorCfg`` ready to pass to ``TerrainImporterCfg``.
    """
    return TerrainGeneratorCfg(
        seed=seed,
        size=terrain_size,
        difficulty_range=(0.0, 0.0),  # ignored by fixed_gates_terrain
        border_width=0.5,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "racing_gates": MeshFixedGatesTerrainCfg(
                proportion=1.0,
                seed=seed,
                gates=gates,
                gate_size=gate_size,
                bar_thickness=bar_thickness,
                gate_color=gate_color,
            ),
        },
    )


def create_multi_track_terrain(
    track_yaml_paths: list[str],
    *,
    gate_size: float = 0.8,
    bar_thickness: float = 0.05,
    gate_color: tuple[float, float, float] = (1.0, 0.3, 0.0),
    terrain_size: tuple[float, float] = (15.0, 15.0),
    border_width: float = 0.5,
    seed: int = 42,
) -> TerrainGeneratorCfg:
    """Return a TerrainGeneratorCfg laying out ``N = len(track_yaml_paths)`` fixed
    circuits one per column, using IsaacLab's curriculum layout so the per-column
    assignment is deterministic.

    With ``curriculum=True``, ``num_rows=1`` and N columns each holding a different
    sub-terrain (proportion ``1/N``), TerrainGenerator builds column ``j`` from
    ``track_yaml_paths[j]``. ``TerrainImporter`` then evenly partitions ``num_envs``
    across columns (``floor(env_id / (num_envs / N))``), so all envs in the same
    column share the same fixed track.

    The collapsed ``difficulty_range=(0.5, 0.5)`` removes the per-cell η perturbation
    (``fixed_gates_terrain`` ignores difficulty anyway, but keeping the range tight
    avoids accidental noise if other parameters ever start to depend on it).

    Args:
        track_yaml_paths: paths to per-column track YAMLs (length = N).
        gate_size:        square aperture side length in metres.
        bar_thickness:    gate bar diameter in metres (visual only).
        gate_color:       RGB in 0–1 range.
        terrain_size:     (width, height) of each terrain cell in metres.
        border_width:     border kept collision-free around each cell edge.
        seed:             base seed (TerrainGenerator differentiates per cell).

    Returns:
        ``TerrainGeneratorCfg`` ready to pass to ``TerrainImporterCfg``.
    """
    if not track_yaml_paths:
        raise ValueError("track_yaml_paths must not be empty.")

    n = len(track_yaml_paths)
    proportion = 1.0 / n
    sub_terrains: dict[str, MeshFixedGatesTerrainCfg] = {}
    for i, yaml_path in enumerate(track_yaml_paths):
        gates = load_gates_from_yaml(yaml_path)
        # seed must differ per cfg so the gate registry key (difficulty, seed)
        # is unique even before TerrainGenerator's per-cell seed differentiation.
        sub_terrains[f"course_{i:03d}"] = MeshFixedGatesTerrainCfg(
            proportion=proportion,
            seed=seed + i,
            gates=gates,
            gate_size=gate_size,
            bar_thickness=bar_thickness,
            gate_color=gate_color,
        )

    return TerrainGeneratorCfg(
        seed=seed,
        size=terrain_size,
        difficulty_range=(0.5, 0.5),  # collapsed; fixed_gates_terrain ignores it anyway
        border_width=border_width,
        num_rows=1,
        num_cols=n,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub_terrains,
    )

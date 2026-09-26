# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for racing-gate mesh sub-terrains (extension-local; not in stock Isaac Lab)."""

from dataclasses import MISSING, dataclass, field

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import fixed_gates, racing_gates


@dataclass
class GateSpec:
    """Specification for a single gate on a fixed-layout track.

    Coordinates are in the sub-terrain's **corner-origin local frame**: (0, 0) is
    the bottom-left corner of the tile; a 15×15 m tile runs to (size_x, size_y).

    Orientation uses extrinsic ``[roll, pitch, yaw]`` Euler angles (RPY, degrees) —
    the conventional ROS/URDF order. ``rotation[2]`` (yaw) follows a
    **clockwise-from-+Y** convention when viewed from above:
      0°  → drone passes through in the +Y direction (tile's forward axis)
      90° → drone passes through in the +X direction (tile's right axis)
    ``rotation[0]`` (roll) rotates about the gate's +X bar axis and tilts the passage
    axis up/down (roll 90° lays the gate flat: fly through vertically);
    ``rotation[1]`` (pitch) rotates about the +Y passage axis and banks the gate.
    See ``gate_base.gate_specs_to_gate_data`` (scipy extrinsic ``"xyz"``).
    """

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.5])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [roll, pitch, yaw] in degrees


@configclass
class MeshRacingGatesTerrainCfg(SubTerrainBaseCfg):
    """Flat ground with racing gates placed along a procedural closed-loop trajectory.

    TerrainGenerator calls ``function(difficulty, cfg)`` once per terrain cell and
    differentiates the seed per cell, so each of the 16 cells in a 4×4 grid receives
    a unique random gate layout.  Difficulty is assigned per row (4 rows → 4 levels);
    columns within a row share the same difficulty but differ in seed.

    Field reference
    ---------------
    trajectory_type     : "figure8" | "oval" | "random" (cubic-spline, most varied)
    num_waypoints       : sample density along the trajectory (100–200 for 15×15 m)
    seed                : overwritten by TerrainGenerator per cell — set here has no effect
    gate_spacing_range  : (min, max) metres between gates; use (n, n) for fixed spacing.
                          Gate count = int(total_arc / avg_spacing); no min/max guard.
    gate_size_range     : aperture in metres, interpolated by difficulty.
                          difficulty=0 → range[1] (large/easy), difficulty=1 → range[0] (small/hard).
                          Formula: size = range[1] - difficulty * (range[1] - range[0])
    gate_height_range   : (min, max) metres above ground for gate centres.
                          Difficulty controls oscillation amplitude within this band.
    bar_thickness       : gate bar diameter in metres (visual only, no physics).
    gate_color          : RGB in 0–1 range; face_colors applied after apply_transform.
    """

    function = racing_gates.racing_gates_terrain

    trajectory_type: str = "figure8"
    num_waypoints: int = 100
    seed: int | None = None
    gate_spacing_range: tuple[float, float] = (3.0, 5.0)
    gate_size_range: tuple[float, float] = (0.4, 0.8)
    gate_height_range: tuple[float, float] = (1.0, 2.5)
    bar_thickness: float = 0.05
    gate_color: tuple[float, float, float] = (1.0, 0.3, 0.0)


@configclass
class MeshFixedGatesTerrainCfg(SubTerrainBaseCfg):
    """Flat ground with racing gates at fixed, user-specified positions.

    Unlike ``MeshRacingGatesTerrainCfg`` no trajectory is generated and no
    randomisation is applied — every cell in the grid receives exactly the same
    circuit defined by ``gates``.

    Field reference
    ---------------
    gates        : List of ``GateSpec``.  Each entry sets one gate's position
                   (in the sub-terrain's corner-origin local frame) and heading
                   (yaw_deg, CW from +Y viewed from above; 0° → +Y, 90° → +X).
    gate_size    : Square aperture side length in metres.
    bar_thickness: Gate bar diameter in metres (visual only).
    gate_color   : RGB in 0–1 range.
    """

    function = fixed_gates.fixed_gates_terrain

    gates: list = MISSING        # list[GateSpec], provided by the caller
    seed: int | None = None
    gate_size: float = 0.8
    bar_thickness: float = 0.05
    gate_color: tuple[float, float, float] = (1.0, 0.3, 0.0)
    # Optional custom mesh builder with the same signature as
    # ``gate_base.create_gate_frame(position, orientation, gate_size, bar_thickness, color)``.
    # None -> default square frame centred on the aperture edges.
    gate_frame_fn: object | None = None

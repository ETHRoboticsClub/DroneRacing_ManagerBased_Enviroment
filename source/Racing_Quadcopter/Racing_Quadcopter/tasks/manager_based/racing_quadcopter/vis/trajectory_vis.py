# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD visualisation utilities: trajectory, gate direction indicators, and gate labels.

All prims created here are UsdGeom.BasisCurves — they carry NO UsdPhysics.CollisionAPI.
PhysX is opt-in in USD: a prim participates in collision only if CollisionAPI is
explicitly applied to it.  BasisCurves.Define() never does this, so these prims are
invisible to the physics engine.  They also live under /World/TrajectoryVis,
/World/GateIndicatorVis, /World/GateLabelVis — completely separate from the terrain
mesh at /World/ground, so the TerrainImporter cannot affect them.
Written once at env startup, zero per-frame CPU/GPU cost.

Entry points:
    draw_trajectory_usd()       — green closed-loop polyline along trajectory waypoints
    draw_gate_indicators_usd()  — blue sticks showing gate forward (+Y) direction
    draw_gate_labels_usd()      — blue 7-segment digit labels floating above each gate
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 7-segment geometry — line endpoint pairs in digit-local 2D (X=right, Y=up).
# Horizontal bars span ±_SL/2 in X; vertical bars span ±_SL/2 in Y.
# These map directly to BasisCurves 2-point segments (no box mesh needed).
# ---------------------------------------------------------------------------
_SL = 0.08          # segment long dimension in metres
_HL = _SL / 2.0     # half-length = 0.04 m
_SEVEN_SEG: dict[str, str] = {
    "0": "abcdef", "1": "bc",    "2": "abdeg", "3": "abcdg",
    "4": "bcfg",   "5": "acdfg", "6": "acdefg","7": "abc",
    "8": "abcdefg","9": "abcdfg",
}
# (start_xy, end_xy) for each segment — constant, independent of gate
_SEG_LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "a": ((-_HL,  0.10), ( _HL,  0.10)),   # top horizontal
    "b": (( 0.05,  0.01), ( 0.05,  0.09)), # top-right vertical
    "c": (( 0.05, -0.09), ( 0.05, -0.01)), # bottom-right vertical
    "d": ((-_HL, -0.10), ( _HL, -0.10)),   # bottom horizontal
    "e": ((-0.05, -0.09), (-0.05, -0.01)), # bottom-left vertical
    "f": ((-0.05,  0.01), (-0.05,  0.09)), # top-left vertical
    "g": ((-_HL,  0.00), ( _HL,  0.00)),   # middle horizontal
}
_CHAR_SPACING = 0.13  # metres between digit centres


def draw_trajectory_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    width: float = 0.015,
    color: tuple[float, float, float] = (0.0, 1.0, 0.2),
    root_path: str = "/World/TrajectoryVis",
) -> None:
    """Create one UsdGeom.BasisCurves prim per terrain cell to visualise the circuit.

    Args:
        gate_datas:         Ordered list of GateData objects (registry insertion order).
        terrain_origins_np: [num_rows, num_cols, 3] world-frame cell centres (numpy).
        centre_offset:      [3] shift from corner-origin to centre-origin (−size/2 in X,Y).
        num_rows:           Number of terrain rows.
        num_cols:           Number of terrain columns.
        width:              Tube diameter in metres for the rendered curve.
        color:              RGB display colour in 0–1 range.
        root_path:          USD prim path prefix; one child prim per cell is created under it.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)
    usd_color = Vt.Vec3fArray([Gf.Vec3f(*color)])

    for row in range(num_rows):
        for col in range(num_cols):
            cell_idx = (row * num_cols + col) % n_cells
            gd = gate_datas[cell_idx]

            pts_local = gd.waypoints + centre_offset  # corner-origin → centre-origin
            origin = terrain_origins_np[row, col]     # world-frame cell centre
            pts_world = pts_local + origin            # [W, 3] world frame

            prim_path = f"{root_path}/cell_{row}_{col}"
            curves = UsdGeom.BasisCurves.Define(stage, prim_path)
            curves.CreateTypeAttr("linear")
            curves.CreateWrapAttr("periodic")  # closes the loop back to waypoints[0]
            curves.CreatePointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts_world])
            )
            curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(pts_world)]))
            curves.CreateWidthsAttr().Set(Vt.FloatArray([width] * len(pts_world)))
            curves.CreateDisplayColorAttr().Set(usd_color)


def draw_gate_indicators_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    length: float = 0.5,
    width: float = 0.04,
    color: tuple[float, float, float] = (0.1, 0.4, 1.0),
    root_path: str = "/World/GateIndicatorVis",
) -> None:
    """Create one UsdGeom.BasisCurves prim per terrain cell showing gate forward directions.

    Each gate contributes a 2-point segment: from gate centre to gate centre + forward * length.
    All segments for a cell share one BasisCurves prim (efficient — one USD prim per cell).

    Gate forward is local +Y rotated by the gate quaternion (wxyz).  The rotation is
    computed analytically from the quaternion elements without any external library.

    Args:
        gate_datas:         Ordered list of GateData objects (registry insertion order).
        terrain_origins_np: [num_rows, num_cols, 3] world-frame cell centres (numpy).
        centre_offset:      [3] shift from corner-origin to centre-origin (−size/2 in X,Y).
        num_rows:           Number of terrain rows.
        num_cols:           Number of terrain columns.
        length:             Arrow length in metres.
        width:              Tube diameter in metres.
        color:              RGB display colour in 0–1 range.
        root_path:          USD prim path prefix; one child prim per cell is created under it.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)
    usd_color = Vt.Vec3fArray([Gf.Vec3f(*color)])

    for row in range(num_rows):
        for col in range(num_cols):
            cell_idx = (row * num_cols + col) % n_cells
            gd = gate_datas[cell_idx]
            origin = terrain_origins_np[row, col]

            points: list[Gf.Vec3f] = []
            counts: list[int] = []

            for pos_local, q in zip(gd.positions, gd.orientations):
                pos_world = pos_local + centre_offset + origin

                # Rotate local +Y through quaternion (w,x,y,z) → world-frame forward.
                # R * [0,1,0] = second column of rotation matrix derived from q:
                #   fwd_x = 2*(qx*qy - qw*qz)
                #   fwd_y = 1 - 2*(qx² + qz²)
                #   fwd_z = 2*(qy*qz + qw*qx)
                qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                fwd = np.array([
                    2.0 * (qx * qy - qw * qz),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz + qw * qx),
                ])

                end_world = pos_world + fwd * length
                points.append(Gf.Vec3f(float(pos_world[0]), float(pos_world[1]), float(pos_world[2])))
                points.append(Gf.Vec3f(float(end_world[0]), float(end_world[1]), float(end_world[2])))
                counts.append(2)

            prim_path = f"{root_path}/cell_{row}_{col}"
            curves = UsdGeom.BasisCurves.Define(stage, prim_path)
            curves.CreateTypeAttr("linear")
            curves.CreateWrapAttr("nonperiodic")
            curves.CreatePointsAttr().Set(Vt.Vec3fArray(points))
            curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(counts))
            curves.CreateWidthsAttr().Set(Vt.FloatArray([width] * len(points)))
            curves.CreateDisplayColorAttr().Set(usd_color)


def draw_gate_labels_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    scale: float = 2.0,
    width: float = 0.012,
    color: tuple[float, float, float] = (0.1, 0.4, 1.0),
    root_path: str = "/World/GateLabelVis",
) -> None:
    """Create one UsdGeom.BasisCurves prim per terrain cell with 7-segment gate index labels.

    Each gate's number is drawn as BasisCurves line segments (one 2-point segment per
    active bar of the 7-segment display) floating gate_size/2 + 0.30 m above the gate
    centre, lying flat in the world XY plane so labels are readable from above.

    Digit geometry is defined in digit-local 2D (X = gate right, Y = gate forward) and
    transformed analytically to world space using the gate quaternion — no scipy needed.

    Args:
        gate_datas:         Ordered list of GateData objects (registry insertion order).
        terrain_origins_np: [num_rows, num_cols, 3] world-frame cell centres (numpy).
        centre_offset:      [3] shift from corner-origin to centre-origin (−size/2 in X,Y).
        num_rows:           Number of terrain rows.
        num_cols:           Number of terrain columns.
        width:              Line tube diameter in metres.
        color:              RGB display colour in 0–1 range.
        root_path:          USD prim path prefix; one child prim per cell is created under it.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)
    usd_color = Vt.Vec3fArray([Gf.Vec3f(*color)])

    for row in range(num_rows):
        for col in range(num_cols):
            cell_idx = (row * num_cols + col) % n_cells
            gd = gate_datas[cell_idx]
            origin = terrain_origins_np[row, col]

            pts: list[Gf.Vec3f] = []
            cnts: list[int] = []

            for gate_idx, (pos_local, q) in enumerate(zip(gd.positions, gd.orientations)):
                pos_world = pos_local + centre_offset + origin

                # Gate right and forward in world frame from quaternion (w,x,y,z).
                # R @ [1,0,0] = first column; R @ [0,1,0] = second column.
                # Gates are always upright so Z components are zero by construction.
                qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                right = np.array([1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy + qw*qz), 0.0])
                right /= np.linalg.norm(right) + 1e-8
                fwd = np.array([2.0*(qx*qy - qw*qz), 1.0 - 2.0*(qx*qx + qz*qz), 0.0])
                fwd /= np.linalg.norm(fwd) + 1e-8

                # Label floats above the gate top bar, lying flat in the XY plane.
                label_origin = np.array([pos_world[0], pos_world[1],
                                         pos_world[2] + gd.gate_size / 2.0 + 0.30])

                digit_str = str(gate_idx)
                n_chars = len(digit_str)
                for char_idx, ch in enumerate(digit_str):
                    x_char = (char_idx - (n_chars - 1) / 2.0) * _CHAR_SPACING * scale
                    active = _SEVEN_SEG.get(ch, "")
                    for seg_id, (p0_2d, p1_2d) in _SEG_LINES.items():
                        if seg_id not in active:
                            continue
                        for dx, dy in (p0_2d, p1_2d):
                            wp = label_origin + (dx * scale + x_char) * right + (dy * scale) * fwd
                            pts.append(Gf.Vec3f(float(wp[0]), float(wp[1]), float(wp[2])))
                        cnts.append(2)

            prim_path = f"{root_path}/cell_{row}_{col}"
            curves = UsdGeom.BasisCurves.Define(stage, prim_path)
            curves.CreateTypeAttr("linear")
            curves.CreateWrapAttr("nonperiodic")
            curves.CreatePointsAttr().Set(Vt.Vec3fArray(pts))
            curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(cnts))
            curves.CreateWidthsAttr().Set(Vt.FloatArray([width] * len(pts)))
            curves.CreateDisplayColorAttr().Set(usd_color)

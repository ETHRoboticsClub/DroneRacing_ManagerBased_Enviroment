# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD visualisation utilities: trajectory, gate direction indicators, gate labels, and demo glow.

All prims created here are UsdGeom.BasisCurves or UsdLux lights — they carry NO
UsdPhysics.CollisionAPI.  PhysX is opt-in in USD: a prim participates in collision
only if CollisionAPI is explicitly applied to it.  BasisCurves.Define() and
UsdLux.SphereLight.Define() never do this, so these prims are invisible to the
physics engine.  They also live under /World/TrajectoryVis, /World/GateIndicatorVis,
/World/GateLabelVis, /World/GateOutlineVis, /World/GateLights — completely separate
from the terrain mesh at /World/ground, so the TerrainImporter cannot affect them.
Written once at env startup, zero per-frame CPU/GPU cost.

Entry points:
    draw_trajectory_usd()       — green closed-loop polyline along trajectory waypoints
    draw_gate_indicators_usd()  — blue sticks showing gate forward (+Y) direction
    draw_gate_labels_usd()      — blue 7-segment digit labels floating above each gate
    draw_gate_outlines_usd()    — emissive square outlines overlaid on each gate frame
    add_gate_sphere_lights_usd()— UsdLux.SphereLight at every gate centre for wide-radius glow
    setup_dark_scene()          — dim dome light for a dark demo environment
    apply_emissive_to_robots()  — apply glowing OmniPBR material to all drone meshes
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


# ---------------------------------------------------------------------------
# Emissive material helper
# ---------------------------------------------------------------------------

def _make_emissive_material(stage, mat_path: str, color: tuple[float, float, float], intensity: float):
    """Create an OmniPBR MDL material with emissive enabled and return it.

    The material is bound to BasisCurves / Mesh prims to produce RTX bloom glow.
    Returns None if USD shader modules are unavailable.
    """
    try:
        from pxr import Gf, Sdf, UsdShade
    except ImportError:
        return None

    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(intensity))

    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def setup_dark_scene(
    dome_light_path: str = "/World/Light",
    ambient_intensity: float = 30.0,
) -> None:
    """Dim the dome light to create a dark environment for the glow demo.

    Args:
        dome_light_path:   USD prim path of the DomeLight (default: /World/Light).
        ambient_intensity: Replacement intensity in nits (default 30.0 — near-black).
    """
    try:
        import omni.usd
        from pxr import UsdLux
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    prim = stage.GetPrimAtPath(dome_light_path)
    if not prim.IsValid():
        return

    light = UsdLux.DomeLight(prim)
    light.CreateIntensityAttr().Set(float(ambient_intensity))


def apply_emissive_to_robots(
    color: tuple[float, float, float] = (1.0, 0.5, 0.0),
    intensity: float = 5.0,
    robots_root: str = "/World/envs",
    mat_root: str = "/World/Looks/RobotGlow",
) -> None:
    """Apply a single shared OmniPBR emissive material to all drone mesh prims.

    Traverses all UsdGeom.Mesh prims under *robots_root* and binds one shared
    glowing material to them — no geometry is modified, only appearance.

    Args:
        color:       RGB glow colour in 0–1 range.
        intensity:   Emissive intensity (higher → brighter bloom in RTX).
        robots_root: Stage path under which robot environment prims live.
        mat_root:    Where to author the shared material prim.
    """
    try:
        import omni.usd
        from pxr import UsdGeom, UsdShade
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    mat = _make_emissive_material(stage, mat_root, color, intensity)
    if mat is None:
        return

    root_prim = stage.GetPrimAtPath(robots_root)
    if not root_prim.IsValid():
        return

    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(robots_root):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


# ---------------------------------------------------------------------------
# Gate glow overlay (demo mode)
# ---------------------------------------------------------------------------

def draw_gate_outlines_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    width: float = 0.06,
    color: tuple[float, float, float] = (1.0, 0.3, 0.0),
    emissive_intensity: float = 20.0,
    root_path: str = "/World/GateOutlineVis",
) -> None:
    """Overlay a glowing square outline on every gate frame using UsdGeom.BasisCurves.

    The outline traces the outer edge of the gate frame (gate_size + bar_thickness on
    each side) as a closed periodic BasisCurves with 4 corner points.  The curve lies in
    the gate's XZ plane (right × world-up), offset a half bar-thickness forward along
    the gate normal so it sits flush with the front face of the frame.

    Args:
        gate_datas:         Ordered list of GateData objects.
        terrain_origins_np: [num_rows, num_cols, 3] world-frame cell centres (numpy).
        centre_offset:      [3] shift from corner-origin to centre-origin.
        num_rows:           Number of terrain rows.
        num_cols:           Number of terrain columns.
        width:              BasisCurves tube diameter in metres.
        color:              RGB colour in 0–1 range.
        emissive_intensity: OmniPBR emissive intensity (>0 = RTX bloom glow).
        root_path:          USD prim path prefix; one child prim per cell.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)
    usd_color = Vt.Vec3fArray([Gf.Vec3f(*color)])

    mat = None
    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat", color, emissive_intensity)

    for row in range(num_rows):
        for col in range(num_cols):
            cell_idx = (row * num_cols + col) % n_cells
            gd = gate_datas[cell_idx]
            origin = terrain_origins_np[row, col]

            # Outer half-extent: opening half + one full bar thickness on each side.
            outer_half = gd.gate_size / 2.0 + gd.bar_thickness

            points: list[Gf.Vec3f] = []
            counts: list[int] = []

            for pos_local, q in zip(gd.positions, gd.orientations):
                pos_world = pos_local + centre_offset + origin

                # Gate right direction: R @ [1,0,0] — horizontal, perpendicular to forward.
                qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                right = np.array([
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy + qw * qz),
                    0.0,  # gates are upright, Z component is zero by construction
                ])
                right /= np.linalg.norm(right) + 1e-8

                # Gate forward (normal): R @ [0,1,0] — offset outline to front face.
                fwd = np.array([
                    2.0 * (qx * qy - qw * qz),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    0.0,
                ])

                # World up is always Z for upright gates.
                up = np.array([0.0, 0.0, 1.0])

                # Offset half a bar-thickness forward so outline sits on the gate front face.
                centre = pos_world + fwd * (gd.bar_thickness / 2.0)

                # 4 corners: top-right, top-left, bottom-left, bottom-right (closed loop).
                corners = [
                    centre + outer_half * right + outer_half * up,
                    centre - outer_half * right + outer_half * up,
                    centre - outer_half * right - outer_half * up,
                    centre + outer_half * right - outer_half * up,
                ]
                for c in corners:
                    points.append(Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))
                counts.append(4)  # periodic BasisCurves closes TR→TL→BL→BR→TR

            prim_path = f"{root_path}/cell_{row}_{col}"
            curves = UsdGeom.BasisCurves.Define(stage, prim_path)
            curves.CreateTypeAttr("linear")
            curves.CreateWrapAttr("periodic")
            curves.CreatePointsAttr().Set(Vt.Vec3fArray(points))
            curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(counts))
            curves.CreateWidthsAttr().Set(Vt.FloatArray([width] * len(points)))
            curves.CreateDisplayColorAttr().Set(usd_color)
            if mat is not None:
                UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)


def add_gate_sphere_lights_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    color: tuple[float, float, float] = (1.0, 0.3, 0.0),
    intensity: float = 3000.0,
    radius: float = 1.5,
    root_path: str = "/World/GateLights",
) -> None:
    """Place a UsdLux.SphereLight at every gate centre for wide-radius real light glow.

    Sphere lights cast actual illumination in RTX, unlike emissive materials which only
    produce bloom.  The *radius* parameter directly controls how far the light reaches —
    larger radius = softer, wider glow.  One light per gate per cell; all cells sharing
    the same circuit layout receive identical lights offset to their world position.

    Args:
        gate_datas:         Ordered list of GateData objects.
        terrain_origins_np: [num_rows, num_cols, 3] world-frame cell centres (numpy).
        centre_offset:      [3] shift from corner-origin to centre-origin.
        num_rows:           Number of terrain rows.
        num_cols:           Number of terrain columns.
        color:              RGB light colour in 0–1 range.
        intensity:          Light intensity in nits.
        radius:             Physical radius of the sphere light in metres — controls glow spread.
        root_path:          USD prim path prefix.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdLux, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)

    for row in range(num_rows):
        for col in range(num_cols):
            cell_idx = (row * num_cols + col) % n_cells
            gd = gate_datas[cell_idx]
            origin = terrain_origins_np[row, col]

            for gate_idx, pos_local in enumerate(gd.positions):
                pos_world = pos_local + centre_offset + origin
                prim_path = f"{root_path}/cell_{row}_{col}/gate_{gate_idx}"

                light = UsdLux.SphereLight.Define(stage, prim_path)
                light.AddTranslateOp().Set(
                    Gf.Vec3d(float(pos_world[0]), float(pos_world[1]), float(pos_world[2]))
                )
                light.CreateRadiusAttr().Set(float(radius))
                light.CreateIntensityAttr().Set(float(intensity))
                light.CreateColorAttr().Set(Gf.Vec3f(*color))
                light.CreateNormalizeAttr().Set(True)  # intensity independent of radius


# ---------------------------------------------------------------------------
# Trajectory / gate visualisation
# ---------------------------------------------------------------------------

def draw_trajectory_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    width: float = 0.015,
    color: tuple[float, float, float] = (0.0, 1.0, 0.2),
    emissive_intensity: float = 0.0,
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
        emissive_intensity: OmniPBR emissive intensity (0 = off; >0 = glowing in RTX).
        root_path:          USD prim path prefix; one child prim per cell is created under it.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)
    usd_color = Vt.Vec3fArray([Gf.Vec3f(*color)])

    mat = None
    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat", color, emissive_intensity)

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
            if mat is not None:
                UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)


def draw_gate_indicators_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    length: float = 0.5,
    width: float = 0.04,
    color: tuple[float, float, float] = (0.1, 0.4, 1.0),
    emissive_intensity: float = 0.0,
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
        emissive_intensity: OmniPBR emissive intensity (0 = off; >0 = glowing in RTX).
        root_path:          USD prim path prefix; one child prim per cell is created under it.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)
    usd_color = Vt.Vec3fArray([Gf.Vec3f(*color)])

    mat = None
    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat", color, emissive_intensity)

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
            if mat is not None:
                UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)


def draw_gate_labels_usd(
    gate_datas: list,
    terrain_origins_np: np.ndarray,
    centre_offset: np.ndarray,
    num_rows: int,
    num_cols: int,
    scale: float = 2.0,
    width: float = 0.012,
    color: tuple[float, float, float] = (0.1, 0.4, 1.0),
    emissive_intensity: float = 0.0,
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
        scale:              Uniform scale factor for digit geometry and character spacing.
        width:              Line tube diameter in metres.
        color:              RGB display colour in 0–1 range.
        emissive_intensity: OmniPBR emissive intensity (0 = off; >0 = glowing in RTX).
        root_path:          USD prim path prefix; one child prim per cell is created under it.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    n_cells = len(gate_datas)
    usd_color = Vt.Vec3fArray([Gf.Vec3f(*color)])

    mat = None
    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat", color, emissive_intensity)

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
            if mat is not None:
                UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Per-step camera-direction arrow (one shaft + arrowhead per drone)
# ---------------------------------------------------------------------------
# Layout per drone: 3 line segments × 2 points = 6 points
#   segment 0: shaft tail → tip
#   segment 1: tip → barb_left
#   segment 2: tip → barb_right
# All segments for all drones share one BasisCurves prim (one prim, N*6 points).

_CAMERA_ARROW_PER_DRONE_POINTS = 6
_CAMERA_ARROW_PER_DRONE_SEGMENTS = 3


def init_camera_arrow_usd(
    num_drones: int,
    width: float = 0.015,
    color: tuple[float, float, float] = (1.0, 0.9, 0.0),
    emissive_intensity: float = 0.0,
    root_path: str = "/World/CameraArrowVis",
) -> str | None:
    """Create one shared UsdGeom.BasisCurves prim holding a camera-direction arrow per drone.

    Points start at the origin; call :func:`update_camera_arrow_usd` every step to set them
    from current drone state.  Returns the prim path (string) or ``None`` if USD is not
    available (e.g. headless without stage).

    Args:
        num_drones:         Number of drones (one arrow each).
        width:              Tube diameter for the BasisCurves.
        color:              RGB display colour (0–1).
        emissive_intensity: OmniPBR emissive intensity (0 = off; >0 = glowing in RTX).
        root_path:          USD prim path of the shared arrow prim.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    n_pts = num_drones * _CAMERA_ARROW_PER_DRONE_POINTS
    n_seg = num_drones * _CAMERA_ARROW_PER_DRONE_SEGMENTS
    points = [Gf.Vec3f(0.0, 0.0, 0.0)] * n_pts
    counts = [2] * n_seg

    prim_path = f"{root_path}/arrows"
    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.CreateTypeAttr("linear")
    curves.CreateWrapAttr("nonperiodic")
    curves.CreatePointsAttr().Set(Vt.Vec3fArray(points))
    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(counts))
    curves.CreateWidthsAttr().Set(Vt.FloatArray([width] * n_pts))
    curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat", color, emissive_intensity)
        UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)
    return prim_path


def update_camera_arrow_usd(
    prim_path: str,
    positions_w: np.ndarray,    # [N, 3] arrow tail anchor in world frame
    directions_w: np.ndarray,   # [N, 3] arrow direction in world frame (need not be unit)
    length: float = 0.25,
    barb_length: float = 0.07,
    barb_angle_deg: float = 25.0,
    z_offset: float = 0.05,
) -> None:
    """Rewrite the points buffer of a camera-arrow prim created by :func:`init_camera_arrow_usd`.

    Each drone gets a shaft from ``positions_w + (0,0,z_offset)`` along normalised
    ``directions_w`` for ``length`` metres, plus two short barbs at the tip forming an
    arrowhead.  For near-vertical shafts the barbs use a world-X reference instead of
    world-Z to avoid degenerate cross products.

    Args:
        prim_path:        Path returned by :func:`init_camera_arrow_usd`.
        positions_w:      [N, 3] tail anchor positions.
        directions_w:     [N, 3] forward directions (will be normalised).
        length:           Shaft length in metres.
        barb_length:      Arrowhead barb length in metres.
        barb_angle_deg:   Half-angle of the arrowhead (between shaft and each barb).
        z_offset:         Vertical offset added to ``positions_w`` so the arrow floats
                          above the drone body (default 5 cm).
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    curves = UsdGeom.BasisCurves(prim)
    if not curves:
        return

    n = positions_w.shape[0]
    dirs = directions_w.astype(np.float64, copy=False)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + 1e-8)

    z_ref = np.array([0.0, 0.0, 1.0])
    x_ref = np.array([1.0, 0.0, 0.0])
    up_dot = np.abs(dirs @ z_ref)
    ref = np.where(up_dot[:, None] < 0.95, z_ref[None, :], x_ref[None, :])
    side = np.cross(dirs, ref)
    side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-8

    tails = positions_w + np.array([0.0, 0.0, z_offset])
    tips = tails + dirs * length

    theta = float(np.deg2rad(barb_angle_deg))
    back = -dirs
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    # `side` is orthogonal to `back`, so the Rodrigues rotation reduces to this 2-term form.
    barb_l = cos_t * back + sin_t * np.cross(side, back)
    barb_r = cos_t * back - sin_t * np.cross(side, back)
    barb_left_end = tips + barb_l * barb_length
    barb_right_end = tips + barb_r * barb_length

    flat = np.empty((n, _CAMERA_ARROW_PER_DRONE_POINTS, 3), dtype=np.float32)
    flat[:, 0] = tails
    flat[:, 1] = tips
    flat[:, 2] = tips
    flat[:, 3] = barb_left_end
    flat[:, 4] = tips
    flat[:, 5] = barb_right_end
    flat = flat.reshape(-1, 3)

    points = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in flat])
    curves.GetPointsAttr().Set(points)


# ---------------------------------------------------------------------------
# Target-gate highlight (FPV race scenario)
# ---------------------------------------------------------------------------
# A single emissive square outline tracking the main drone's next gate. The
# prim is created once at startup with placeholder geometry; its 4 corner
# points are rewritten each step by update_target_gate_highlight_usd() as the
# main drone passes gates.

_TARGET_HIGHLIGHT_POINTS = 4  # closed periodic BasisCurves with 4 corners


def init_target_gate_highlight_usd(
    color: tuple[float, float, float] = (0.0, 1.0, 1.0),
    width: float = 0.10,
    emissive_intensity: float = 30.0,
    root_path: str = "/World/TargetGateHighlight",
) -> str | None:
    """Author a single periodic BasisCurves prim with 4 placeholder corners.

    Returns the prim path so the env can mutate the points each step.
    Returns None when running headless (omni.usd unavailable) so callers
    can no-op safely.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    prim_path = f"{root_path}/Outline"
    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.CreateTypeAttr("linear")
    curves.CreateWrapAttr("periodic")
    placeholder = Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * _TARGET_HIGHLIGHT_POINTS)
    curves.CreatePointsAttr().Set(placeholder)
    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([_TARGET_HIGHLIGHT_POINTS]))
    curves.CreateWidthsAttr().Set(Vt.FloatArray([width] * _TARGET_HIGHLIGHT_POINTS))
    curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat", color, emissive_intensity)
        UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)

    return prim_path


def update_target_gate_highlight_usd(
    prim_path: str,
    gate_pos_w: np.ndarray,
    gate_quat_wxyz: np.ndarray,
    gate_size: float,
    bar_thickness: float,
) -> None:
    """Rewrite the 4 corner points of the highlight outline for the current target gate.

    Geometry matches draw_gate_outlines_usd: outline traces (gate_size/2 + bar_thickness)
    half-extent, offset bar_thickness/2 forward along the gate normal so it sits flush
    with the front face of the frame.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    curves = UsdGeom.BasisCurves(prim)

    qw, qx, qy, qz = (float(gate_quat_wxyz[0]), float(gate_quat_wxyz[1]),
                       float(gate_quat_wxyz[2]), float(gate_quat_wxyz[3]))
    # Gate right axis: R @ [1,0,0] horizontal.
    right = np.array([
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qw * qz),
        0.0,
    ])
    right /= np.linalg.norm(right) + 1e-8
    # Gate forward (passage normal): R @ [0,1,0].
    fwd = np.array([
        2.0 * (qx * qy - qw * qz),
        1.0 - 2.0 * (qx * qx + qz * qz),
        0.0,
    ])
    up = np.array([0.0, 0.0, 1.0])

    outer_half = gate_size / 2.0 + bar_thickness
    centre = np.asarray(gate_pos_w, dtype=float) + fwd * (bar_thickness / 2.0)
    corners = [
        centre + outer_half * right + outer_half * up,
        centre - outer_half * right + outer_half * up,
        centre - outer_half * right - outer_half * up,
        centre + outer_half * right - outer_half * up,
    ]
    points = Vt.Vec3fArray([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in corners])
    curves.GetPointsAttr().Set(points)


# ---------------------------------------------------------------------------
# Opponent glow marker (race scenario)
# ---------------------------------------------------------------------------
# A glowing sphere parented at /World/OpponentGlow that tracks env_1's drone
# position each step. Lives OUTSIDE the env namespace so it bypasses the USD
# instance-proxy issue (env_1's mesh prims are inside env_0's prototype and
# can't be material-bound independently).

def init_opponent_glow_marker_usd(
    color: tuple[float, float, float] = (0.75, 0.20, 1.0),
    radius: float = 0.0,                        # kept for backward-compat; ignored
    emissive_intensity: float = 0.0,            # ditto
    light_radius: float = 0.03,
    light_intensity: float = 5.0e7,
    root_path: str = "/World/OpponentGlow",
) -> str | None:
    """Drone glow: a tiny visible UsdLux.SphereLight whose body is small enough to be
    a single bright dot from any reasonable viewing distance.

    No UsdGeom.Sphere is created — the user's "trailing sphere" complaint was about a
    separate visible mesh that lagged the drone by a frame. The previous attempts to
    hide the SphereLight body (treatAsPoint=True, MakeInvisible()) also killed the
    illumination on this Hydra/RTX delegate. So instead the light keeps its default
    visibility but its body shrinks to ~3 cm and intensity is bumped ~1000× from the
    original 4e4 baseline, giving a strong halo on the surrounding geometry and a tiny
    bright LED-looking core at the drone position.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux
    except ImportError:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    parent = UsdGeom.Xform.Define(stage, root_path)
    parent_xform = UsdGeom.Xformable(parent.GetPrim())
    parent_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    light_path = f"{root_path}/Halo"
    light = UsdLux.SphereLight.Define(stage, light_path)
    light.CreateRadiusAttr().Set(float(light_radius))
    light.CreateIntensityAttr().Set(float(light_intensity))
    light.CreateColorAttr().Set(Gf.Vec3f(*color))

    return root_path


def update_opponent_glow_marker_usd(prim_path: str, pos_w: np.ndarray) -> None:
    """Move the parent Xform of the opponent glow to match env_1's body world position.

    `prim_path` is the parent root (created by init_opponent_glow_marker_usd); both
    the visible sphere and the SphereLight are its children, so a single translate
    op moves everything together.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if not ops:
        return
    ops[0].Set(Gf.Vec3d(float(pos_w[0]), float(pos_w[1]), float(pos_w[2])))


# ---------------------------------------------------------------------------
# Drone light trails (per-step neon ribbon behind each drone)
# ---------------------------------------------------------------------------

def init_drone_trail_usd(
    env_id: int,
    color: tuple[float, float, float],
    trail_length: int,
    width: float = 0.04,
    emissive_intensity: float = 25.0,
    root_path: str = "/World/DroneTrails",
) -> str | None:
    """Create one linear BasisCurves prim per drone with `trail_length` placeholder points.

    Points are rewritten each step from a rolling position buffer; the curve is non-periodic
    so the head ↔ tail are visually distinct.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    prim_path = f"{root_path}/trail_env_{env_id}"
    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.CreateTypeAttr("linear")
    curves.CreateWrapAttr("nonperiodic")
    placeholder = Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * trail_length)
    curves.CreatePointsAttr().Set(placeholder)
    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([trail_length]))
    curves.CreateWidthsAttr().Set(Vt.FloatArray([width] * trail_length))
    curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat_env_{env_id}", color, emissive_intensity)
        if mat is not None:
            UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)

    return prim_path


def update_drone_trail_usd(prim_path: str, positions_w: np.ndarray) -> None:
    """Rewrite the BasisCurves points from a numpy array of shape [trail_length, 3]."""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    curves = UsdGeom.BasisCurves(prim)
    pts = Vt.Vec3fArray([
        Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions_w
    ])
    curves.GetPointsAttr().Set(pts)


# ---------------------------------------------------------------------------
# Stadium perimeter pillars (decorative ring around the racing area)
# ---------------------------------------------------------------------------

def draw_stadium_pillars_usd(
    terrain_origins_np: np.ndarray,
    count: int = 12,
    ring_radius: float = 10.0,
    pillar_radius: float = 0.3,
    pillar_height: float = 4.0,
    colors: tuple[tuple[float, float, float], ...] = ((1.0, 0.45, 0.0), (0.3, 0.7, 1.0)),
    emissive_intensity: float = 60.0,
    root_path: str = "/World/StadiumPillars",
) -> None:
    """Ring of glowing cylindrical pillars around the racing area.

    Pillars cycle through ``colors`` so the ring has alternating hues. One emissive
    material is authored per unique color (deduped by index).
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    centre = terrain_origins_np.reshape(-1, 3)[0]
    cx, cy = float(centre[0]), float(centre[1])

    # One material per unique color in the palette.
    mats = []
    if emissive_intensity > 0.0:
        for k, col in enumerate(colors):
            mats.append(_make_emissive_material(stage, f"{root_path}/Mat_{k}", col, emissive_intensity))

    for i in range(count):
        theta = 2.0 * np.pi * i / count
        x = cx + ring_radius * np.cos(theta)
        y = cy + ring_radius * np.sin(theta)
        z = pillar_height / 2.0  # base at z=0
        col = colors[i % len(colors)]

        prim_path = f"{root_path}/pillar_{i:02d}"
        cyl = UsdGeom.Cylinder.Define(stage, prim_path)
        cyl.CreateAxisAttr().Set("Z")
        cyl.CreateRadiusAttr().Set(float(pillar_radius))
        cyl.CreateHeightAttr().Set(float(pillar_height))
        cyl.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*col)]))

        xform = UsdGeom.Xformable(cyl.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))

        if mats:
            mat = mats[i % len(mats)]
            if mat is not None:
                UsdShade.MaterialBindingAPI(cyl.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Cyberpunk atmosphere — RTX post-processing + fog
# ---------------------------------------------------------------------------

def setup_cyberpunk_atmosphere(
    fog_color: tuple[float, float, float] = (0.05, 0.02, 0.12),
    fog_intensity: float = 1.0,
    fog_start_dist: float = 2.0,
    fog_end_dist: float = 80.0,
    fog_distance_density: float = 0.025,
    chromatic_aberration_strength: float = 0.005,
    dome_light_intensity: float = 80.0,
    dome_light_color: tuple[float, float, float] = (0.10, 0.05, 0.22),
    dome_light_path: str = "/World/Light",
) -> None:
    """Apply the carb-level RTX post-processing for a cyberpunk vibe.

    - Volumetric distance fog: light bleeds atmospherically, makes neon emissives
      bloom into "god rays" through the medium.
    - Chromatic aberration: subtle RGB fringing at frame edges — classic cyberpunk
      lens look.
    - Dim, deep-violet dome light: near-black sky tinted dark indigo.

    All of these are global render settings (carb), so they affect the whole scene
    once. No per-step cost.
    """
    # Dim and tint the existing dome light.
    try:
        import omni.usd
        from pxr import Gf, UsdLux
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            prim = stage.GetPrimAtPath(dome_light_path)
            if prim.IsValid():
                light = UsdLux.DomeLight(prim)
                light.CreateIntensityAttr().Set(float(dome_light_intensity))
                light.CreateColorAttr().Set(Gf.Vec3f(*dome_light_color))
    except ImportError:
        pass

    # Carb / RTX settings — fog + chromatic aberration.
    try:
        import carb
        s = carb.settings.get_settings()
        # Volumetric distance fog.
        s.set_bool("/rtx/fog/enabled", True)
        s.set("/rtx/fog/fogColor", list(fog_color))
        s.set_float("/rtx/fog/fogColorIntensity", float(fog_intensity))
        s.set_bool("/rtx/fog/fogDistanceBased/enabled", True)
        s.set_float("/rtx/fog/fogStartDist", float(fog_start_dist))
        s.set_float("/rtx/fog/fogEndDist", float(fog_end_dist))
        s.set_float("/rtx/fog/fogDistanceDensity", float(fog_distance_density))

        # Chromatic aberration — only meaningful with non-zero strength.
        if chromatic_aberration_strength != 0.0:
            s.set_bool("/rtx/post/chromaticAberration/enabled", True)
            # R/B fringe outward, G mostly neutral — classic look.
            s.set_float("/rtx/post/chromaticAberration/strengthR", float(chromatic_aberration_strength))
            s.set_float("/rtx/post/chromaticAberration/strengthG", 0.0)
            s.set_float("/rtx/post/chromaticAberration/strengthB", float(-chromatic_aberration_strength))
    except ImportError:
        return


# ---------------------------------------------------------------------------
# Wet reflective ground — the cyberpunk core technique
# ---------------------------------------------------------------------------

def _make_pbr_material(
    stage,
    mat_path: str,
    diffuse_color: tuple[float, float, float],
    metallic: float,
    roughness: float,
):
    """OmniPBR material with metallic + roughness inputs, no emission.

    Low roughness + high metallic → sharp mirror-like reflections. Used for the
    wet ground that mirrors all the neon emissives in the scene.
    """
    try:
        from pxr import Gf, Sdf, UsdShade
    except ImportError:
        return None

    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_color))
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(False)

    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def add_wet_reflective_ground(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: float = 80.0,
    color: tuple[float, float, float] = (0.012, 0.012, 0.025),
    metallic: float = 0.75,
    roughness: float = 0.18,
    z_offset: float = -0.005,
    prim_path: str = "/World/WetGround",
) -> None:
    """Large flat quad with a wet-asphalt PBR material under the racing area.

    Sits slightly below the terrain mesh so it doesn't z-fight. Reflects all
    emissive prims (gates, pillars, drone trails) — the doubled neon is what
    sells the cyberpunk look.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    cx, cy, cz = centre
    z = cz + z_offset
    half = size / 2.0

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    pts = [
        Gf.Vec3f(cx - half, cy - half, z),
        Gf.Vec3f(cx + half, cy - half, z),
        Gf.Vec3f(cx + half, cy + half, z),
        Gf.Vec3f(cx - half, cy + half, z),
    ]
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    # Upward-facing normal so it shows up correctly.
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0)] * 4))
    mesh.SetNormalsInterpolation("vertex")

    mat = _make_pbr_material(stage, f"{prim_path}/Mat", color, metallic, roughness)
    if mat is not None:
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Holographic floor grid (Tron-style)
# ---------------------------------------------------------------------------

def add_holographic_floor_grid(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: float = 30.0,
    spacing: float = 1.5,
    line_width: float = 0.025,
    color: tuple[float, float, float] = (0.0, 1.0, 0.95),
    emissive_intensity: float = 18.0,
    z_offset: float = 0.005,
    root_path: str = "/World/HoloGrid",
) -> None:
    """Emissive BasisCurves grid lines on the ground plane.

    Reflects strongly off the wet ground beneath it → "depth" effect.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    cx, cy, cz = centre
    z = cz + z_offset
    half = size / 2.0
    n_lines = int(size / spacing) + 1

    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    for i in range(n_lines):
        offset = -half + i * spacing
        # Line parallel to X axis
        points.append(Gf.Vec3f(cx - half, cy + offset, z))
        points.append(Gf.Vec3f(cx + half, cy + offset, z))
        counts.append(2)
        # Line parallel to Y axis
        points.append(Gf.Vec3f(cx + offset, cy - half, z))
        points.append(Gf.Vec3f(cx + offset, cy + half, z))
        counts.append(2)

    prim_path = f"{root_path}/Grid"
    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.CreateTypeAttr("linear")
    curves.CreateWrapAttr("nonperiodic")
    curves.CreatePointsAttr().Set(Vt.Vec3fArray(points))
    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(counts))
    curves.CreateWidthsAttr().Set(Vt.FloatArray([line_width] * len(points)))
    curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    if emissive_intensity > 0.0:
        mat = _make_emissive_material(stage, f"{root_path}/Mat", color, emissive_intensity)
        if mat is not None:
            UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Neon walls between stadium pillars
# ---------------------------------------------------------------------------

def add_neon_walls_between_pillars(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pillar_count: int = 12,
    ring_radius: float = 9.5,
    heights: tuple[float, ...] = (0.4, 1.4, 2.4, 3.4),
    colors: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.9),    # magenta
        (0.0, 1.0, 1.0),    # cyan
        (1.0, 0.0, 0.9),
        (0.0, 1.0, 1.0),
    ),
    line_width: float = 0.05,
    emissive_intensity: float = 30.0,
    root_path: str = "/World/NeonWalls",
) -> None:
    """Closed-loop BasisCurves connecting all pillars at multiple heights.

    Each height level is one BasisCurves prim with its own emissive material, so
    colors can alternate (magenta/cyan stripes). This is the "fence of light"
    that defines the cyberpunk arena boundary.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    cx, cy, _ = centre
    # Pre-compute pillar centre positions on a ring (matches draw_stadium_pillars_usd).
    pillar_xy = []
    for i in range(pillar_count):
        theta = 2.0 * np.pi * i / pillar_count
        pillar_xy.append((cx + ring_radius * np.cos(theta),
                          cy + ring_radius * np.sin(theta)))

    for level_idx, (z_level, line_color) in enumerate(zip(heights, colors)):
        points = [Gf.Vec3f(float(x), float(y), float(z_level)) for x, y in pillar_xy]
        prim_path = f"{root_path}/level_{level_idx}"
        curves = UsdGeom.BasisCurves.Define(stage, prim_path)
        curves.CreateTypeAttr("linear")
        curves.CreateWrapAttr("periodic")  # closes the loop
        curves.CreatePointsAttr().Set(Vt.Vec3fArray(points))
        curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([pillar_count]))
        curves.CreateWidthsAttr().Set(Vt.FloatArray([line_width] * pillar_count))
        curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*line_color)]))

        if emissive_intensity > 0.0:
            mat = _make_emissive_material(
                stage, f"{root_path}/Mat_{level_idx}", line_color, emissive_intensity
            )
            if mat is not None:
                UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Hemispherical dome cap above the stadium pillars
# ---------------------------------------------------------------------------

def add_neon_dome_cap(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_z: float = 10.0,
    radius: float = 9.5,
    n_meridians: int = 12,
    n_latitudes: int = 5,
    meridian_segments: int = 18,
    colors: tuple[tuple[float, float, float], ...] = ((1.0, 0.45, 0.0), (0.3, 0.7, 1.0)),
    line_width: float = 0.05,
    emissive_intensity: float = 50.0,
    root_path: str = "/World/NeonDomeCap",
) -> None:
    """Hemispherical wireframe cap sitting above the pillar tops.

    Composed of two BasisCurves families:
    - Meridians: arcs from the base ring (z=base_z) curving up to the apex
      (z=base_z + radius). One arc per meridian, sampled at `meridian_segments+1` points.
    - Latitudes: closed loops at evenly-spaced θ values from horizontal toward the apex.

    Together they form the visual impression of a half-sphere arena ceiling. Colors
    alternate between palette entries (orange/light-blue by default).
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    cx, cy, _ = centre
    half_pi = float(np.pi) / 2.0

    # Material cache (one per color).
    mats = []
    if emissive_intensity > 0.0:
        for k, col in enumerate(colors):
            mats.append(_make_emissive_material(stage, f"{root_path}/Mat_{k}", col, emissive_intensity))

    # --- Meridian arcs ---
    for m in range(n_meridians):
        phi = 2.0 * np.pi * m / n_meridians
        cos_phi = float(np.cos(phi))
        sin_phi = float(np.sin(phi))
        pts = []
        for k in range(meridian_segments + 1):
            theta = half_pi * k / meridian_segments  # 0 = equator, π/2 = apex
            r = radius * float(np.cos(theta))
            z = base_z + radius * float(np.sin(theta))
            pts.append(Gf.Vec3f(cx + r * cos_phi, cy + r * sin_phi, z))

        col = colors[m % len(colors)]
        prim_path = f"{root_path}/meridian_{m:02d}"
        curves = UsdGeom.BasisCurves.Define(stage, prim_path)
        curves.CreateTypeAttr("linear")
        curves.CreateWrapAttr("nonperiodic")
        curves.CreatePointsAttr().Set(Vt.Vec3fArray(pts))
        curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))
        curves.CreateWidthsAttr().Set(Vt.FloatArray([line_width] * len(pts)))
        curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*col)]))
        if mats:
            mat = mats[m % len(mats)]
            if mat is not None:
                UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)

    # --- Latitudinal rings (skip the equator — it's already the top neon wall — and skip the apex) ---
    for j in range(1, n_latitudes + 1):
        theta_j = half_pi * j / (n_latitudes + 1)
        r_j = radius * float(np.cos(theta_j))
        z_j = base_z + radius * float(np.sin(theta_j))
        loop_pts = []
        for m in range(n_meridians):
            phi = 2.0 * np.pi * m / n_meridians
            loop_pts.append(Gf.Vec3f(cx + r_j * float(np.cos(phi)), cy + r_j * float(np.sin(phi)), z_j))

        col = colors[j % len(colors)]
        prim_path = f"{root_path}/latitude_{j:02d}"
        curves = UsdGeom.BasisCurves.Define(stage, prim_path)
        curves.CreateTypeAttr("linear")
        curves.CreateWrapAttr("periodic")
        curves.CreatePointsAttr().Set(Vt.Vec3fArray(loop_pts))
        curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([n_meridians]))
        curves.CreateWidthsAttr().Set(Vt.FloatArray([line_width] * n_meridians))
        curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*col)]))
        if mats:
            mat = mats[j % len(mats)]
            if mat is not None:
                UsdShade.MaterialBindingAPI(curves.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Sky / dome interior accent points (procedural "starfield" texture)
# ---------------------------------------------------------------------------

def add_sky_dome_accents(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_z: float = 10.0,
    radius: float = 9.5,
    count: int = 160,
    palette: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.5, 0.0),     # orange
        (1.0, 0.7, 0.3),     # warm amber
        (0.3, 0.7, 1.0),     # light blue
        (1.0, 1.0, 1.0),     # white
    ),
    point_radius_range: tuple[float, float] = (0.04, 0.12),
    emissive_intensity: float = 80.0,
    seed: int = 7,
    root_path: str = "/World/SkyAccents",
) -> None:
    """Scatter small emissive spheres on the inner surface of the hemispherical dome.

    Looks like distant city lights / a stylised starfield. Gives the otherwise-flat
    dome ceiling visible texture and adds reflected highlights on the wet ground.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade, Vt
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    cx, cy, _ = centre
    rng = np.random.default_rng(seed)
    # Reuse one material per palette color.
    mats = [
        _make_emissive_material(stage, f"{root_path}/Mat_{k}", col, emissive_intensity)
        for k, col in enumerate(palette)
    ]

    half_pi = float(np.pi) / 2.0
    inset = 0.97  # place slightly inside the dome surface so they aren't culled
    for i in range(count):
        # Uniform sample on the upper hemisphere.
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        cos_theta = float(rng.uniform(0.05, 1.0))  # bias toward upper portion
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        # z = sin_theta in the half-pi parametrisation (i.e., cos_theta here = sin(latitude))
        x = cx + radius * inset * cos_theta * float(np.cos(phi))
        y = cy + radius * inset * cos_theta * float(np.sin(phi))
        z = base_z + radius * inset * sin_theta * 0.0 + radius * inset * 0.0  # placeholder
        # Recompute z with elevation = arcsin(cos_theta) — we want it on the hemisphere
        # mapped to the cap above base_z. Let's use elevation_frac in [0, 1]:
        elev = float(rng.uniform(0.05, 1.0))   # 0 = equator, 1 = apex
        theta = half_pi * elev
        r = radius * inset * float(np.cos(theta))
        z = base_z + radius * inset * float(np.sin(theta))
        x = cx + r * float(np.cos(phi))
        y = cy + r * float(np.sin(phi))

        col_idx = i % len(palette)
        radius_pt = float(rng.uniform(point_radius_range[0], point_radius_range[1]))

        prim_path = f"{root_path}/dot_{i:03d}"
        s = UsdGeom.Sphere.Define(stage, prim_path)
        s.CreateRadiusAttr().Set(radius_pt)
        s.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*palette[col_idx])]))
        xform = UsdGeom.Xformable(s.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        mat = mats[col_idx]
        if mat is not None:
            UsdShade.MaterialBindingAPI(s.GetPrim()).Bind(mat)


# ---------------------------------------------------------------------------
# Show / competition lights — sweeping spotlights mounted near the dome apex
# ---------------------------------------------------------------------------

def _make_translucent_emissive_material(stage, mat_path: str, color, intensity: float, opacity: float):
    """OmniPBR with emission + opacity, used for visible beam-shaft cones."""
    try:
        from pxr import Gf, Sdf, UsdShade
    except ImportError:
        return None

    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(intensity))
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def init_show_lights_usd(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    count: int = 8,
    ring_radius: float = 5.0,
    ring_height: float = 17.0,
    base_pitch_deg: float = -15.0,
    cone_angle_deg: float = 10.0,
    cone_softness: float = 0.4,
    sphere_radius: float = 0.05,
    intensity: float = 5.0e7,
    colors: tuple[tuple[float, float, float], ...] = (
        (0.3, 0.7, 1.0),    # light blue
        (0.7, 0.2, 1.0),    # purple
        (1.0, 1.0, 1.0),    # white
    ),
    phase_speed_jitter: tuple[float, ...] = (1.0, 1.2, 0.85, 1.1, 0.9, 1.3, 0.95, 1.15),
    beam_shaft_length: float = 22.0,
    beam_shaft_radius_scale: float = 1.0,
    beam_shaft_emissive: float = 8.0,
    beam_shaft_opacity: float = 0.18,
    root_path: str = "/World/ShowLights",
) -> list[dict]:
    """N spotlights near the dome apex with visible translucent beam shafts.

    Each light is a UsdGeom.Xform parent containing:
      - A UsdLux.SphereLight with cone shaping (point source, casts illumination)
      - A UsdGeom.Cone with translucent emissive material (visible beam shaft)

    The parent's op chain `Translate → RotateZ(sweep) → RotateY(pitch)` means the local
    -Z direction is the aim/beam direction. The cone is a child translated to
    (0, 0, -beam_shaft_length/2) so its apex sits at the light origin and its base
    extends along -Z (the beam direction) to reach the floor.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, UsdShade
    except ImportError:
        return []

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []

    cx, cy, _ = centre
    lights: list[dict] = []

    # Cache one beam-shaft material per color (translucent, low emissive).
    beam_mats: list = []
    for k, col in enumerate(colors):
        beam_mats.append(_make_translucent_emissive_material(
            stage, f"{root_path}/BeamMat_{k}", col, beam_shaft_emissive, beam_shaft_opacity
        ))

    # Cone geometry: base radius = tan(half-cone-angle) × length.
    cone_half = float(np.radians(cone_angle_deg)) * 0.5
    base_radius = beam_shaft_length * float(np.tan(cone_half)) * beam_shaft_radius_scale

    for i in range(count):
        initial_yaw = 360.0 * i / count
        rx = cx + ring_radius * float(np.cos(np.radians(initial_yaw)))
        ry = cy + ring_radius * float(np.sin(np.radians(initial_yaw)))
        rz = ring_height
        col = colors[i % len(colors)]
        phase = float(phase_speed_jitter[i % len(phase_speed_jitter)])

        parent_path = f"{root_path}/light_{i:02d}"
        parent = UsdGeom.Xform.Define(stage, parent_path)
        x_parent = UsdGeom.Xformable(parent.GetPrim())
        x_parent.AddTranslateOp().Set(Gf.Vec3d(rx, ry, rz))
        # Mutable yaw (sweep) op — addressed later by op name "sweep".
        rot_z = x_parent.AddRotateZOp(opSuffix="sweep")
        rot_z.Set(initial_yaw)
        # Fixed downward tilt.
        x_parent.AddRotateYOp(opSuffix="pitch").Set(float(base_pitch_deg))

        # SphereLight with cone shaping = real spotlight. Visible body kept tiny so the
        # source itself appears as a single bright dot at the dome apex, while the
        # cone shaping focuses emission downward in a sweeping beam. (Both treatAsPoint
        # and MakeInvisible suppressed emission entirely on this RTX delegate.)
        light_path = f"{parent_path}/Light"
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateRadiusAttr().Set(float(sphere_radius))
        light.CreateIntensityAttr().Set(float(intensity))
        light.CreateColorAttr().Set(Gf.Vec3f(*col))
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr().Set(float(cone_angle_deg))
        shaping.CreateShapingConeSoftnessAttr().Set(float(cone_softness))

        # Visible beam shaft: UsdGeom.Cone. Default cone has apex at +Z*H/2 and base at
        # -Z*H/2 (axis=Z). To put the apex at the parent origin (light source) and the
        # base at -Z*H (the floor along the aim direction), translate by -H/2 in Z.
        # NO rotation needed — the previous RotateX(180) flipped the cone upside-down.
        cone_path = f"{parent_path}/BeamShaft"
        cone = UsdGeom.Cone.Define(stage, cone_path)
        cone.CreateAxisAttr().Set("Z")
        cone.CreateHeightAttr().Set(float(beam_shaft_length))
        cone.CreateRadiusAttr().Set(float(base_radius))
        x_cone = UsdGeom.Xformable(cone.GetPrim())
        x_cone.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -beam_shaft_length / 2.0))
        cone.CreateDoubleSidedAttr().Set(True)
        beam_mat = beam_mats[i % len(beam_mats)]
        if beam_mat is not None:
            UsdShade.MaterialBindingAPI(cone.GetPrim()).Bind(beam_mat)

        lights.append({
            "path": parent_path,
            "yaw_init_deg": initial_yaw,
            "phase_speed": phase,
        })
    return lights


def update_show_light_yaw_usd(parent_path: str, yaw_deg: float) -> None:
    """Mutate the RotateZ-sweep op on a show light's parent Xform."""
    try:
        import omni.usd
        from pxr import UsdGeom
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(parent_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if "sweep" in op.GetOpName():
            op.Set(float(yaw_deg))
            return


# ---------------------------------------------------------------------------
# Party lights — random scattered drone-glow-style SphereLights flickering on/off
# ---------------------------------------------------------------------------

def init_party_lights_usd(
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    count: int = 20,
    plan_radius_range: tuple[float, float] = (3.0, 9.4),
    height_range: tuple[float, float] = (1.5, 14.0),
    palette: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.3, 0.0),    # orange
        (0.3, 0.7, 1.0),    # light blue
        (0.75, 0.20, 1.0),  # purple
        (1.0, 1.0, 1.0),    # white
        (0.0, 1.0, 0.5),    # mint
        (1.0, 0.1, 0.6),    # hot pink
    ),
    light_body_radius: float = 0.04,
    base_intensity: float = 2.0e7,
    intensity_freq_range: tuple[float, float] = (0.4, 2.2),
    sharpness_range: tuple[float, float] = (1.5, 4.0),
    color_change_period_range: tuple[float, float] = (2.5, 8.0),
    seed: int = 42,
    root_path: str = "/World/PartyLights",
) -> list[dict]:
    """Scatter `count` SphereLights at random positions in the arena volume.

    Each light uses the same emission style as the drone glow markers: a small visible
    body (~4 cm) with very high intensity. Per-step modulation produces a sharp on/off
    flicker (``base * max(0, sin(2πf·t + φ))**sharpness``) so each light blinks on/off
    on its own schedule. Every ``color_change_period`` seconds the light also switches
    to another palette color. The aggregate looks like a randomised concert light show.

    Returns a list of dicts with per-light metadata for `update_party_lights_usd()`.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux
    except ImportError:
        return []

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []

    cx, cy, cz = centre
    rng = np.random.default_rng(seed)
    lights: list[dict] = []
    n_palette = len(palette)

    r_min, r_max = float(plan_radius_range[0]), float(plan_radius_range[1])
    for i in range(count):
        # Linear-in-radius sampling within the [r_min, r_max] annulus. This biases
        # toward the outer ring (wall side) naturally because the circumference at a
        # given radius grows with r, so equal radius-bin density means more lights
        # along the perimeter.
        theta = rng.uniform(0.0, 2.0 * np.pi)
        r = float(rng.uniform(r_min, r_max))
        x = cx + r * float(np.cos(theta))
        y = cy + r * float(np.sin(theta))
        z = cz + rng.uniform(height_range[0], height_range[1])

        color_idx = int(rng.integers(0, n_palette))
        color = palette[color_idx]
        freq = float(rng.uniform(*intensity_freq_range))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        sharpness = float(rng.uniform(*sharpness_range))
        color_change_period = float(rng.uniform(*color_change_period_range))
        # Phase-offset first color change so they don't all swap together.
        first_change_at = float(rng.uniform(0.0, color_change_period))

        prim_path = f"{root_path}/light_{i:03d}"
        # Parent Xform with a translate (fixed; lights don't move).
        parent = UsdGeom.Xform.Define(stage, prim_path)
        x_parent = UsdGeom.Xformable(parent.GetPrim())
        x_parent.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))

        light_path = f"{prim_path}/Halo"
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateRadiusAttr().Set(float(light_body_radius))
        light.CreateIntensityAttr().Set(0.0)                    # starts off
        light.CreateColorAttr().Set(Gf.Vec3f(*color))

        lights.append({
            "path": light_path,
            "freq": freq,
            "phase": phase,
            "sharpness": sharpness,
            "color_idx": color_idx,
            "color_change_period": color_change_period,
            "next_color_change_at": first_change_at,
            "base_intensity": float(base_intensity),
        })
    return lights


def update_party_light_intensity_usd(light_path: str, intensity: float) -> None:
    """Set ``inputs:intensity`` on a UsdLux.SphereLight prim."""
    try:
        import omni.usd
        from pxr import UsdLux
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(light_path)
    if not prim.IsValid():
        return
    UsdLux.SphereLight(prim).GetIntensityAttr().Set(float(intensity))


def update_party_light_color_usd(light_path: str, color: tuple[float, float, float]) -> None:
    """Set the color attribute on a UsdLux.SphereLight prim."""
    try:
        import omni.usd
        from pxr import Gf, UsdLux
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(light_path)
    if not prim.IsValid():
        return
    UsdLux.SphereLight(prim).GetColorAttr().Set(Gf.Vec3f(*color))


# ---------------------------------------------------------------------------
# Emissive intensity mutator (used for flickering effects)
# ---------------------------------------------------------------------------

def update_emissive_intensity_usd(mat_path: str, intensity: float) -> None:
    """Set the ``inputs:emissive_intensity`` attribute on the OmniPBR shader under mat_path."""
    try:
        import omni.usd
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    shader_prim = stage.GetPrimAtPath(f"{mat_path}/Shader")
    if not shader_prim.IsValid():
        return
    attr = shader_prim.GetAttribute("inputs:emissive_intensity")
    if attr.IsValid():
        attr.Set(float(intensity))


def update_emissive_material_full_usd(
    mat_path: str,
    color: tuple[float, float, float],
    intensity: float,
) -> None:
    """Update color + intensity on a bound emissive material in one shot.

    Mutates ``inputs:emissive_color``, ``inputs:diffuse_color_constant`` and
    ``inputs:emissive_intensity`` on the OmniPBR shader. Setting all three together
    makes color flicker visible — some Hydra delegates only refresh the bound material
    when multiple inputs change.
    """
    try:
        import omni.usd
        from pxr import Gf
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    shader_prim = stage.GetPrimAtPath(f"{mat_path}/Shader")
    if not shader_prim.IsValid():
        return
    col = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    for name in ("inputs:emissive_color", "inputs:diffuse_color_constant"):
        attr = shader_prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(col)
    intensity_attr = shader_prim.GetAttribute("inputs:emissive_intensity")
    if intensity_attr.IsValid():
        intensity_attr.Set(float(intensity))


def update_curves_display_color_usd(prim_path: str, color: tuple[float, float, float]) -> None:
    """Set the ``primvars:displayColor`` attribute on a BasisCurves prim.

    Used as a fallback / additional channel for flicker effects — displayColor updates
    every step are picked up by all USD renderers immediately.
    """
    try:
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    curves = UsdGeom.BasisCurves(prim)
    curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

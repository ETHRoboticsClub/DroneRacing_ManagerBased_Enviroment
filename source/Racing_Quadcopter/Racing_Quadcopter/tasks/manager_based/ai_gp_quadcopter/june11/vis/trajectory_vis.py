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

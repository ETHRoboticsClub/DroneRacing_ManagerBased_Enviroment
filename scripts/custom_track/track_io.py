"""Track I/O utilities for the drone-racing custom track designer.

Pure Python (numpy + scipy + PyYAML only). No Isaac Lab or torch imports.

Disk format (YAML, ``schema_version == 1``)::

    schema_version: 1
    gate_size: 0.7
    bar_thickness: 0.05
    loop_closed: true
    gate_track:
      - position: [0.0, 0.0, 1.5]
        rotation: [0.0, 0.0, 0.0]   # [roll, pitch, yaw] in degrees
        auto_yaw: true
      - ...

Conventions
-----------
- World frame, metres.
- ``rotation`` is ``[roll, pitch, yaw]`` in **degrees** (extrinsic XYZ / ROS RPY order):
    1. roll  — rotation about gate-local +X (right bar axis)
    2. pitch — rotation about gate-local +Y (passage direction)
    3. yaw   — rotation about world +Z, **clockwise from +Y**:
               0° → gate passage faces +Y, 90° → faces +X
- ``auto_yaw=true`` overrides the stored yaw with the closed-loop spline tangent
  direction at export time. It does **not** touch pitch or roll.

Gate ordering in ``gate_track`` is the order the drone visits them, matching
``RacingQuadcopterEnv._gate_positions`` insertion order.

Loaded dict fields (returned by ``load_track``)
------------------------------------------------
  gate_track   : list of gate dicts {position, rotation, auto_yaw}
  positions    : list[list[float]]  # [N, 3] — gate centres
  orientations : list[list[float]]  # [N, 3] — [roll, pitch, yaw] degrees
  gate_size    : float
  bar_thickness: float
  loop_closed  : bool
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# ---------------------------------------------------------------------------
# Load defaults from config.yaml
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        import yaml
    except ModuleNotFoundError:
        return {"gate_size": 0.7, "bar_thickness": 0.05, "loop_closed": True}
    if _CONFIG_PATH.exists():
        cfg = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        return {
            "gate_size":     float(cfg.get("gate_size", 0.7)),
            "bar_thickness": float(cfg.get("bar_thickness", 0.05)),
            "loop_closed":   bool(cfg.get("loop_closed", True)),
        }
    return {"gate_size": 0.7, "bar_thickness": 0.05, "loop_closed": True}


DEFAULTS: dict = _load_config()


# ---------------------------------------------------------------------------
# Construction / persistence
# ---------------------------------------------------------------------------

def default_track() -> dict:
    """Return an empty track populated with default global parameters."""
    return {
        "schema_version":  SCHEMA_VERSION,
        "track_name":      "",
        "description":     "",
        "elements_tested": [],
        "gate_size":       DEFAULTS["gate_size"],
        "bar_thickness":   DEFAULTS["bar_thickness"],
        "loop_closed":     DEFAULTS["loop_closed"],
        "gate_track":      [],
        "positions":       [],
        "orientations":    [],
    }


def _gate_to_dict(g: dict) -> dict:
    """Normalise a gate to the canonical on-disk form."""
    pos = g.get("position", [0.0, 0.0, 1.5])
    rot = g.get("rotation", [0.0, 0.0, 0.0])
    return {
        "position": [float(pos[0]), float(pos[1]), float(pos[2])],
        "rotation": [float(rot[0]), float(rot[1]), float(rot[2])],
        "auto_yaw": bool(g.get("auto_yaw", True)),
    }


def save_track(path: str | Path, track: dict) -> None:
    """Write track to disk as YAML. Strips non-schema fields."""
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML is required to save tracks (pip install pyyaml).") from e

    gates = track.get("gate_track", [])
    clean = {
        "schema_version":  SCHEMA_VERSION,
        "track_name":      str(track.get("track_name", "")),
        "description":     str(track.get("description", "")),
        "elements_tested": list(track.get("elements_tested", [])),
        "gate_size":       float(track.get("gate_size",     DEFAULTS["gate_size"])),
        "bar_thickness":   float(track.get("bar_thickness", DEFAULTS["bar_thickness"])),
        "loop_closed":     bool(track.get("loop_closed",    DEFAULTS["loop_closed"])),
        "gate_track": [_gate_to_dict(g) for g in gates],
    }
    Path(path).write_text(yaml.dump(clean, default_flow_style=None, sort_keys=False))


def load_track(path: str | Path) -> dict:
    """Load and validate a track YAML from disk.

    Returns a dict with:
      gate_track   : list of gate dicts {position, rotation, auto_yaw}
      positions    : list[list[float]]  # [N, 3]
      orientations : list[list[float]]  # [N, 3] euler degrees [roll,pitch,yaw]
      gate_size, bar_thickness, loop_closed
    """
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML is required to load tracks (pip install pyyaml).") from e

    data = yaml.safe_load(Path(path).read_text())
    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported track schema_version: {sv!r} (expected {SCHEMA_VERSION})"
        )

    gate_track_raw = data.get("gate_track", [])
    gate_track: list[dict] = []
    for g in gate_track_raw:
        gate_track.append({
            "position": [float(v) for v in g.get("position", [0.0, 0.0, 1.5])],
            "rotation": [float(v) for v in g.get("rotation", [0.0, 0.0, 0.0])],
            "auto_yaw": bool(g.get("auto_yaw", True)),
        })

    # Extract flat lists for convenient access; convert to numpy on first use.
    positions:    list[list[float]] = [g["position"] for g in gate_track]
    orientations: list[list[float]] = [g["rotation"] for g in gate_track]

    return {
        "schema_version":  SCHEMA_VERSION,
        "track_name":      str(data.get("track_name", "")),
        "description":     str(data.get("description", "")),
        "elements_tested": list(data.get("elements_tested", [])),
        "gate_size":       float(data.get("gate_size",     DEFAULTS["gate_size"])),
        "bar_thickness":   float(data.get("bar_thickness", DEFAULTS["bar_thickness"])),
        "loop_closed":     bool(data.get("loop_closed",    DEFAULTS["loop_closed"])),
        "gate_track":      gate_track,
        "positions":       positions,
        "orientations":    orientations,
    }


# ---------------------------------------------------------------------------
# Spline utilities
# ---------------------------------------------------------------------------

def _build_spline(centres: np.ndarray, closed: bool):
    """Return (CubicSpline, knots, total_length) or None if degenerate."""
    from scipy.interpolate import CubicSpline

    centres = np.asarray(centres, dtype=float)
    if centres.shape[0] < 2:
        return None

    ext = np.vstack([centres, centres[:1]]) if closed else centres
    diffs = np.linalg.norm(np.diff(ext, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(diffs)])
    if u[-1] <= 0.0:
        return None

    bc = "periodic" if closed else "natural"
    try:
        cs = CubicSpline(u, ext, bc_type=bc, axis=0)
    except ValueError:
        cs = CubicSpline(u, ext, bc_type="natural", axis=0)
    return cs, u, float(u[-1])


def interpolate_loop(
    centres: np.ndarray,
    num_samples: int = 200,
    closed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the spline through ``centres`` at ``num_samples`` evenly-spaced points.

    Returns:
        waypoints: ``[num_samples, 3]``
        tangents:  ``[num_samples, 3]`` unit vectors
    """
    centres = np.asarray(centres, dtype=float)
    if centres.shape[0] < 2:
        return centres.copy(), np.zeros_like(centres)

    built = _build_spline(centres, closed)
    if built is None:
        return centres.copy(), np.zeros_like(centres)
    cs, u, total = built

    ts = np.linspace(0.0, total, num_samples, endpoint=not closed)
    pts = cs(ts)
    tng = cs(ts, 1)
    norms = np.linalg.norm(tng, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return pts, tng / norms


def gate_tangents_from_spline(
    centres: np.ndarray,
    closed: bool = True,
) -> np.ndarray:
    """Unit tangent at each input centre. Used for auto-yaw resolution."""
    centres = np.asarray(centres, dtype=float)
    if centres.shape[0] < 2:
        return np.zeros_like(centres)

    built = _build_spline(centres, closed)
    if built is None:
        return np.zeros_like(centres)
    cs, u, _ = built

    knots = u[:-1] if closed else u
    d = cs(knots, 1)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return d / norms


# ---------------------------------------------------------------------------
# Euler / quaternion conversion
# ---------------------------------------------------------------------------
# Convention: extrinsic XYZ (= ROS RPY) with angles [roll, pitch, yaw_deg]
#   roll  — about world +X
#   pitch — about world +Y (after roll in extrinsic)
#   yaw   — about world +Z, clockwise from +Y (0°→+Y, 90°→+X)
#
# Since scipy uses counter-clockwise conventions, yaw is negated on entry.

def euler_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """IsaacLab-order ``(w, x, y, z)`` quaternion for extrinsic XYZ Euler.

    Args:
        roll_deg:  rotation about +X, degrees
        pitch_deg: rotation about +Y, degrees
        yaw_deg:   clockwise rotation about +Z from +Y, degrees
    """
    from scipy.spatial.transform import Rotation as R
    qx, qy, qz, qw = R.from_euler(
        "xyz", [float(roll_deg), float(pitch_deg), -float(yaw_deg)], degrees=True
    ).as_quat()
    return np.array([qw, qx, qy, qz])


def quat_axes_wxyz(quats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply each ``(w,x,y,z)`` quaternion to the canonical basis.

    Returns ``(right, forward, up)`` — ``[N, 3]`` arrays of world-frame directions
    for gate-local ``+X``, ``+Y``, ``+Z``.
    """
    from scipy.spatial.transform import Rotation as R
    if quats.shape[0] == 0:
        z = np.zeros((0, 3))
        return z, z, z
    sci = quats[:, [1, 2, 3, 0]]   # wxyz → xyzw
    rot = R.from_quat(sci)
    return (
        rot.apply(np.array([1.0, 0.0, 0.0])),
        rot.apply(np.array([0.0, 1.0, 0.0])),
        rot.apply(np.array([0.0, 0.0, 1.0])),
    )


# ---------------------------------------------------------------------------
# Track → numpy arrays for the simulator
# ---------------------------------------------------------------------------

def track_to_arrays(track: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convert a track dict to ``(positions, quats_wxyz, meta)``.

    Converts ``positions: list[list]`` and ``orientations: list[list]`` to numpy
    arrays immediately for performance.  Auto-yaw gates resolve yaw against the
    closed-loop spline tangent; pitch and roll always come from the stored value.

    Returns:
        positions:  ``[N, 3]`` gate centres, world frame metres
        quats_wxyz: ``[N, 4]`` quaternions (w, x, y, z), IsaacLab order
        meta:       dict with gate_size, bar_thickness, loop_closed
    """
    gate_track = track.get("gate_track", [])
    meta = {
        "gate_size":     float(track.get("gate_size",     DEFAULTS["gate_size"])),
        "bar_thickness": float(track.get("bar_thickness", DEFAULTS["bar_thickness"])),
        "loop_closed":   bool(track.get("loop_closed",    DEFAULTS["loop_closed"])),
    }

    if not gate_track:
        return np.zeros((0, 3)), np.zeros((0, 4)), meta

    # Convert lists to numpy immediately.
    positions = np.array(track["positions"], dtype=float)
    rolls  = np.array([g["rotation"][0] for g in gate_track], dtype=float)
    pitches = np.array([g["rotation"][1] for g in gate_track], dtype=float)
    yaws   = np.array([g["rotation"][2] for g in gate_track], dtype=float)
    auto_mask = np.array([g.get("auto_yaw", True) for g in gate_track], dtype=bool)

    # Resolve auto-yaw: yaw_cw_deg = degrees(arctan2(tx, ty)).
    if auto_mask.any() and len(gate_track) >= 2:
        closed = meta["loop_closed"]
        tangents = gate_tangents_from_spline(positions, closed=closed)
        th = tangents.copy()
        th[:, 2] = 0.0
        n = np.linalg.norm(th, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        th /= n
        auto_yaws = np.degrees(np.arctan2(th[:, 0], th[:, 1]))
        yaws = np.where(auto_mask, auto_yaws, yaws)

    quats = np.stack(
        [euler_to_quat_wxyz(r, p, y) for r, p, y in zip(rolls, pitches, yaws)],
        axis=0,
    )
    return positions, quats, meta


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULTS",
    "default_track",
    "save_track",
    "load_track",
    "interpolate_loop",
    "gate_tangents_from_spline",
    "euler_to_quat_wxyz",
    "quat_axes_wxyz",
    "track_to_arrays",
]

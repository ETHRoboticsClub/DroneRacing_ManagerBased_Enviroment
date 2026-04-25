"""Track I/O utilities for the drone-racing custom track designer.

Pure Python (numpy + scipy + stdlib only). No Isaac Lab or torch imports, so
this module is safe to use both in the offline GUI designer and in any
play-side script that loads the saved track and feeds it to a trained policy.

Disk format (JSON, ``schema_version == 1``)::

    {
      "schema_version": 1,
      "gate_size":     0.7,
      "bar_thickness": 0.05,
      "loop_closed":   true,
      "gates": [
        {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0, "auto_yaw": true},
        ...
      ]
    }

Conventions
-----------
- World frame, metres. ``x``/``y``/``z`` are the gate centre coordinates.
- ``yaw`` is rotation about world ``+Z`` in **radians**. ``yaw = 0`` means the
  gate's passage direction (``+Y`` in gate-local frame) coincides with world
  ``+Y``. This matches racing_gates._place_gates_along_trajectory.
- Gates are always upright (no roll/pitch).
- ``auto_yaw=True`` overrides the stored yaw with the closed-loop spline
  tangent direction at that gate when the track is exported via
  ``track_to_arrays``.

The ``Gate`` ordering in the list is the order the drone visits them in
(matches ``RacingQuadcopterEnv._gate_positions`` insertion order).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

DEFAULTS: dict = {
    "gate_size": 0.7,
    "bar_thickness": 0.05,
    "loop_closed": True,
}


# ---------------------------------------------------------------------------
# Construction / persistence
# ---------------------------------------------------------------------------

def default_track() -> dict:
    """Return an empty track populated with default global parameters."""
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_size": DEFAULTS["gate_size"],
        "bar_thickness": DEFAULTS["bar_thickness"],
        "loop_closed": DEFAULTS["loop_closed"],
        "gates": [],
    }


def save_track(path: str | Path, track: dict) -> None:
    """Write track to disk as pretty-printed JSON. Strips non-schema fields."""
    clean = {
        "schema_version": SCHEMA_VERSION,
        "gate_size": float(track.get("gate_size", DEFAULTS["gate_size"])),
        "bar_thickness": float(track.get("bar_thickness", DEFAULTS["bar_thickness"])),
        "loop_closed": bool(track.get("loop_closed", DEFAULTS["loop_closed"])),
        "gates": [
            {
                "x": float(g["x"]),
                "y": float(g["y"]),
                "z": float(g["z"]),
                "yaw": float(g.get("yaw", 0.0)),
                "auto_yaw": bool(g.get("auto_yaw", True)),
            }
            for g in track.get("gates", [])
        ],
    }
    Path(path).write_text(json.dumps(clean, indent=2))


def load_track(path: str | Path) -> dict:
    """Load and validate a track JSON from disk. Back-fills optional fields."""
    data = json.loads(Path(path).read_text())
    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported track schema_version: {sv!r} (expected {SCHEMA_VERSION})"
        )
    for k, v in DEFAULTS.items():
        data.setdefault(k, v)
    for g in data.get("gates", []):
        g.setdefault("auto_yaw", True)
        g.setdefault("yaw", 0.0)
    return data


# ---------------------------------------------------------------------------
# Spline utilities
# ---------------------------------------------------------------------------
# Chord-length parameterised periodic CubicSpline. Mirrors the construction
# used by racing_gates._generate_random_track so the visual feedback in the
# designer matches what the simulator renders.

def _build_spline(centres: np.ndarray, closed: bool):
    """Return (CubicSpline, knot_array u, total_length) or None if degenerate."""
    from scipy.interpolate import CubicSpline

    centres = np.asarray(centres, dtype=float)
    if centres.shape[0] < 2:
        return None

    if closed:
        ext = np.vstack([centres, centres[:1]])
    else:
        ext = centres

    diffs = np.linalg.norm(np.diff(ext, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(diffs)])
    if u[-1] <= 0.0:
        return None

    bc = "periodic" if closed else "natural"
    try:
        cs = CubicSpline(u, ext, bc_type=bc, axis=0)
    except ValueError:
        # Periodic BC requires identical endpoints (we ensured that) AND
        # at least 3 knots; if it still fails, fall back to natural BC.
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
        tangents:  ``[num_samples, 3]`` unit vectors (zero where degenerate)

    For < 2 input centres returns the input as-is and zero tangents.
    """
    centres = np.asarray(centres, dtype=float)
    if centres.shape[0] < 2:
        return centres.copy(), np.zeros_like(centres)

    built = _build_spline(centres, closed)
    if built is None:
        return centres.copy(), np.zeros_like(centres)
    cs, u, total = built

    # endpoint=False on closed loops avoids a duplicated sample at the seam.
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
    """Unit tangent at each input centre. Used for auto-yaw."""
    centres = np.asarray(centres, dtype=float)
    if centres.shape[0] < 2:
        return np.zeros_like(centres)

    built = _build_spline(centres, closed)
    if built is None:
        return np.zeros_like(centres)
    cs, u, _ = built

    # On closed loops u has N+1 knots (last duplicates first); drop the duplicate.
    knots = u[:-1] if closed else u
    d = cs(knots, 1)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return d / norms


# ---------------------------------------------------------------------------
# Yaw / quaternion conversion
# ---------------------------------------------------------------------------
# Convention recap: an upright gate with yaw ψ has its passage direction
# (gate-local +Y) mapped to world (-sin ψ, cos ψ, 0).  Inverting:
#     ψ = atan2(-tx, ty)
# where (tx, ty) is the desired horizontal forward direction.

def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """IsaacLab-order (w, x, y, z) quaternion for an upright gate."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def yaw_from_horizontal_tangent(tx: float, ty: float) -> float:
    """Yaw angle such that gate-local +Y aligns with horizontal direction (tx, ty)."""
    return float(np.arctan2(-tx, ty))


# ---------------------------------------------------------------------------
# Track → numpy arrays for the simulator
# ---------------------------------------------------------------------------

def track_to_arrays(track: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convert a track dict to ``(positions, quats_wxyz, meta)``.

    Positions are world frame metres ``[N, 3]``. Quaternions are IsaacLab
    order ``(w, x, y, z)`` ``[N, 4]``.  Auto-yaw gates are resolved against
    the closed-loop spline tangent here (manual yaws pass through unchanged).

    For an empty track returns shape ``(0, 3)`` and ``(0, 4)`` arrays.
    """
    gates = track.get("gates", [])
    meta = {k: track.get(k, v) for k, v in DEFAULTS.items()}

    if len(gates) == 0:
        return np.zeros((0, 3)), np.zeros((0, 4)), meta

    positions = np.array([[g["x"], g["y"], g["z"]] for g in gates], dtype=float)
    yaws = np.array([float(g.get("yaw", 0.0)) for g in gates])
    auto_mask = np.array([bool(g.get("auto_yaw", True)) for g in gates])

    # Resolve auto-yaw against the spline tangent (horizontal projection).
    if auto_mask.any() and len(gates) >= 2:
        closed = bool(meta["loop_closed"])
        tangents = gate_tangents_from_spline(positions, closed=closed)
        th = tangents.copy()
        th[:, 2] = 0.0
        n = np.linalg.norm(th, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        th /= n
        auto_yaws = np.arctan2(-th[:, 0], th[:, 1])
        yaws = np.where(auto_mask, auto_yaws, yaws)

    quats = np.stack([yaw_to_quat_wxyz(y) for y in yaws], axis=0)
    return positions, quats, meta


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULTS",
    "default_track",
    "save_track",
    "load_track",
    "interpolate_loop",
    "gate_tangents_from_spline",
    "yaw_to_quat_wxyz",
    "yaw_from_horizontal_tangent",
    "track_to_arrays",
]

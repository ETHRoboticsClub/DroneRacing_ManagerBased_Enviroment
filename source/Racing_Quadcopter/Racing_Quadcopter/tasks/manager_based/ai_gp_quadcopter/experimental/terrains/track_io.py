# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""YAML <-> :class:`GateSpec` conversion for fixed-layout racing tracks.

Track YAML format
-----------------
A track YAML is either a top-level list of gate entries, or a mapping with a
``gate_track`` key holding that list. Each entry has::

    - position: [x, y, z]            # metres, in the sub-terrain's local frame
      rotation: [roll, pitch, yaw]   # degrees, ROS/URDF RPY order

``rotation`` may be omitted (treated as zero) or specified as 1 element (yaw
only) or 3 elements (full RPY).
"""

from __future__ import annotations

from pathlib import Path

from .racing_gates_cfg import GateSpec


def _parse_rotation(raw) -> tuple[float, float, float]:
    if raw is None:
        return 0.0, 0.0, 0.0
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"'rotation' must be a list/tuple of 1 or 3 numbers, got: {raw!r}")
    if len(raw) == 1:
        return 0.0, 0.0, float(raw[0])
    if len(raw) == 3:
        return float(raw[0]), float(raw[1]), float(raw[2])
    raise ValueError(f"'rotation' must have 1 or 3 elements, got {len(raw)}: {raw!r}")


def load_gates_from_yaml(path: str | Path) -> list[GateSpec]:
    """Load a list of :class:`GateSpec` from a track YAML file.

    Accepts either a top-level list or a mapping with a ``gate_track`` key.
    """
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML is required to load track YAMLs (pip install pyyaml).") from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Track YAML not found: {p}")

    data = yaml.safe_load(p.read_text())
    if isinstance(data, dict):
        if "gate_track" not in data:
            raise ValueError(f"Track YAML mapping must contain a 'gate_track' key: {p}")
        entries = data["gate_track"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Track YAML must be a list or a mapping, got {type(data).__name__}: {p}")

    if not entries:
        raise ValueError(f"Track YAML contains no gates: {p}")

    gates: list[GateSpec] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "position" not in entry:
            raise ValueError(f"Gate {i} in {p} must be a mapping with a 'position' key, got: {entry!r}")
        pos = entry["position"]
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            raise ValueError(f"Gate {i} in {p}: 'position' must be a list of 3 numbers, got: {pos!r}")
        roll, pitch, yaw = _parse_rotation(entry.get("rotation"))
        gates.append(
            GateSpec(
                position=[float(pos[0]), float(pos[1]), float(pos[2])],
                rotation=[roll, pitch, yaw],
            )
        )
    return gates

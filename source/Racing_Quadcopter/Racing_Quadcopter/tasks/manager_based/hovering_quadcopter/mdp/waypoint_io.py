# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""YAML loader for hovering-task waypoint tracks.

Format (configs/waypoint_tracks/*.yaml)::

    collection_time_s: 1.0   # optional, falls back to cfg.collection_time_s
    waypoint_track:
      - position: [x, y, z]
      - position: [x, y, z]

Distinct from the racing track loader: the top-level key is ``waypoint_track``
(NOT ``gate_track``) and there is no orientation field. Files using the
racing key are rejected explicitly to prevent accidental mixing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_waypoints_from_yaml(path: str | Path) -> tuple[np.ndarray, float | None]:
    """Load a waypoint track YAML.

    Returns:
        waypoints:           [G, 3] positions in cell-local frame (metres).
        collection_time_s:   The track's T_c if specified in the YAML, else ``None``
                             (caller substitutes the cfg default).
    """
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML is required to load waypoint YAMLs.") from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Waypoint YAML not found: {p}")

    data = yaml.safe_load(p.read_text())
    if isinstance(data, dict):
        if "gate_track" in data:
            raise ValueError(
                f"{p}: contains 'gate_track' — this is a racing track. "
                "Use a 'waypoint_track' YAML for the hovering env."
            )
        if "waypoint_track" not in data:
            raise ValueError(f"{p}: mapping must contain a 'waypoint_track' key.")
        entries = data["waypoint_track"]
        tc = data.get("collection_time_s", None)
    elif isinstance(data, list):
        entries = data
        tc = None
    else:
        raise ValueError(f"{p}: YAML root must be a list or mapping, got {type(data).__name__}.")

    if not entries:
        raise ValueError(f"{p}: contains no waypoints.")

    positions: list[list[float]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "position" not in entry:
            raise ValueError(f"{p}: waypoint {i} must be a mapping with a 'position' key.")
        pos = entry["position"]
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            raise ValueError(f"{p}: waypoint {i} 'position' must be a 3-element list.")
        positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

    waypoints = np.array(positions, dtype=np.float64)
    tc_out: float | None = float(tc) if tc is not None else None
    return waypoints, tc_out
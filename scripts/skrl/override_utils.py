# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI override helpers for skrl train/play scripts.

This module centralizes env_cfg override parsing so train.py and play.py
behave identically during sweeps.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence


def _parse_override_value(raw: str):
    """Parse CLI override value with Python-literal support.

    Examples:
    - "10" -> 10
    - "0.02" -> 0.02
    - "True" -> True
    - "None" -> None
    - "(8.0, 15.0)" -> (8.0, 15.0)
    - "[1, 2, 3]" -> [1, 2, 3]
    - "{'a': 1}" -> {'a': 1}
    - "cuda:0" -> "cuda:0" (fallback to raw string)
    """
    lower = raw.lower()
    if lower in ("none", "null"):
        return None
    if lower in ("true", "false"):
        return lower == "true"

    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _coerce_override_value(current, parsed, key_path: str):
    """Coerce a parsed value to match the current field type when possible."""
    if parsed is None:
        return None

    if isinstance(current, bool):
        if isinstance(parsed, bool):
            return parsed
        if isinstance(parsed, str):
            return parsed.lower() not in ("0", "false", "no")
        return bool(parsed)

    if current is None:
        return parsed

    if isinstance(current, tuple):
        if isinstance(parsed, tuple):
            return parsed
        if isinstance(parsed, list):
            return tuple(parsed)
        raise ValueError(f"Override '{key_path}' expects tuple-like value; got {type(parsed).__name__}: {parsed}")

    if isinstance(current, list):
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        raise ValueError(f"Override '{key_path}' expects list-like value; got {type(parsed).__name__}: {parsed}")

    if isinstance(current, dict):
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Override '{key_path}' expects dict value; got {type(parsed).__name__}: {parsed}")

    # Scalar / enum-like / string-like fallback.
    try:
        return type(current)(parsed)
    except Exception as e:
        raise ValueError(
            f"Failed to cast override '{key_path}' value {parsed!r} to {type(current).__name__} "
            f"(current={current!r}): {e}"
        ) from e


def apply_overrides(cfg, overrides: Sequence[str]) -> None:
    """Apply KEY=VAL overrides to env_cfg.

    Supports dotted attribute paths (e.g. ``scene.num_envs=512``) and Python
    literals for complex types (e.g. tuples/lists/dicts/None).
    """
    for item in overrides:
        key, sep, raw_val = item.partition("=")
        if not sep:
            raise ValueError(f"Malformed override '{item}'. Expected KEY=VAL.")

        keys = key.split(".")
        obj = cfg
        for k in keys[:-1]:
            if not hasattr(obj, k):
                raise AttributeError(f"Unknown override path '{key}': missing attribute '{k}'.")
            obj = getattr(obj, k)

        attr = keys[-1]
        if not hasattr(obj, attr):
            raise AttributeError(f"Unknown override key '{key}': attribute '{attr}' not found.")

        current = getattr(obj, attr)
        parsed = _parse_override_value(raw_val)
        casted = _coerce_override_value(current, parsed, key)
        setattr(obj, attr, casted)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static red-sphere waypoint visualisation.

Draws one red ``UsdGeom.Sphere`` per waypoint of every cell. The translations
never change (waypoint tracks are fixed once generated), so we publish them
once at env init and never touch them again.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


_WAYPOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Waypoints",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
    },
)


def draw_waypoints(positions_world: np.ndarray, device: str | torch.device) -> VisualizationMarkers:
    """Spawn red sphere markers at every supplied world-frame position.

    Args:
        positions_world: ``[M, 3]`` array of positions across all cells, in
            metres, world frame.
        device: torch device that the env runs on.

    Returns:
        The created :class:`VisualizationMarkers` handle (kept alive by the
        env so the prim isn't garbage-collected).
    """
    if positions_world.size == 0:
        # Still construct the prim so subsequent calls find a valid handle,
        # but skip the visualize step to avoid an empty-translations call.
        return VisualizationMarkers(_WAYPOINT_MARKER_CFG)

    marker = VisualizationMarkers(_WAYPOINT_MARKER_CFG)
    translations = torch.as_tensor(positions_world, device=device, dtype=torch.float32)
    marker.visualize(translations=translations)
    return marker
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn one drone with **no gravity** and **HDRI sky** (image-based dome lighting).

Body **+Z** (red) and **+X** (green) axis lines (0.5 m) are drawn from the robot root.

.. code-block:: bash

    python scripts/tools/view_drone.py --drone crazyflie
    python scripts/tools/view_drone.py --drone anemos_urdf \\
        --sky_hdr 'omniverse://localhost/NVIDIA/Assets/Skies/...'

Override ``--sky_hdr`` if your Nucleus layout differs. ``racing_quadcopter_env_cfg`` must import;
unset invalid ``$DRONE_MODEL``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import MISSING

from isaaclab.app import AppLauncher

_DRONE_CHOICES = ("crazyflie", "anemos_usd", "anemos_urdf")

_DEFAULT_SKY_HDR_REL = (
    "Materials/Textures/Skies/PolyHaven/kloofendal_48d_partly_cloudy_4k.hdr"
)

# Line from body origin along +axis (cylinder height axis = local +Z).
_AXIS_LINE_LEN_M = 0.5
_AXIS_LINE_RADIUS_M = 0.003

parser = argparse.ArgumentParser(description="View drone with HDRI sky (no gravity).")
parser.add_argument(
    "--drone",
    type=str,
    choices=_DRONE_CHOICES,
    default="crazyflie",
    help="Robot profile from DRONE_PROFILES (same assets as the racing env).",
)
parser.add_argument(
    "--sky_hdr",
    type=str,
    default=None,
    help=(
        "Full path or omniverse:// URI to a lat-long HDR/EXR for the dome. "
        "Default: ISAAC_NUCLEUS_DIR / " + _DEFAULT_SKY_HDR_REL
    ),
)
parser.add_argument(
    "--dome_intensity",
    type=float,
    default=800.0,
    help="Linear intensity scale for the HDRI dome (tune if scene is too dark/bright).",
)
parser.add_argument(
    "--emissive",
    action="store_true",
    help="Tint robot meshes with a weak emissive (helps tiny models; off by default for natural IBL).",
)
parser.add_argument(
    "--no_axes",
    action="store_true",
    help="Disable body axis line visualization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after simulation app is live."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

import Racing_Quadcopter.tasks  # noqa: F401
from Racing_Quadcopter.tasks.manager_based.racing_quadcopter.racing_quadcopter_env_cfg import (
    DRONE_PROFILES,
)
from Racing_Quadcopter.tasks.manager_based.racing_quadcopter.vis.trajectory_vis import (
    apply_emissive_to_robots,
)

_BASE_DOME_SPAWN = sim_utils.DomeLightCfg(
    intensity=800.0,
    color=(1.0, 1.0, 1.0),
    texture_file=None,
    texture_format="latlong",
    visible_in_primary_ray=True,
)


@configclass
class DroneViewSceneCfg(InteractiveSceneCfg):
    """Ground plane + HDRI dome; no extra distant light (sky provides sun)."""

    replicate_physics: bool = False
    clone_in_fabric: bool = False

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.32, 0.26)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=_BASE_DOME_SPAWN)


def _line_marker(prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "line": sim_utils.CylinderCfg(
                radius=_AXIS_LINE_RADIUS_M,
                height=_AXIS_LINE_LEN_M,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _make_body_axis_lines(
    device: str,
) -> tuple[VisualizationMarkers, VisualizationMarkers, torch.Tensor, torch.Tensor]:
    """Red +Z and green +X lines from the robot root (0.5 m each)."""
    z_marker = _line_marker("/World/Visuals/body_axis_z", (1.0, 0.0, 0.0))
    x_marker = _line_marker("/World/Visuals/body_axis_x", (0.0, 1.0, 0.0))

    half = _AXIS_LINE_LEN_M * 0.5
    offset_z_b = torch.tensor([[0.0, 0.0, half]], device=device)
    offset_x_b = torch.tensor([[half, 0.0, 0.0]], device=device)

    # Cylinder +Z -> body +X: rotate -90 deg about body Y.
    quat_cyl_to_body_x = quat_from_euler_xyz(
        torch.zeros(1, device=device),
        torch.tensor([-math.pi / 2.0], device=device),
        torch.zeros(1, device=device),
    )
    return z_marker, x_marker, offset_z_b, offset_x_b, quat_cyl_to_body_x


def _update_body_axis_lines(
    robot: Articulation,
    z_marker: VisualizationMarkers,
    x_marker: VisualizationMarkers,
    offset_z_b: torch.Tensor,
    offset_x_b: torch.Tensor,
    quat_cyl_to_body_x: torch.Tensor,
) -> None:
    pos_w = robot.data.root_pos_w
    quat_w = robot.data.root_quat_w

    mid_z = pos_w + quat_apply(quat_w, offset_z_b)
    mid_x = pos_w + quat_apply(quat_w, offset_x_b)

    z_marker.visualize(translations=mid_z, orientations=quat_w)
    x_marker.visualize(translations=mid_x, orientations=quat_mul(quat_w, quat_cyl_to_body_x))


def main() -> None:
    profile = DRONE_PROFILES[args_cli.drone]
    robot_cfg = profile.robot_cfg

    sky_path = args_cli.sky_hdr
    if not sky_path:
        sky_path = f"{ISAAC_NUCLEUS_DIR}/{_DEFAULT_SKY_HDR_REL}"

    dome_spawn = _BASE_DOME_SPAWN.replace(intensity=args_cli.dome_intensity, texture_file=sky_path)

    scene_cfg = DroneViewSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        robot=robot_cfg,
        dome_light=AssetBaseCfg(
            prim_path="/World/Light",
            spawn=dome_spawn,
        ),
    )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
        gravity=(0.0, 0.0, 0.0),
        render_interval=1,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.6, 2.6, 1.1], target=[0.0, 0.0, 0.22])

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    if args_cli.emissive:
        apply_emissive_to_robots(
            color=(0.88, 0.9, 0.95),
            intensity=2.5,
            robots_root="/World/envs",
        )

    robot: Articulation = scene["robot"]
    axis_lines: tuple[VisualizationMarkers, VisualizationMarkers, torch.Tensor, torch.Tensor, torch.Tensor] | None = (
        None
    )
    if not args_cli.no_axes:
        axis_lines = _make_body_axis_lines(sim.device)

    print(
        f"[INFO] Drone={args_cli.drone!r}  gravity=off  sky HDR={sky_path}  "
        f"dome_intensity={args_cli.dome_intensity}  "
        f"axes={'off' if args_cli.no_axes else f'+Z red / +X green ({_AXIS_LINE_LEN_M} m lines)'}"
    )

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if axis_lines is not None:
            _update_body_axis_lines(robot, axis_lines[0], axis_lines[1], axis_lines[2], axis_lines[3], axis_lines[4])


if __name__ == "__main__":
    main()
    simulation_app.close()

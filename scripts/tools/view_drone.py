# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn one drone with **no gravity** and **HDRI sky** (image-based dome lighting).

Axis frames are not drawn here — use the manager-based env / Isaac Lab debug if you need them.

.. code-block:: bash

    python scripts/tools/view_drone.py --drone crazyflie
    python scripts/tools/view_drone.py --drone anemos_urdf \\
        --sky_hdr 'omniverse://localhost/NVIDIA/Assets/Skies/...'

Override ``--sky_hdr`` if your Nucleus layout differs. ``racing_quadcopter_env_cfg`` must import;
unset invalid ``$DRONE_MODEL``.
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING

from isaaclab.app import AppLauncher

_DRONE_CHOICES = ("crazyflie", "anemos_usd", "anemos_urdf")

# Poly Haven lat-long HDR shipped with Isaac Nucleus (path may vary slightly by Sim version).
_DEFAULT_SKY_HDR_REL = (
    "Materials/Textures/Skies/PolyHaven/kloofendal_48d_partly_cloudy_4k.hdr"
)

parser = argparse.ArgumentParser(description="View drone with HDRI sky (no gravity).")
parser.add_argument(
    "--drone",
    type=str,
    choices=_DRONE_CHOICES,
    default="anemos_urdf",
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after simulation app is live."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass

import Racing_Quadcopter.tasks  # noqa: F401
from Racing_Quadcopter.tasks.manager_based.racing_quadcopter.racing_quadcopter_env_cfg import (
    DRONE_PROFILES,
)
from Racing_Quadcopter.tasks.manager_based.racing_quadcopter.vis.trajectory_vis import (
    apply_emissive_to_robots,
)

# Shared template — InteractiveSceneCfg subclasses do not keep arbitrary attrs on the *class* object
# (so do not read ``DroneViewSceneCfg.dome_light`` from the type).
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

    print(
        f"[INFO] Drone={args_cli.drone!r}  gravity=off  sky HDR={sky_path}  "
        f"dome_intensity={args_cli.dome_intensity}"
    )

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()

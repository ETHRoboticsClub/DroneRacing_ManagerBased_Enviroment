"""Anemos robot config for the frozen AI-GP experimental snapshot.

Copied from ``Racing_Quadcopter/assets/robots_cfg.py`` (2026-09-26). Only the URDF config this
variant uses is kept; the URDF and its meshes live next to this file.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANEMOS_DIR = _THIS_DIR

##
# Configuration
##

ANEMOS_DRONE_CFG_URDF = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(_ANEMOS_DIR, "anemos.urdf"),
        force_usd_conversion=True,
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

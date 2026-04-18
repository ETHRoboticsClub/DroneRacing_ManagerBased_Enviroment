# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter manager env configuration (standalone)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from Racing_Quadcopter.terrains.racing_terrain import create_racing_gates_terrain
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs.mdp import *

from isaaclab_assets import CRAZYFLIE_CFG

from .mdp.actions import QuadcopterThrustActionCfg
from .mdp.events import reset_racing_spawn
from .mdp.observations import _compute_next_gates_rel_and_fwd_body, velocity_limit_obs
from .mdp import rewards as racing_rewards
from .mdp.terminations import collision_termination, lap_complete, runaway_velocity, upside_down



@configclass
class GateObservationConfig:
    """Observation parameters."""

    num_next_gates: int = 3
    include_velocity_limit: bool = True


@configclass
class RacingQuadcopterSceneCfg(InteractiveSceneCfg):
    """Scene: racing-gates terrain, Crazyflie, contact sensor, dome light."""

    terrain: TerrainImporterCfg | None = None

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot").replace(
        spawn=CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True)
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
        update_period=0.01,
        track_air_time=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class ActionsCfg:
    """Thrust + moments on the quadcopter base."""

    thrust = QuadcopterThrustActionCfg(asset_name="robot", thrust_to_weight=1.9, moment_scale=0.01, action_dim=4)


@configclass
class ObservationsCfg:
    """Policy observations."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        history_length: int = 3
        base_lin_vel = ObsTerm(
            func=base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=history_length,
        )
        
        #IMU 
        base_ang_vel = ObsTerm(
            func=base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=history_length,
        )
        projected_gravity = ObsTerm(
            func=projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=history_length,
        )

        #ROBOT
        #TODO: Add motors speeds

        #ACTION
        last_action = ObsTerm(func=last_action, history_length=history_length)


        #Perception
        obs_gates = ObsTerm(
            func=_compute_next_gates_rel_and_fwd_body,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=history_length,
        )
        velocity_limit = ObsTerm(func=velocity_limit_obs, history_length=history_length)
        



        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Same terms as direct ``GateRewardManager``; each callable returns unweighted factors in ``mdp.rewards``.

    Weights are ``MANAGER_SCALE *`` the default coefficients from the direct task.
    """

    gate_passage = RewTerm(func=racing_rewards.reward_gate_passage, weight=racing_rewards.WEIGHT_GATE_PASSAGE)
    gate_progress = RewTerm(func=racing_rewards.reward_gate_progress, weight=racing_rewards.WEIGHT_GATE_PROGRESS)
    gate_centering = RewTerm(func=racing_rewards.reward_gate_centering, weight=racing_rewards.WEIGHT_GATE_CENTERING)
    velocity_toward_gate = RewTerm(
        func=racing_rewards.reward_velocity_toward_gate, weight=racing_rewards.WEIGHT_VELOCITY_TOWARD_GATE
    )
    velocity_limit = RewTerm(func=racing_rewards.penalty_velocity_limit, weight=racing_rewards.WEIGHT_VELOCITY_LIMIT)
    # perception = RewTerm(func=racing_rewards.penalty_perception, weight=racing_rewards.WEIGHT_PERCEPTION)
    orientation = RewTerm(func=racing_rewards.penalty_orientation, weight=racing_rewards.WEIGHT_ORIENTATION)
    angular_velocity = RewTerm(
        func=racing_rewards.penalty_angular_velocity, weight=racing_rewards.WEIGHT_ANGULAR_VELOCITY
    )
    action_smoothness = RewTerm(
        func=racing_rewards.penalty_action_smoothness, weight=racing_rewards.WEIGHT_ACTION_SMOOTHNESS
    )
    action_magnitude = RewTerm(
        func=racing_rewards.penalty_action_magnitude, weight=racing_rewards.WEIGHT_ACTION_MAGNITUDE
    )
    crash = RewTerm(func=racing_rewards.penalty_crash, weight=racing_rewards.WEIGHT_CRASH)
    wrong_side = RewTerm(func=racing_rewards.penalty_wrong_side, weight=racing_rewards.WEIGHT_WRONG_SIDE)


@configclass
class EventsCfg:
    """Reset pipeline: Isaac scene defaults, then racing start pose and buffers."""

    randomize_robot_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_robot_actuator_gains = EventTerm(
        func=randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_scene_to_default = EventTerm(func=reset_scene_to_default, mode="reset")
    reset_racing_spawn = EventTerm(func=reset_racing_spawn, mode="reset", params={"method": "uniform"})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    upside_down = DoneTerm(func=upside_down, time_out=False)
    runaway = DoneTerm(func=runaway_velocity, time_out=False)
    collision = DoneTerm(func=collision_termination, time_out=False)
    lap_complete = DoneTerm(func=lap_complete, time_out=True)


@configclass
class RacingQuadcopterEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for manager-based racing quadcopter."""

    seed: int = 42

    scene: RacingQuadcopterSceneCfg = RacingQuadcopterSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    observation: GateObservationConfig = GateObservationConfig()

    decimation: int = 2
    episode_length_s: float = 20.0

    velocity_limit: float = 15.0
    randomize_velocity_limit: bool = False
    velocity_limit_range: tuple[float, float] = (8.0, 15.0)

    upside_down_threshold: float = 0.7
    max_velocity: float = 50.0
    collision_force_threshold: float = 0.001

    eval_mode: bool = False
    draw_trajectory: bool = True        # green closed-loop polyline through all waypoints (USD, no collision)
    draw_gate_indicators: bool = True   # blue sticks showing each gate's forward (+Y) direction (USD, no collision)
    draw_gate_labels: bool = True       # blue 7-segment gate index numbers floating above each gate (USD, no collision)

    # Demo mode: dark scene + OmniPBR emissive glow on all vis prims and drones (RTX only).
    demo_mode: bool = False
    demo_emissive_intensity: float = 30.0        # emissive intensity for trajectory/label/indicator vis prims
    demo_gate_color: tuple[float, float, float] = (1.0, 0.3, 0.0)   # orange, matching gate mesh color
    demo_gate_outline_emissive: float = 20.0     # emissive intensity for gate outline overlay curves
    demo_gate_outline_width: float = 0.06        # tube width of gate outline BasisCurves in metres
    demo_gate_light_intensity: float = 3000.0    # UsdLux.SphereLight intensity in nits
    demo_gate_light_radius: float = 1.5          # sphere light radius in metres — controls glow spread
    demo_drone_glow_color: tuple[float, float, float] = (1.0, 0.5, 0.0)  # orange drone glow
    demo_drone_glow_intensity: float = 5.0

    curriculum_min_gates_passed: int = 10
    curriculum_min_mean_velocity: float = 2.5
    curriculum_max_center_offset: float = 0.25

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    viewer: ViewerCfg = ViewerCfg(eye=(15.0, 15.0, 7.0), lookat=(3.0, 3.0, 0.0), origin_type="world")

    def __post_init__(self) -> None:
        terrain_generator = create_racing_gates_terrain(self.seed)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_generator,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=True,
        )
        self.sim.render_interval = self.decimation

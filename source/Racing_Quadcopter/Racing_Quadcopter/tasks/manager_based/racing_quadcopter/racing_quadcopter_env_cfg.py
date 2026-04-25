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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from Racing_Quadcopter.terrains import GateSpec
from Racing_Quadcopter.terrains.racing_terrain import create_fixed_track_terrain, create_racing_gates_terrain
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs.mdp import *

from isaaclab_assets import CRAZYFLIE_CFG

from .mdp.actions import QuadcopterThrustActionCfg
from .mdp.curriculums import apply_staged_curriculum
from .mdp.events import reset_racing_spawn
from .mdp.observations import (
    compute_next_gates_rel_and_fwd_body,
    velocity_limit_obs,
    wrap_base_lin_vel,
    wrap_base_ang_vel,
    wrap_projected_gravity,
    wrap_last_action,
)
from .mdp import rewards as racing_rewards
from .mdp.terminations import below_height_threshold, collision_termination, lap_complete, runaway_velocity, upside_down



@configclass
class GateObservationConfig:
    """Observation parameters."""

    num_next_gates: int = 3
    include_velocity_limit: bool = True


@configclass
class ObservationNoiseCfg:
    """Config for corrupting ground-truth observations to simulate state estimation.

    Keep defaults off; enable selectively in experiments. The wrappers in
    `mdp/observations.py` read this config to apply per-step Gaussian noise,
    optional per-episode bias, and dropout.
    """

    # Conservative training defaults: enabled in training, disabled in *_PLAY configs.
    enable: bool = True
    per_episode_bias: bool = True

    # default std if term-specific not set
    gaussian_std_default: float = 0.0
    gaussian_std_base_lin_vel: float = 0.0
    gaussian_std_base_ang_vel: float = 0.0
    gaussian_std_projected_gravity: float = 0.0
    gaussian_std_obs_gates: float = 0.0
    gaussian_std_velocity_limit: float = 0.0
    gaussian_std_last_action: float = 0.0

    # per-episode bias ranges (uniform sampling [min, max])
    bias_range_base_lin_vel: tuple[float, float] = (-0.0, 0.0)
    bias_range_base_ang_vel: tuple[float, float] = (-0.0, 0.0)
    bias_range_projected_gravity: tuple[float, float] = (-0.0, 0.0)
    bias_range_obs_gates: tuple[float, float] = (-0.0, 0.0)
    bias_range_velocity_limit: tuple[float, float] = (-0.0, 0.0)
    bias_range_last_action: tuple[float, float] = (-0.0, 0.0)

    # dropout probability (per-term, per-step)
    dropout_prob: float = 0.0


@configclass
class ActionNoiseCfg:
    """Config for corrupting actions / simulating actuator noise/lag.

    - gaussian_std: per-step additive noise on actions
    - per_episode_bias: sampled bias added to actions each episode
    - lpf_alpha: first-order low-pass applied to actions (1.0=no smoothing)
    """

    # Conservative training defaults: enabled in training, disabled in *_PLAY configs.
    enable: bool = True
    per_episode_bias: bool = True
    gaussian_std: float = 0.0
    bias_range: tuple[float, float] = (-0.0, 0.0)
    lpf_alpha: float = 1.0 # No lag; set <1.0 to simulate actuator lag.
    clip_actions: bool = True


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
            func=wrap_base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=history_length,
        )
        
        #IMU 
        base_ang_vel = ObsTerm(
            func=wrap_base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=history_length,
        )
        projected_gravity = ObsTerm(
            func=wrap_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=history_length,
        )

        #ROBOT
        #TODO: Add motors speeds

        #ACTION
        last_action = ObsTerm(func=wrap_last_action, history_length=history_length)


        #Perception
        obs_gates = ObsTerm(
            func=compute_next_gates_rel_and_fwd_body,
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
class CurriculumCfg:
    """Staged curriculum: stay-alive → full racing → recovery spawns."""

    staged = CurrTerm(func=apply_staged_curriculum)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    upside_down = DoneTerm(func=upside_down, time_out=False)
    below_height = DoneTerm(func=below_height_threshold, time_out=False)
    runaway = DoneTerm(func=runaway_velocity, time_out=False)
    collision = DoneTerm(func=collision_termination, time_out=False)
    lap_complete = DoneTerm(func=lap_complete, time_out=True)


@configclass
class RacingQuadcopterEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for manager-based racing quadcopter."""

    seed: int = 42

    scene: RacingQuadcopterSceneCfg = RacingQuadcopterSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    observation: GateObservationConfig = GateObservationConfig()
    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()
    action_noise: ActionNoiseCfg = ActionNoiseCfg()

    decimation: int = 2
    episode_length_s: float = 20.0

    velocity_limit: float = 15.0
    randomize_velocity_limit: bool = False
    velocity_limit_range: tuple[float, float] = (8.0, 15.0)

    upside_down_threshold: float = 0.7
    min_height: float = 0.2
    max_velocity: float = 50.0
    collision_force_threshold: float = 0.001

    eval_mode: bool = False
    crash_termination: bool = True   # master switch; stage gate (>= 2) applies on top in training
    fixed_spawn: bool = False        # if True, all drones spawn at gate 0 instead of a random gate
    recovery_spawn_prob: float = 0.2 # fraction [0,1] of envs that spawn with random orientation + velocity
    speed_log_interval_s: float = 10.0
    stats_warmup_s: float = 0.0    # skip stats accumulation for this many simulation seconds at episode start
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

    curriculum_stage1_steps: int = 10000
    curriculum_stage2_steps: int = 10000
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


@configclass
class RacingQuadcopterEnvCfg_PLAY(RacingQuadcopterEnvCfg):
    """Evaluation config: eval_mode on, fixed spawn at gate 0, no randomisation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.eval_mode = True
        self.fixed_spawn = True
        self.observation_noise.enable = False
        self.action_noise.enable = False
        self.events.randomize_robot_mass = None
        self.events.randomize_robot_actuator_gains = None
        self.stats_warmup_s = 30.0
        print("[INFO] Evaluation mode is ON.")


# ---------------------------------------------------------------------------
# Fixed-track variants
# ---------------------------------------------------------------------------
# Replace FIXED_GATES with your own circuit.  Positions are in the sub-terrain's
# corner-origin local frame (a 15×15 m tile runs from (0,0) to (15,15) in XY).
# yaw_deg is clockwise from +Y: 0° → drone passes in +Y, 90° → in +X.

FIXED_GATES: list[GateSpec] = [
    # 0 — low start, heading north
    GateSpec(position=(7.5,  2.0, 1.2), yaw_deg=0),

    # 1-3 — climbing right sweep toward the top of the tile
    GateSpec(position=(12.5, 4.5, 2.5), yaw_deg=65),
    GateSpec(position=(13.5, 9.0, 4.5), yaw_deg=105),
    GateSpec(position=(10.5, 13.0, 4.5), yaw_deg=160),

    # 4-5 — SPLIT-S
    # Gate 4: enter heading south at 4.2 m altitude.
    # Drone rolls inverted, dives, and exits gate 5 heading north at 1.5 m
    # (2.7 m altitude drop, ~1.5 m forward — extremely tight).
    GateSpec(position=(7.5, 13.5, 4.2), yaw_deg=180),  # entry
    GateSpec(position=(7.5, 12.0, 1.5), yaw_deg=0),    # exit

    # 6-7 — left-side swoop
    GateSpec(position=(4.0, 10.5, 2.0), yaw_deg=270),
    GateSpec(position=(1.8,  7.5, 1.5), yaw_deg=230),

    # 8-9 — CHICANE (offset gates, tight lateral snap)
    GateSpec(position=(3.5,  3.5, 1.8), yaw_deg=350),
    GateSpec(position=(5.5,  2.0, 2.5), yaw_deg=25),
]

FIXED_GATES_: list[GateSpec] = [
    # 0 — low start, heading north
    #GateSpec(position=(7.5,  2.0, 1.2), yaw_deg=0),

    # 1-3 — climbing right sweep toward the top of the tile
    GateSpec(position=(7.5, 5.0, 4.5), yaw_deg=45),
    GateSpec(position=(7.5, 7.5, 4.5), yaw_deg=315),
    GateSpec(position=(5.0, 7.5, 4.5), yaw_deg=225),
    GateSpec(position=(5.0, 5.0, 4.5), yaw_deg=135),
]


@configclass
class RacingQuadcopterEnvCfg_FIXED(RacingQuadcopterEnvCfg):
    """Training config with a fixed, deterministic racing circuit.

    Override ``FIXED_GATES`` at the top of this file (or subclass and replace
    ``__post_init__``) to define your own track layout.
    """

    def __post_init__(self) -> None:
        terrain_generator = create_fixed_track_terrain(FIXED_GATES, seed=self.seed)
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


@configclass
class RacingQuadcopterEnvCfg_FIXED_PLAY(RacingQuadcopterEnvCfg_FIXED):
    """Evaluation config for the fixed circuit: eval_mode on, fixed spawn at gate 0, no randomisation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.eval_mode = True
        self.fixed_spawn = True
        self.observation_noise.enable = False
        self.action_noise.enable = False
        self.events.randomize_robot_mass = None
        self.events.randomize_robot_actuator_gains = None
        self.stats_warmup_s = 30.0
        print("[INFO] Fixed-track evaluation mode is ON.")

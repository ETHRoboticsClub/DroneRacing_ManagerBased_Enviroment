# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter manager env configuration (standalone)."""

from __future__ import annotations

from dataclasses import dataclass

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
from isaaclab.utils.noise import GaussianNoiseCfg
from Racing_Quadcopter.terrains.racing_terrain import create_fixed_track_terrain, create_racing_gates_terrain
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG
from Racing_Quadcopter.assets.robots_cfg import ANEMOS_DRONE_CFG_USD, ANEMOS_DRONE_CFG_URDF

from .mdp import observations, events, rewards, terminations, scale_dataclass_attributes
from .mdp.actions import QuadcopterThrustActionCfg
from .mdp.curriculums import apply_staged_curriculum
from .mdp.noise import drone_noise_model_factory

@dataclass
class DroneProfile:
    """Bundle a robot config with the physics parameters tuned for it."""

    robot_cfg: ArticulationCfg
    body_name: str
    thrust_to_weight: float
    moment_scale: float


def _make_profile(
    cfg: ArticulationCfg, body_name: str, thrust_to_weight: float, moment_scale: float
) -> DroneProfile:
    return DroneProfile(
        robot_cfg=cfg.replace(prim_path="{ENV_REGEX_NS}/Robot").replace(
            spawn=cfg.spawn.replace(activate_contact_sensors=True)
        ),
        body_name=body_name,
        thrust_to_weight=thrust_to_weight,
        moment_scale=moment_scale,
    )


# Anemos: ~0.85 kg total (base_link 0.835 + 4 props), inertia ixx≈2.16e-3, iyy≈2.92e-3.
# T/W ~4 is conservative for a 5" racing quad; moment_scale=0.1 gives ~35-45 rad/s² peak.
DRONE_PROFILES: dict[str, DroneProfile] = {
    "crazyflie":   _make_profile(CRAZYFLIE_CFG,         body_name="body",      thrust_to_weight=1.9, moment_scale=0.01),
    "anemos_usd":  _make_profile(ANEMOS_DRONE_CFG_USD,  body_name="base_link", thrust_to_weight=4.0, moment_scale=0.1),
    "anemos_urdf": _make_profile(ANEMOS_DRONE_CFG_URDF, body_name="base_link", thrust_to_weight=4.0, moment_scale=0.1),
}

# ---------------------------------------------------------------------------
# Top-of-file experiment knobs — defaults below; each can be overridden by an
# environment variable so you can switch without editing the file:
#
#   DRONE_MODEL=anemos_urdf TRACK_YAML=configs/tracks/simple.yaml \
#       python scripts/skrl/train.py --task Isaac-Racing-Quadcopter-Manager-Fixed-v0 ...
#
# DRONE_MODEL: one of the keys in DRONE_PROFILES.
#              "anemos_usd" requires a pre-converted USD; generate it via
#              scripts/tools/generate_anemos_usd.py.
# DEFAULT_TRACK_YAML: path to the gate track YAML (RPY in degrees, see
#              configs/tracks/default.yaml for the format). Read in
#              RacingQuadcopterEnvCfg_FIXED.__post_init__.
# ---------------------------------------------------------------------------
import os as _os
from Racing_Quadcopter.terrains import load_gates_from_yaml as _load_gates_from_yaml

DRONE_MODEL_ENV_VAR = "DRONE_MODEL"
TRACK_YAML_ENV_VAR = "TRACK_YAML"

DRONE_MODEL = _os.environ.get(DRONE_MODEL_ENV_VAR, "crazyflie")

_REPO_ROOT = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)
DEFAULT_TRACK_YAML = _os.path.join(_REPO_ROOT, "configs", "tracks", "default.yaml")

assert DRONE_MODEL in DRONE_PROFILES, (
    f"Invalid DRONE_MODEL={DRONE_MODEL!r}. Set ${DRONE_MODEL_ENV_VAR} to one of: {list(DRONE_PROFILES)}"
)
PROFILE = DRONE_PROFILES[DRONE_MODEL]


# SKRL additionally applies rewards_shaper_scale=0.01 on top.
@scale_dataclass_attributes(100.0)
@dataclass
class RewardWeights:
    GATE_PASSAGE: float = 250.0
    GATE_PROGRESS: float = 20.0
    GATE_CENTERING: float = 0.5
    VELOCITY_TOWARD_GATE: float = 2.0
    VELOCITY_LIMIT: float = -0.05
    PERCEPTION: float = -0.01
    ORIENTATION: float = -0.005
    ANGULAR_VELOCITY: float = -0.001
    ACTION_SMOOTHNESS: float = -0.0001
    ACTION_MAGNITUDE: float = -0.001
    CRASH: float = -50.0
    WRONG_SIDE: float = -0.1



@configclass
class GateObservationConfig:
    """Observation parameters."""

    num_next_gates: int = 3
    include_velocity_limit: bool = True


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

    robot: ArticulationCfg = PROFILE.robot_cfg

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
    replicate_physics: bool = False
    clone_in_fabric: bool = False


@configclass
class ActionsCfg:
    """Thrust + moments on the quadcopter base."""

    thrust = QuadcopterThrustActionCfg(
        asset_name="robot",
        thrust_to_weight=PROFILE.thrust_to_weight,
        moment_scale=PROFILE.moment_scale,
        action_dim=4,
        min_delay=0,
        max_delay=0,
        body_name=PROFILE.body_name,
    )


@configclass
class ObservationsCfg:
    """Policy observations."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        enable_corruption: bool = True
        concatenate_terms: bool = True
        
        history_length: int = 3
        
        base_lin_vel = ObsTerm(
            func=observations.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(std_noise=0.05, std_bias=0.01, dropout=0.0),
            history_length=history_length,
        )
        
        #IMU 
        base_ang_vel = ObsTerm(
            func=observations.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(std_noise=0.02, std_bias=0.005, dropout=0.0),
            history_length=history_length,
        )
        projected_gravity = ObsTerm(
            func=observations.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(std_noise=0.02, std_bias=0.005, dropout=0.0),
            history_length=history_length,
        )

        #ROBOT
        #TODO: Add motors speeds

        #ACTION
        last_action = ObsTerm(
            func=observations.last_action,
            history_length=history_length,
        )


        #Perception
        obs_gates = ObsTerm(
            func=observations.compute_next_gates_rel_and_fwd_body,
            noise=drone_noise_model_factory(std_noise=0.02, std_bias=0.0, dropout=0.0),
            history_length=history_length,
        )
        
        velocity_limit = ObsTerm(
            func=observations.velocity_limit_obs,
            history_length=history_length,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Same terms as direct ``GateRewardManager``; each callable returns unweighted factors in ``mdp.rewards``.

    Weights are ``MANAGER_SCALE *`` the default coefficients from the direct task.
    """

    gate_passage = RewTerm(func=rewards.reward_gate_passage, weight=RewardWeights.GATE_PASSAGE)
    gate_progress = RewTerm(func=rewards.reward_gate_progress, weight=RewardWeights.GATE_PROGRESS)
    gate_centering = RewTerm(func=rewards.reward_gate_centering, weight=RewardWeights.GATE_CENTERING)
    velocity_toward_gate = RewTerm(
        func=rewards.reward_velocity_toward_gate, weight=RewardWeights.VELOCITY_TOWARD_GATE
    )
    velocity_limit = RewTerm(func=rewards.penalty_velocity_limit, weight=RewardWeights.VELOCITY_LIMIT)
    # perception = RewTerm(func=rewards.penalty_perception, weight=RewardWeights.PERCEPTION)
    orientation = RewTerm(func=rewards.penalty_orientation, weight=RewardWeights.ORIENTATION)
    angular_velocity = RewTerm(
        func=rewards.penalty_angular_velocity, weight=RewardWeights.ANGULAR_VELOCITY
    )
    action_smoothness = RewTerm(
        func=rewards.penalty_action_smoothness, weight=RewardWeights.ACTION_SMOOTHNESS
    )
    action_magnitude = RewTerm(
        func=rewards.penalty_action_magnitude, weight=RewardWeights.ACTION_MAGNITUDE
    )
    crash = RewTerm(func=rewards.penalty_crash, weight=RewardWeights.CRASH)
    wrong_side = RewTerm(func=rewards.penalty_wrong_side, weight=RewardWeights.WRONG_SIDE)


@configclass
class EventsCfg:
    """Reset pipeline: Isaac scene defaults, then racing start pose and buffers."""

    randomize_robot_mass = EventTerm(
        func=events.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_base_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body|base_link"),
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
        },
    )
    randomize_robot_actuator_gains = EventTerm(
        func=events.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    base_external_force_torque = EventTerm(
        func=events.apply_external_force_torque,
        mode="interval",
        interval_range_s=(3.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body|base_link"),
            "force_range": (-0.02, 0.02),
            "torque_range": (-0.02, 0.02),
        },
    )
    push_robot = EventTerm(
        func=events.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 10.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3)}},
    )
    reset_scene_to_default = EventTerm(func=events.reset_scene_to_default, mode="reset")
    reset_racing_spawn = EventTerm(func=events.reset_racing_spawn, mode="reset", params={"method": "uniform"})


@configclass
class CurriculumCfg:
    """Staged curriculum: stay-alive → full racing → recovery spawns."""

    staged = CurrTerm(func=apply_staged_curriculum)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    upside_down = DoneTerm(func=terminations.upside_down, time_out=False)
    below_height = DoneTerm(func=terminations.below_height_threshold, time_out=False)
    runaway = DoneTerm(func=terminations.runaway_velocity, time_out=False)
    collision = DoneTerm(func=terminations.collision_termination, time_out=False)
    lap_complete = DoneTerm(func=terminations.lap_complete, time_out=True)


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
        self.observations.policy.enable_corruption = False
        self.action_noise.enable = False
        self.events.randomize_robot_mass = None
        self.events.randomize_robot_actuator_gains = None
        self.stats_warmup_s = 30.0
        print("[INFO] Evaluation mode is ON.")


# ---------------------------------------------------------------------------
# Fixed-track variants
# ---------------------------------------------------------------------------
# Tracks are defined in YAML under ``configs/tracks/``. The path is set on
# ``RacingQuadcopterEnvCfg_FIXED.gate_track_yaml`` and can be overridden via the
# ``RACING_GATE_TRACK_YAML`` env variable (read in __post_init__ before terrain
# generation, so it works through subprocess launches like the sweep runner).
#
# Legacy hard-coded layouts kept for reference — see configs/tracks/default.yaml
# and configs/tracks/simple.yaml for the YAML equivalents.
# ---------------------------------------------------------------------------
# FIXED_GATES (full circuit, see configs/tracks/default.yaml):
#   GateSpec(position=(7.5,  2.0, 1.2), yaw_deg=0),
#   GateSpec(position=(12.5, 4.5, 2.5), yaw_deg=65),
#   GateSpec(position=(13.5, 9.0, 4.5), yaw_deg=105),
#   GateSpec(position=(10.5, 13.0, 4.5), yaw_deg=160),
#   GateSpec(position=(7.5, 13.5, 4.2), yaw_deg=180),  # SPLIT-S entry
#   GateSpec(position=(7.5, 12.0, 1.5), yaw_deg=0),    # SPLIT-S exit
#   GateSpec(position=(4.0, 10.5, 2.0), yaw_deg=270),
#   GateSpec(position=(1.8,  7.5, 1.5), yaw_deg=230),
#   GateSpec(position=(3.5,  3.5, 1.8), yaw_deg=350),
#   GateSpec(position=(5.5,  2.0, 2.5), yaw_deg=25),
#
# FIXED_GATES_simple (4-gate diamond, see configs/tracks/simple.yaml):
#   GateSpec(position=(7.5, 5.0, 4.5), yaw_deg=45),
#   GateSpec(position=(7.5, 7.5, 4.5), yaw_deg=315),
#   GateSpec(position=(5.0, 7.5, 4.5), yaw_deg=225),
#   GateSpec(position=(5.0, 5.0, 4.5), yaw_deg=135),


@configclass
class RacingQuadcopterEnvCfg_FIXED(RacingQuadcopterEnvCfg):
    """Training config with a fixed, deterministic racing circuit.

    The track is loaded from ``gate_track_yaml`` (a path to a track YAML, see
    ``configs/tracks/default.yaml`` for the format). The ``RACING_GATE_TRACK_YAML``
    environment variable takes precedence — useful for subprocess overrides
    (sweep runner) since cfg-attribute overrides via ``--override`` are applied
    *after* ``__post_init__`` and therefore too late to influence track generation.
    """

    gate_track_yaml: str = DEFAULT_TRACK_YAML

    def __post_init__(self) -> None:
        track_path = _os.environ.get(TRACK_YAML_ENV_VAR, self.gate_track_yaml)
        gates = _load_gates_from_yaml(track_path)
        terrain_generator = create_fixed_track_terrain(gates, seed=self.seed)
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
        self.observations.policy.enable_corruption = False
        self.action_noise.enable = False
        self.events.randomize_robot_mass = None
        self.events.randomize_robot_actuator_gains = None
        self.stats_warmup_s = 30.0
        print("[INFO] Fixed-track evaluation mode is ON.")

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hovering / waypoint-following env configuration.

Sibling package to ``racing_quadcopter``. The only cross-package import is the
:class:`QuadcopterRatesActionCfg` (and the anemos drone profile constants)
from ``racing_quadcopter``, which the user explicitly asked to mirror as-is.
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass

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
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Reuse the rates-PID action term and the anemos drone profiles from racing.
from Racing_Quadcopter.tasks.manager_based.racing_quadcopter.mdp.actions import BetaflightSitlSurrogateActionCfg
from Racing_Quadcopter.tasks.manager_based.racing_quadcopter.racing_quadcopter_env_cfg import DRONE_PROFILES

from .mdp import events, observations, rewards, terminations, scale_dataclass_attributes
from .terrain import create_waypoint_terrain


# ---------------------------------------------------------------------------
# Drone selection
# ---------------------------------------------------------------------------
DRONE_MODEL_ENV_VAR = "HOVERING_DRONE_MODEL"
WAYPOINTS_ENV_VAR = "WAYPOINTS"

_ALLOWED_DRONE_MODELS = ("anemos_sitl", "anemos_usd", "anemos_urdf")
DRONE_MODEL = _os.environ.get(DRONE_MODEL_ENV_VAR, "anemos_sitl")
assert DRONE_MODEL in _ALLOWED_DRONE_MODELS, (
    f"Invalid {DRONE_MODEL_ENV_VAR}={DRONE_MODEL!r}. Allowed: {_ALLOWED_DRONE_MODELS}"
)
PROFILE = DRONE_PROFILES[DRONE_MODEL]
RATES_SCALE = 9.0
MAXIMUM_THRUST = 0.60
R_CAPTURE = 0.30    
SITL_ACTUATOR_CALIBRATION_PATH = _os.path.join(
    _os.path.dirname(__file__),
    "config",
    "sitl_actuator_calibration.json",
)
# ---------------------------------------------------------------------------
# Weight dataclass (scaled by 100 so SKRL's rewards_shaper_scale=0.01 restores
# the human-readable magnitudes — matches the racing convention)
# ---------------------------------------------------------------------------
# Mirror of the old direct-env hovering weights, pre-multiplied by the old
# env's ``reward_scale = 0.2``. ``@scale_dataclass_attributes(100.0)`` then
# scales by 100 so SKRL's ``rewards_shaper_scale = 0.01`` recovers the
# original effective magnitudes.
#
# Source values (from the direct env's ``RewardWeights`` dataclass):
#   lin_vel = -0.2,  ang_vel = -0.05,  distance_to_goal = -1.0,
#   action_magnitude = -0.02,  oversaturation = -0.03,
#   action_smoothness = -0.02,  crashed = -100.0,  reward_scale = 0.2
@scale_dataclass_attributes(50.0)
@dataclass
class WaypointRewardWeights:
    HOVERING_TO_WAYPOINT: float = 20.0 #To collect the waypoint
    MOVING_TO_TARGET: float = 20.0     #Better than distance reward, +1 if the distance is getting smaller             
    ANG_VEL: float = -0.3              #For smoother movements
    VEL_NEAR_WAYPOINT: float = -0.2    #Settle when close to the target (maybe )
    CRASH: float = -1000.0 
    GOAL_COLLECTED: float = 100.0      #Adjust this and decrease ANG_VEL for faster movements
    HIGH_THRUST: float = -100.0        #To ensure safety 
    RATE_TRACKING: float = -0.5        #It encourages small error commands of the controller
    ORIENTATION: float = -0.5          #Unwanted tilts
    ACTION_SMOOTHNESS: float = -2.0  
    ACTION_MAGNITUDE: float = -0.02
    ACTION_OUT_OF_RANGE: float = -50.0


@configclass
class ActionNoiseCfg:
    """Action-space domain randomisation (mirrors the racing cfg).

    Read by ``QuadcopterRatesAction.process_actions`` (see racing's
    ``mdp/actions.py``). Default off; flip ``enable=True`` to activate.
    """

    enable: bool = False
    per_episode_bias: bool = True
    gaussian_std: float = 0.0
    bias_range: tuple[float, float] = (-0.0, 0.0)
    lpf_alpha: float = 1.0
    clip_actions: bool = True


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class HoveringQuadcopterSceneCfg(InteractiveSceneCfg):
    """Scene: 8x8 flat-plane terrain + anemos drone + dome light. No contacts."""

    terrain: TerrainImporterCfg | None = None
    robot: ArticulationCfg = PROFILE.robot_cfg
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    replicate_physics: bool = False
    clone_in_fabric: bool = False


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    rates: BetaflightSitlSurrogateActionCfg = BetaflightSitlSurrogateActionCfg(
        body_name="base_link",
        thrust_to_weight=4.0,
        rate_command_scale=RATES_SCALE,
        thrust_command_limit=MAXIMUM_THRUST,
        sitl_calibration_path=SITL_ACTUATOR_CALIBRATION_PATH,
        pid_scales=1.0,
        # roll/pitch: 18-19% of τ_max at full rate error — unsaturated, ~3 rad/s bandwidth
        # yaw: 3.0 → 70% of 0.020 N·m at full rate error — max usable given k_d=1e-8
        #      D=0: yaw authority is too low for derivative term to add meaningful damping
        pid_p=(0.281, 0.281, 0.281), #computed from 
        pid_i=(0.0, 0.0, 0.0),
        pid_d=(0.04, 0.04, 0.04),
        pid_f=(0.0, 0.0, 0.0),
        pterm_scale = 1.0,
        dterm_scale = 0.0,
        iterm_scale = 0.0,
        feedforward_scale=0.0,
        torque_limit=2.0,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """Asymmetric actor-critic obs groups.

    - ``policy`` is what the actor sees — strictly mocap+IMU-derivable so it
      matches the deployment-time observability constraint.
    - ``critic`` adds the privileged ``base_lin_vel`` (true body-frame linear
      velocity from sim). The IsaacLab→SKRL wrapper surfaces this group
      through the ``STATES`` token; the value network in
      ``agents/skrl_ppo_cfg.yaml`` reads it via ``input: STATES``.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        enable_corruption: bool = False
        concatenate_terms: bool = True
        history_length: int = 1

        projected_gravity = ObsTerm(
            func=observations.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=history_length,
        )
        base_ang_vel = ObsTerm(
            func=observations.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=history_length,
        )

        #Command 
        next_waypoints = ObsTerm(
            func=observations.next_waypoints_rel_body,
            params={"num_waypoints": 1},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=history_length,
        )
        base_lin_vel = ObsTerm(
            func=observations.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=history_length,
        )


    policy: PolicyCfg = PolicyCfg()



# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
@scale_dataclass_attributes(50.0)
@configclass
class RewardsCfg:

    # TASK REWARDS
    hovering_to_waypoint = RewTerm(
        func=rewards.reward_hovering_to_waypoint, weight=WaypointRewardWeights.HOVERING_TO_WAYPOINT,
        params={"coefficient": 1.0}
    ) # Distance reward

    moving_to_target = RewTerm(
        func=rewards.moving_to_target, weight=WaypointRewardWeights.MOVING_TO_TARGET,
        params={"factor": 0.00}
    ) #+1 if distance is getting smaller
    goal_collected = RewTerm(
        func=rewards.reward_waypoint_collected, weight=WaypointRewardWeights.GOAL_COLLECTED
    )

    # REGULARIZATION & CONTROL REWARDS
    ang_vel = RewTerm(func=rewards.penalty_ang_vel, weight=WaypointRewardWeights.ANG_VEL)
    #PENALTIZES HIGH ANGULAR VELOCITY

    vel_near_waypoint = RewTerm(
        func=rewards.penalty_vel_near_waypoint,
        weight=WaypointRewardWeights.VEL_NEAR_WAYPOINT,
        params={"r_threshold": R_CAPTURE, "factor": 5.0},
    ) #LOW SPEED AT HOVERING
    

    high_thrust = RewTerm(
        func=rewards.penalty_high_thrust,
        weight=WaypointRewardWeights.HIGH_THRUST,
        params={"threshold": MAXIMUM_THRUST},
    ) #PENALTIZES [0,MAXIMUM_THRUST]

    penalty_rate_tracking = RewTerm(
        func=rewards.penalty_rate_tracking, weight=WaypointRewardWeights.RATE_TRACKING,
        params={"threshold": 1.0, "rate_command_scale": RATES_SCALE}
    ) #GOOD TO GIVE COMMANDS CLOSE TO OBSERVATIONS

    action_smoothness = RewTerm(
        func=rewards.penalty_action_smoothness, weight=WaypointRewardWeights.ACTION_SMOOTHNESS,
    ) #FIXES BANG-BANG
    action_magnitude = RewTerm(
        func=rewards.penalty_action_magnitude, weight=WaypointRewardWeights.ACTION_MAGNITUDE,
    )
    action_out_of_range = RewTerm(
        func=rewards.penalty_action_out_of_range, weight=WaypointRewardWeights.ACTION_OUT_OF_RANGE,
        params={"threshold": 0.95},
    )

    # SAFETY REWARDS

    penalty_orientation = RewTerm(
        func=rewards.penalty_orientation, weight=WaypointRewardWeights.ORIENTATION,
        params={"threshold": 0.70}
    )

    crash = RewTerm(func=rewards.penalty_crash, weight=WaypointRewardWeights.CRASH)



# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@configclass
class EventsCfg:
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
    randomize_robot_inertia = EventTerm(
        func=events.randomize_rigid_body_inertia_diagonal,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "inertia_distribution_params": (0.9, 1.1),
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
    reset_scene_to_default = EventTerm(func=events.reset_scene_to_default, mode="reset")
    reset_waypoint_spawn = EventTerm(func=events.reset_waypoint_spawn, mode="reset")


@configclass
class EvalCardinalEventsCfg(EventsCfg):
    """Events for the cardinal-spawn eval: same DR fields as the base events
    (so ``_apply_dr_flags`` can null them) but with the spawn term swapped
    for ``reset_cardinal_spawn``.
    """

    reset_waypoint_spawn = EventTerm(func=events.reset_cardinal_spawn, mode="reset")


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    crashed = DoneTerm(func=terminations.crashed, time_out=False)
    all_collected = DoneTerm(func=terminations.all_collected, time_out=True)


# ---------------------------------------------------------------------------
# Env cfg
# ---------------------------------------------------------------------------
@configclass
class HoveringQuadcopterEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the manager-based hovering / waypoint env."""

    seed: int = 42

    scene: HoveringQuadcopterSceneCfg = HoveringQuadcopterSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=False, clone_in_fabric=False
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    action_noise: ActionNoiseCfg = ActionNoiseCfg()

    decimation: int = 2
    episode_length_s: float = 15.0

    # Terrain layout (8x8 cells, each 15x15 m).
    terrain_num_rows: int = 8
    terrain_num_cols: int = 8
    cell_size: tuple[float, float] = (15.0, 15.0)

    # Waypoint generation defaults (used when WAYPOINTS env var is unset).
    num_waypoints_range: tuple[int, int] = (4, 8)
    waypoint_xy_range: float = 3.0
    waypoint_z_range: tuple[float, float] = (0.5, 2.5)
    min_waypoint_spacing: float = 0.8
    num_zero_tc_cells: int = 16
    collection_time_range_s: tuple[float, float] = (0.2, 2.0)
    #: Fallback ``T_c`` used when a YAML override omits the field.
    collection_time_s: float = 1.0

    # Proximity hysteresis.
    r_capture: float = R_CAPTURE
    r_hold: float = 0.60

    # Reset distribution around waypoint 0. The defaults preserve the previous
    # near-target spawn behaviour; sweeps can widen z to train takeoff-to-hover.
    spawn_xy_jitter: float = 0.3
    spawn_z_jitter: tuple[float, float] = (-0.1, 0.1)

    # Flight envelope.
    min_height: float = 0.10
    max_height: float = 5.0
    upside_down_threshold: float = 0.7

    # SITL parity defaults are deterministic; re-enable these only after parity is measured.
    randomize_mass: bool = False
    randomize_com: bool = False
    randomize_inertia: bool = False
    randomize_actuator_gains: bool = False
    apply_external_disturbances: bool = False

    # Vis toggle.
    draw_waypoints: bool = True

    # YAML override path (env var takes precedence). None means random per-cell.
    waypoint_track_yaml: str | None = None

    # Eval flag (used by ``HoveringQuadcopterEnvCfg_PLAY``).
    eval_mode: bool = False

    # Cardinal-spawn eval (used by ``HoveringEvalQuadcopterEnvCfg``). When True
    # the env replaces random track generation with a single waypoint per cell
    # at cell-local [0, 0, 1] and assigns each env one of E/W/N/S as the spawn
    # offset — mirrors ETHRC-Drone-Racing's ``test_controller_offline.py``.
    eval_cardinal_spawn: bool = False
    eval_initial_vz: float = 0.5  # m/s, world frame, applied at every reset

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    viewer: ViewerCfg = ViewerCfg(eye=(20.0, 20.0, 12.0), lookat=(7.5, 7.5, 1.0), origin_type="world")

    def _apply_dr_flags(self) -> None:
        if not self.randomize_mass:
            self.events.randomize_robot_mass = None
        if not self.randomize_com:
            self.events.randomize_base_com = None
        if not self.randomize_inertia:
            self.events.randomize_robot_inertia = None
        if not self.randomize_actuator_gains:
            self.events.randomize_robot_actuator_gains = None
        if not self.apply_external_disturbances:
            self.events.base_external_force_torque = None

    def __post_init__(self) -> None:
        terrain_generator = create_waypoint_terrain(
            seed=self.seed,
            num_rows=self.terrain_num_rows,
            num_cols=self.terrain_num_cols,
            cell_size=self.cell_size,
        )
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
            debug_vis=False,
        )
        self.sim.render_interval = self.decimation
        self._apply_dr_flags()


@configclass
class HoveringQuadcopterEnvCfg_PLAY(HoveringQuadcopterEnvCfg):
    """Play / evaluation config: longer episode, fewer envs, eval flag on."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.episode_length_s = 80.0
        self.eval_mode = True
        print("[INFO] Hovering quadcopter — evaluation mode is ON.")


@configclass
class HoveringEvalQuadcopterEnvCfg(HoveringQuadcopterEnvCfg):
    """Cardinal-spawn evaluation config (mirror of ``test_controller_offline.py``).

    Layout: exactly 4 terrain cells (1 row × 4 cols). Each cell has one
    waypoint at cell-local ``[0, 0, 1]``; envs are bound to cells by
    ``env_id % 4`` and spawn 1 m from the waypoint in the direction
    determined by ``HoveringQuadcopterRLEnv._CARDINAL_OFFSETS``:

        cell 0 — east  (drone at +X of goal)
        cell 1 — west  (drone at −X of goal)
        cell 2 — north (drone at +Y of goal)
        cell 3 — south (drone at −Y of goal)

    Identity attitude, zero body rates, vz = ``eval_initial_vz`` (m/s, world
    frame). All domain randomisation / external disturbances disabled, action
    noise off, so a deterministic policy produces a reproducible trajectory.

    Pair with ``Isaac-Hovering-Eval-Quadcopter-Manager-v0`` via the rsl_rl
    or skrl play scripts and load a checkpoint to inspect the converged
    hover behaviour beyond just the first action.
    """

    # 4 cells in a row.
    terrain_num_rows: int = 1
    terrain_num_cols: int = 4
    cell_size: tuple[float, float] = (6.0, 6.0)

    # One env per cell by default — bump to a multiple of 4 for parallel rollouts.
    eval_cardinal_spawn: bool = True
    eval_mode: bool = True

    # Everything off — eval must be deterministic.
    randomize_mass: bool = False
    randomize_com: bool = False
    randomize_inertia: bool = False
    apply_external_disturbances: bool = False

    events: EvalCardinalEventsCfg = EvalCardinalEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.episode_length_s = 10.0
        self.action_noise.enable = False
        print("[INFO] Hovering quadcopter — cardinal-spawn evaluation mode is ON.")

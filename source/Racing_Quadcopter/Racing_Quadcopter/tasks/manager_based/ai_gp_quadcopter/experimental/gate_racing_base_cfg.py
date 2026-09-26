# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config schema for this snapshot's :class:`RacingQuadcopterRLEnv` copy (``competition_env.py``).

Holds what the env class reads: the scene layout, the termination terms and plain runtime
fields. Everything tunable is owned by ``competition_env_cfg``.

This module reads no environment variables and selects no drone, so importing it has no side
effects. ``robot`` and the ``motor_*`` mixer fields are ``MISSING`` and must be set by the task.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from Racing_Quadcopter.tasks.manager_based.ai_gp_quadcopter.experimental.mdp.shared_features import ActionNoiseCfg, SpeedLimitsCfg

from Racing_Quadcopter.tasks.manager_based.ai_gp_quadcopter.experimental.mdp import terminations

#: Env var holding a gate-track YAML path; read in each fixed-track task's ``__post_init__``.
TRACK_YAML_ENV_VAR = "TRACK_YAML"


@configclass
class GateObservationConfig:
    """Observation parameters."""

    num_next_gates: int = 3
    mono_race_num_next_gates: int = 1
    mono_race_future_gates_pose_ahead: int = 2
    # Fraction of the gate aperture to shrink off each side before reporting it as
    # the 7th observation value (effective_half = half * (1 - 2*frac)). 0.0 reports
    # the true aperture; 0.2 reports a conservative 60%-width inscribed area.
    gate_size_obs_buffer_frac: float = 0.0


@configclass
class GateRacingSceneCfg(InteractiveSceneCfg):
    """Scene: racing-gates terrain, drone, contact sensor, dome light."""

    terrain: TerrainImporterCfg | None = None

    #: Set by the task (its drone profile's ``robot_cfg``).
    robot: ArticulationCfg = MISSING

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
        update_period=0.01,  # sensors refresh at most once per physics step (racing: every 0.02 s)
        track_air_time=False,
    )

    # FPV camera (opt-in). Populated in RacingQuadcopterEnvCfg.__post_init__ when
    # `enable_fpv_camera=True`. Default off — rendering a per-env camera is expensive
    # and not needed for blind-state training.
    fpv_camera: CameraCfg | None = None

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    replicate_physics: bool = False
    clone_in_fabric: bool = False


@configclass
class GateRacingTerminationsCfg:
    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    upside_down = DoneTerm(func=terminations.upside_down, time_out=False)
    below_height = DoneTerm(func=terminations.below_height_threshold, time_out=False)
    runaway = DoneTerm(func=terminations.runaway_velocity, time_out=False)
    collision = DoneTerm(func=terminations.collision_termination, time_out=False)
    lap_complete = DoneTerm(func=terminations.lap_complete, time_out=True)

    # On most tracks this is irrelevant but on some it is necessary (and has to be activated at a later
    # curriculum stage to avoid hindering early learning).
    # wrong_direction = DoneTerm(func=terminations.wrong_direction_crossing, time_out=False)


@configclass
class GateRacingEnvCfg(ManagerBasedRLEnvCfg):
    """Fields read by :class:`RacingQuadcopterRLEnv` and the shared racing MDP code.

    Tasks subclass this and supply ``observations``, ``actions``, ``rewards``, ``events``,
    ``curriculum``, ``scene.robot`` and the ``motor_*`` mixer parameters.
    """

    seed: int = 42

    scene: GateRacingSceneCfg = GateRacingSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=False, clone_in_fabric=False)

    terminations: GateRacingTerminationsCfg = GateRacingTerminationsCfg()
    observation: GateObservationConfig = GateObservationConfig()
    action_noise: ActionNoiseCfg = ActionNoiseCfg()
    #: Episode-conditioned speed limits; disabled here, tasks opt in.
    speed_limits: SpeedLimitsCfg = SpeedLimitsCfg()

    # Gates the shared route-progress curricula that declare ``enabled_attr``.
    adaptive_curriculum_enabled: bool = False
    # Expensive action percentiles are opt-in and intended for evaluation/debug.
    action_diagnostics_enabled: bool = False
    action_diagnostics_interval_s: float = 1.0

    decimation: int = 2
    episode_length_s: float = 20.0

    upside_down_threshold: float = 0.7
    min_height: float = 0.2
    max_velocity: float = 50.0
    collision_force_threshold: float = 0.001

    eval_mode: bool = False
    # Master switch. Collision termination is additionally gated PER ENV by the competence
    # curriculum (curriculum_min_* below, measured on the previous episode); eval_mode and
    # force_collision_termination bypass that gate.
    crash_termination: bool = True
    force_collision_termination: bool = False  # skip the per-env competence curriculum and terminate on any collision
    fixed_spawn: bool = False        # if True, all drones spawn at gate 0 instead of a random gate
    recovery_spawn_prob: float = 0.2 # fraction [0,1] of envs that spawn with random orientation + velocity

    speed_log_interval_s: float = 10.0
    stats_warmup_s: float = 0.0    # skip stats accumulation for this many simulation seconds at episode start
    draw_trajectory: bool = False        # green closed-loop polyline through all waypoints (USD, no collision)
    draw_gate_indicators: bool = False   # blue sticks showing each gate's forward (+Y) direction (USD, no collision)
    draw_gate_labels: bool = True       # blue 7-segment gate index numbers floating above each gate (USD, no collision)
    draw_camera_arrow: bool = False     # yellow arrow above each drone showing its camera's true world-frame forward axis (per-step USD update — visualisation only)

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

    # Staged curriculum (racing): step counts are absolute env steps. With both at 10000, stage 1
    # lasts a single reset: upside-down termination turns off, the speed-limit latch fires and
    # recovery spawns start together. Use stage2 > stage1 to separate them.
    curriculum_stage1_steps: int = 10000
    curriculum_stage2_steps: int = 10000
    curriculum_min_gates_passed: int = 10
    curriculum_min_mean_velocity: float = 2.5
    curriculum_max_center_offset: float = 0.25

    # Motor mixing (:math:`B`, :math:`B^{-1}` built once in the env); set from the task's drone profile.
    motor_k_t: float = MISSING
    motor_k_d: float = MISSING
    motor_arm_length_m: float = MISSING
    motor_omega_max_rad_s: float = MISSING

    #: Threshold Δu_thresh (rad/s): motor-speed penalty counts excess only above this per rotor per step.
    motor_speed_delta_thresh_rad_s: float = 0.1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50,
        render_interval=2,
        # The default 2**21 aggregate-pair buffer is short by one pair per
        # environment at 8192 envs, which makes PhysX drop broad-phase
        # interactions. Keep a power-of-two margin for competition sweeps.
        physx=sim_utils.PhysxCfg(gpu_total_aggregate_pairs_capacity=2**22),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    viewer: ViewerCfg = ViewerCfg(eye=(15.0, 15.0, 7.0), lookat=(3.0, 3.0, 0.0), origin_type="world")

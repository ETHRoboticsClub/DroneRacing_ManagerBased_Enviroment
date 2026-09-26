# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter manager env configuration.

Everything config for the racing task in one module: the runtime fields the env class
(:class:`RacingQuadcopterRLEnv`) reads directly (scene layout, termination terms, plain runtime
fields such as episode length/thresholds/demo toggles), racing's drone profiles, managers and
reward presets (``RACING_REWARD_PRESET``), the route-adaptive randomization profile, the
fixed-track and FPV variants, and the skill-evaluation config (``SkillTerrainCfg``,
``RacingQuadcopterSkillEvalEnvCfg``).

The AI-GP experimental snapshot has its own frozen copy of the runtime-field schema
(``ai_gp_quadcopter/experimental/``).
"""

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
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.noise import GaussianNoiseCfg
from Racing_Quadcopter.terrains import create_multi_track_terrain
from Racing_Quadcopter.terrains.racing_terrain import create_fixed_track_terrain, create_racing_gates_terrain
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG
from Racing_Quadcopter.assets.robots_cfg import ANEMOS_DRONE_CFG_USD, ANEMOS_DRONE_CFG_URDF, ANEMOS_SITL_DRONE_CFG_URDF
from Racing_Quadcopter.tasks.manager_based.mdp.adaptive_curriculums import (
    AdaptiveRandomizationFromRouteProgress,
    AdaptiveRewardScaleFromRouteProgress,
    CurriculumTarget,
    get_randomization_profile,
    register_randomization_profile,
)
from Racing_Quadcopter.tasks.manager_based.mdp import shared_features
from Racing_Quadcopter.tasks.manager_based.mdp.shared_features import ActionNoiseCfg, SpeedLimitsCfg
from Racing_Quadcopter.tasks.manager_based.mdp.noise import drone_noise_model_factory
from Racing_Quadcopter.tasks.manager_based.drone_profile import DroneProfile, make_drone_profile

from Racing_Quadcopter.tasks.manager_based.mdp import events, observations, rewards, terminations, scale_dataclass_attributes
from Racing_Quadcopter.tasks.manager_based.mdp.actions import QuadcopterRatesActionCfg, QuadcopterMomentsActionCfg
from Racing_Quadcopter.tasks.manager_based.mdp.curriculums import apply_staged_curriculum

# Kept for existing imports (scripts/tools, deployment): the schema now lives in drone_profile.py.
_make_profile = make_drone_profile

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


# Anemos: ~0.85 kg total (base_link 0.835 + 4 props), inertia ixx≈2.16e-3, iyy≈2.92e-3.
#
# moment_scale derivation (per-axis, from URDF geometry and k_t/k_d):
#   τ_roll_max  = min(Σ|y_left| , Σ|y_right|) × k_t × Ω_max²
#               = min(0.1438, 0.1360) × 8.54858 = 1.16 N·m   (motors at y=±0.068–0.072 m)
#   τ_pitch_max = min(Σ|x_back| , Σ|x_front|) × k_t × Ω_max²
#               = min(0.1640, 0.1771) × 8.54858 = 1.40 N·m   (motors at x=±0.082–0.089 m)
#   τ_yaw_max   = 2 × k_d × Ω_max²
#               = 2 × 1e-8 × 1e6 = 0.020 N·m
#   Conservative: min of both ± directions so action ±1 is achievable without saturation.
DRONE_PROFILES: dict[str, DroneProfile] = {
    # Motor values taken from Crazyflie 2.0:
    # https://skoge.folk.ntnu.no/prost/proceedings/ICSTCC-2018-Sinaia/ICSTCC2018/papers/0157.pdf
    # Crazyflie moment_scale: roll/pitch max ≈ 0.011 N·m, yaw max ≈ 0.010 N·m → (0.01, 0.01, 0.01) is correct.
    "crazyflie": _make_profile(
        CRAZYFLIE_CFG,
        body_name="body",
        thrust_to_weight=1.9,
        moment_scale=(0.01, 0.01, 0.01),
        motor_k_t=2.9e-8,
        motor_k_d=7.3e-10,
        motor_arm_length_m=0.0397,
        motor_omega_max_rad_s=2600.0,
        rate_command_scale=30.0,
        pid_scales=0.00005,
        pid_p=(45.0, 47.0, 45.0),
        pid_i=(0.0, 0.0, 0.0),
        pid_d=(32.0, 34.0, 32.0),
        pid_f=(0.0, 0.0, 0.0),
    ),
    "anemos_usd": _make_profile(
        ANEMOS_DRONE_CFG_USD,
        body_name="base_link",
        thrust_to_weight=4.1056,  # 4×k_t×Ω_max² / (m_total×g) = 34.194 / 8.329 N (from URDF masses)
        moment_scale=(1.16, 1.40, 0.020),   # (roll, pitch, yaw) N·m — see derivation above
        motor_k_t=8.54858e-6,
        motor_k_d=1e-8,
        motor_arm_length_m=0.110,  # mean centre-to-rotor from URDF joints: 0.107–0.114 m
        motor_omega_max_rad_s=1000.0,
        rate_command_scale=30.0,
        pid_scales=0.005,
        # roll/pitch: 18-19% of τ_max at full rate error — unsaturated, ~3 rad/s bandwidth
        # yaw: 0.64 → ~15% of 0.020 N·m at full rate error (Kp≈1.0e-4 N·m/(rad/s),
        #      Izz=2.74e-3 → ~27 s time constant); D=0.45 is negligible (Kd≈1.2e-6)
        pid_p=(45.0, 47.0, 0.64),
        pid_i=(0.0, 0.0, 0.0),
        pid_d=(32.0, 34.0, 0.45),
        pid_f=(0.0, 0.0, 0.0),
    ),
    "anemos_urdf": _make_profile(
        ANEMOS_DRONE_CFG_URDF,
        body_name="base_link",
        thrust_to_weight=4.1056,  # 4×k_t×Ω_max² / (m_total×g) = 34.194 / 8.329 N (from URDF masses)
        moment_scale=(1.16, 1.40, 0.020),   # (roll, pitch, yaw) N·m — see derivation above
        motor_k_t=8.54858e-6,
        motor_k_d=1e-8,
        motor_arm_length_m=0.110,  # mean centre-to-rotor from URDF joints: 0.107–0.114 m
        motor_omega_max_rad_s=1000.0,
        rate_command_scale=30.0,
        pid_scales=0.005,
        pid_p=(45.0, 47.0, 0.64),
        pid_i=(0.0, 0.0, 0.0),
        pid_d=(32.0, 34.0, 0.45),
        pid_f=(0.0, 0.0, 0.0),
    ),
    # NOT a SITL-parity setup: racing builds the direct-mode QuadcopterRatesActionCfg with the
    # default Betaflight term scales, so pid_p=0.281 becomes Kp ≈ 0.009 (the SITL surrogate
    # assumes pterm_scale=1), there is no SITL allocation / throttle cap, and motor_k_d (0.00125)
    # is a torque/thrust ratio, not the N·m/(rad/s)² the mixer expects. Unvalidated for training.
    "anemos_sitl": _make_profile(
        ANEMOS_SITL_DRONE_CFG_URDF,
        body_name="base_link",
        thrust_to_weight=4.0,
        moment_scale=(1.16, 1.40, 0.020),
        motor_k_t=8.1896182,
        motor_k_d=0.00125,
        motor_arm_length_m=0.110,
        motor_omega_max_rad_s=1.0,
        rate_command_scale=9.0,
        pid_scales=1.0,
        pid_p=(0.281, 0.281, 0.281),
        pid_i=(0.0, 0.0, 0.0),
        pid_d=(0.04, 0.04, 0.04),
        pid_f=(0.0, 0.0, 0.0),
    ),
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
#              Motor mixer k_T, k_D, l and obs scale Ω_max live on DroneProfile;
#              env builds B^{-1} once from RacingQuadcopterEnvCfg.
# ACTION_TYPE: "rates"   — thrust + inner-loop rate PID → body torques (default).
#              "moments" — direct thrust + body moments, no inner PID loop.
# RACING_REWARD_PRESET: "may" (default) or "current" — see RACING_REWARD_PRESETS below.
#              Use the same value for train and play (it changes the observation size).
# DEFAULT_TRACK_YAML: path to the gate track YAML (RPY in degrees, see
#              configs/tracks/default.yaml for the format). Read in
#              RacingQuadcopterEnvCfg_FIXED.__post_init__.
# ---------------------------------------------------------------------------
import os as _os
from Racing_Quadcopter.terrains import load_gates_from_yaml as _load_gates_from_yaml

DRONE_MODEL_ENV_VAR = "DRONE_MODEL"
ACTION_TYPE_ENV_VAR = "ACTION_TYPE"
OBSERVATION_SPACE_ENV_VAR = "OBSERVATION_SPACE"
REWARD_PRESET_ENV_VAR = "RACING_REWARD_PRESET"

_ACTION_TYPES = ("rates", "moments")
_OBSERVATION_SPACES = ("mono_race", "ours")

DRONE_MODEL = _os.environ.get(DRONE_MODEL_ENV_VAR, "crazyflie")
ACTION_TYPE = _os.environ.get(ACTION_TYPE_ENV_VAR, "rates")
OBSERVATION_SPACE = _os.environ.get(OBSERVATION_SPACE_ENV_VAR, "ours")
REWARD_PRESET = _os.environ.get(REWARD_PRESET_ENV_VAR, "may")

_REPO_ROOT = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)
DEFAULT_TRACK_YAML = _os.path.join(_REPO_ROOT, "configs", "tracks", "default.yaml")

assert DRONE_MODEL in DRONE_PROFILES, (
    f"Invalid DRONE_MODEL={DRONE_MODEL!r}. Set ${DRONE_MODEL_ENV_VAR} to one of: {list(DRONE_PROFILES)}"
)
assert ACTION_TYPE in _ACTION_TYPES, (
    f"Invalid ACTION_TYPE={ACTION_TYPE!r}. Set ${ACTION_TYPE_ENV_VAR} to one of: {list(_ACTION_TYPES)}"
)
assert OBSERVATION_SPACE in _OBSERVATION_SPACES, (
    f"Invalid OBSERVATION_SPACE={OBSERVATION_SPACE!r}. Set ${OBSERVATION_SPACE_ENV_VAR} to one of: {list(_OBSERVATION_SPACES)}"
)


# ---------------------------------------------------------------------------
# Route-adaptive randomization profile (formerly racing_randomization.py).
#
# Base-racing curriculum policy; every value below is local to base racing. Registered with the
# shared registry in mdp/adaptive_curriculums.py (hovering registers its own profile the same
# way, see hovering_randomization.py).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RacingRandomizationProfile:
    action_gaussian_std: float = 0.02
    action_bias_range: tuple[float, float] = (-0.05, 0.05)
    action_lpf_start_range: tuple[float, float] = (0.9, 1.0)
    action_lpf_range: tuple[float, float] = (0.8, 1.0)
    # Match the AI-GP experimental observation corruption. Dynamics remain deliberately
    # milder for the base PID-rate action than AI-GP's second-order one.
    base_lin_vel_noise: tuple[float, float] = (0.2, 1.0)
    base_ang_vel_noise: tuple[float, float] = (0.05, 0.2)
    gravity_noise: tuple[float, float] = (0.02, 0.04)
    max_delay: int = 1
    pid_gain_randomization_frac: float = 0.05
    thrust_to_weight_randomization_frac: float = 0.03
    aero_randomization_frac: float = 0.05
    linear_drag_coeff: tuple[float, float, float] = (0.04, 0.04, 0.06)
    quadratic_drag_coeff: tuple[float, float, float] = (0.005, 0.005, 0.009)
    angular_drag_coeff: tuple[float, float, float] = (0.001, 0.001, 0.0006)


# Default values for the racing randomization profile. Named distinctly from the module-level
# ``RACING_RANDOMIZATION`` below (the profile fetched back from the shared registry) to keep the
# two uses of that name apart.
_RACING_RANDOMIZATION_DEFAULTS = RacingRandomizationProfile()


def racing_randomization_targets(*, observation_terms: tuple[str, ...], action_type: str) -> tuple[CurriculumTarget, ...]:
    """Explicit target list for fields which exist in the selected racing action."""
    p = _RACING_RANDOMIZATION_DEFAULTS
    def env(name, path, start, target): return CurriculumTarget(name, "env_cfg", path, "lerp", target, start)
    def action(name, path, start, target, *, dtype=None):
        return CurriculumTarget(name, "action_term_cfg", path, "lerp", target, start, term_name="rates", dtype=dtype, clamp_min=0 if name == "rates_max_delay" else None)
    obs_values = {"base_lin_vel": p.base_lin_vel_noise, "base_ang_vel": p.base_ang_vel_noise, "projected_gravity": p.gravity_noise}
    targets = [
        env("action_noise_enable", "action_noise.enable", False, True),
        env("action_noise_gaussian_std", "action_noise.gaussian_std", 0.0, p.action_gaussian_std),
        env("action_noise_bias_range", "action_noise.bias_range", (0.0, 0.0), p.action_bias_range),
        env("action_noise_lpf_alpha_range", "action_noise.lpf_alpha_range", p.action_lpf_start_range, p.action_lpf_range),
    ]
    for term in observation_terms:
        std, bias = obs_values[term]
        targets += [
            CurriculumTarget(f"obs_{term}_enable", "observation_term_cfg", "noise.enable", "lerp", True, False, term, "policy"),
            CurriculumTarget(f"obs_{term}_std", "observation_term_cfg", "noise.noise_cfg.std", "lerp", std, 0.0, term, "policy"),
            CurriculumTarget(f"obs_{term}_bias_std", "observation_term_cfg", "noise.bias_noise_cfg.std", "lerp", bias, 0.0, term, "policy"),
        ]
    if action_type == "moments":
        targets.append(CurriculumTarget("moments_max_delay", "action_term_cfg", "max_delay", "lerp", p.max_delay, 0, term_name="moments", dtype="int", clamp_min=0))
        return tuple(targets)
    if action_type != "rates": raise ValueError(f"Unsupported base-racing action type {action_type!r}.")
    zero3 = (0.0, 0.0, 0.0)
    targets += [
        action("rates_max_delay", "max_delay", 0, p.max_delay, dtype="int"),
        action("rates_pid_gain_rand_frac", "pid_gain_randomization_frac", 0.0, p.pid_gain_randomization_frac),
        action("rates_thrust_to_weight_rand_frac", "thrust_to_weight_randomization_frac", 0.0, p.thrust_to_weight_randomization_frac),
        action("rates_aero_randomization_enable", "aero_randomization_enable", False, True),
        action("rates_aero_randomization_frac", "aero_randomization_frac", 0.0, p.aero_randomization_frac),
        action("rates_linear_drag_coeff", "linear_drag_coeff", zero3, p.linear_drag_coeff),
        action("rates_quadratic_drag_coeff", "quadratic_drag_coeff", zero3, p.quadratic_drag_coeff),
        action("rates_angular_drag_coeff", "angular_drag_coeff", zero3, p.angular_drag_coeff),
    ]
    return tuple(targets)


def register_racing_randomization_profile(
    *, observation_terms: tuple[str, ...], action_type: str
) -> None:
    """Register base-racing's local profile and its selected action/obs bindings."""
    register_randomization_profile(
        "racing",
        _RACING_RANDOMIZATION_DEFAULTS,
        lambda: racing_randomization_targets(
            observation_terms=observation_terms, action_type=action_type
        ),
    )


# Register before observation configs are constructed, then retrieve the local
# profile through the shared registry.  This keeps configuration values and
# curriculum endpoints in one task-local source of truth.
_RACING_NOISE_TERMS = ("base_ang_vel",) if OBSERVATION_SPACE == "mono_race" else (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
)
register_racing_randomization_profile(
    observation_terms=_RACING_NOISE_TERMS,
    action_type=ACTION_TYPE,
)
RACING_RANDOMIZATION = get_randomization_profile("racing")
PROFILE = DRONE_PROFILES[DRONE_MODEL]


# ---------------------------------------------------------------------------
# Racing reward presets — selected with the RACING_REWARD_PRESET env var.
#
#   may     (default) — the reward balance of the May regression baseline (28f45cb): the
#                       regularizers are tiny next to gate passage (250·dt = 10 per gate), no
#                       adaptive curricula, no episode speed limits.
#   current           — the balance racing inherited from the AI-GP tuning (b1dd396,
#                       3bde690): heavy action/smoothness regularization with the ×100
#                       out-of-range action penalty, adaptive curricula and speed limits on.
#
# The preset changes the observation size (speed limits add 2 inputs), so train and play
# must use the same preset. The AI-GP task keeps its own reward space (ai_gp_quadcopter/).
# ---------------------------------------------------------------------------
@dataclass
class MayRewardWeights:
    GATE_PASSAGE: float = 250.0
    GATE_PROGRESS: float = 20.0
    GATE_CENTERING: float = 0.5
    VELOCITY_TOWARD_GATE: float = 2.0
    # Inert under this preset: speed limits are disabled, so the shared velocity-limit penalty
    # returns zero. (The May baseline penalized speed above a fixed 15 m/s cap.)
    VELOCITY_LIMIT: float = -0.05
    ANGULAR_VELOCITY_LIMIT: float = 0.0
    PERCEPTION: float = -0.01
    ORIENTATION: float = -0.005
    ANGULAR_VELOCITY: float = -0.001
    ACTION_SMOOTHNESS: float = -0.0001
    ACTION_MAGNITUDE: float = -0.001
    # Fires on contact OR upside-down (the May baseline penalized contact only).
    CRASH: float = -50.0
    WRONG_SIDE: float = -0.1


@dataclass
class CurrentRewardWeights:
    GATE_PASSAGE: float = 250.0
    GATE_PROGRESS: float = 20.0
    GATE_CENTERING: float = 4.0
    VELOCITY_TOWARD_GATE: float = 2.0
    VELOCITY_LIMIT: float = -1.5
    ANGULAR_VELOCITY_LIMIT: float = -0.2
    PERCEPTION: float = -0.01
    ORIENTATION: float = -0.005
    ANGULAR_VELOCITY: float = -0.03
    ACTION_SMOOTHNESS: float = -5.0
    ACTION_MAGNITUDE: float = -1.0
    CRASH: float = -50.0
    WRONG_SIDE: float = -0.1

    # Currently not in use
    # ACTION_RATE_THRESH: float = -0.005
    # LOW_ACTION: float = -0.001
    # WRONG_DIRECTION: float = -50.0


@dataclass(frozen=True)
class RacingRewardPreset:
    weights: type
    #: Multiplier on the action-magnitude penalty when any raw action leaves [-1, 1].
    action_magnitude_clip_excess_multiplier: float
    adaptive_curriculum_enabled: bool
    speed_limits_enabled: bool


RACING_REWARD_PRESETS: dict[str, RacingRewardPreset] = {
    "may": RacingRewardPreset(
        weights=MayRewardWeights,
        action_magnitude_clip_excess_multiplier=1.0,
        adaptive_curriculum_enabled=False,
        speed_limits_enabled=False,
    ),
    "current": RacingRewardPreset(
        weights=CurrentRewardWeights,
        action_magnitude_clip_excess_multiplier=100.0,
        adaptive_curriculum_enabled=True,
        speed_limits_enabled=True,
    ),
}
assert REWARD_PRESET in RACING_REWARD_PRESETS, (
    f"Invalid {REWARD_PRESET_ENV_VAR}={REWARD_PRESET!r}. Set ${REWARD_PRESET_ENV_VAR} to one of: {list(RACING_REWARD_PRESETS)}"
)
RACING_REWARD_PRESET = RACING_REWARD_PRESETS[REWARD_PRESET]
RewardWeights = RACING_REWARD_PRESET.weights


@configclass
class RacingQuadcopterSceneCfg(InteractiveSceneCfg):
    """Racing scene: gate-racing terrain, contact sensor, dome light, with the ``DRONE_MODEL`` robot."""

    terrain: TerrainImporterCfg | None = None

    robot: ArticulationCfg = PROFILE.robot_cfg

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
class TerminationsCfg:
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
class ActionsCfg:
    """Action space selected by ACTION_TYPE env var ('rates' or 'moments')."""

    rates: QuadcopterRatesActionCfg | None = None
    moments: QuadcopterMomentsActionCfg | None = None

    def __post_init__(self):
        if ACTION_TYPE == "rates":
            self.rates = QuadcopterRatesActionCfg(
                asset_name="robot",
                thrust_to_weight=PROFILE.thrust_to_weight,
                action_dim=4,
                body_name=PROFILE.body_name,
                pid_scales=PROFILE.pid_scales,
                pid_p=PROFILE.pid_p,
                pid_i=PROFILE.pid_i,
                pid_d=PROFILE.pid_d,
                pid_f=PROFILE.pid_f,
                rate_command_scale=PROFILE.rate_command_scale,
            )
        else:  # moments
            self.moments = QuadcopterMomentsActionCfg(
                asset_name="robot",
                thrust_to_weight=PROFILE.thrust_to_weight,
                action_dim=4,
                body_name=PROFILE.body_name,
                moment_scale=PROFILE.moment_scale,
            )

# Observation history windows. Module-level (not class attributes): the ObservationManager
# enumerates every attribute in a group's __dict__ as a term, and the name must not be
# ``history_length`` (the inherited ObsGroup field, which would override every term).
_OURS_OBS_WINDOW = 1
_MONORACE_OBS_WINDOW = 1


@configclass
class ObservationsCfg:
    """Policy observations."""

    @configclass
    class OursObservationsCfg(ObsGroup):
        # Individual noise models default off; this keeps their configs alive
        # for the optional shared curriculum.
        enable_corruption: bool = True
        concatenate_terms: bool = True


        base_lin_vel = ObsTerm(
            func=observations.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(
                std_noise=RACING_RANDOMIZATION.base_lin_vel_noise[0],
                std_bias=RACING_RANDOMIZATION.base_lin_vel_noise[1],
                dropout=0.0,
            ),
            history_length=_OURS_OBS_WINDOW,
        )

        #IMU
        base_ang_vel = ObsTerm(
            func=observations.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(
                std_noise=RACING_RANDOMIZATION.base_ang_vel_noise[0],
                std_bias=RACING_RANDOMIZATION.base_ang_vel_noise[1],
                dropout=0.0,
            ),
            history_length=_OURS_OBS_WINDOW,
        )
        projected_gravity = ObsTerm(
            func=observations.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(
                std_noise=RACING_RANDOMIZATION.gravity_noise[0],
                std_bias=RACING_RANDOMIZATION.gravity_noise[1],
                dropout=0.0,
            ),
            history_length=_OURS_OBS_WINDOW,
        )

        #ACTION
        last_action = ObsTerm(
            func=observations.last_action,
            history_length=_OURS_OBS_WINDOW,
        )


        #Perception
        obs_gates = ObsTerm(
            func=observations.compute_next_gates_rel_and_fwd_body,
            noise=drone_noise_model_factory(std_noise=0.02, std_bias=0.0, dropout=0.0),
            history_length=_OURS_OBS_WINDOW,
        )

        speed_limits = ObsTerm(
            func=observations.speed_limits_obs,
        )

    @configclass
    class MonoRaceObservationsCfg(ObsGroup):
        enable_corruption: bool = True
        concatenate_terms: bool = True


        mono_race_gate_position_velocity_gate_frame = ObsTerm(
            func=observations.mono_race_gate_position_velocity_gate_frame,
            history_length=_MONORACE_OBS_WINDOW,
        )
        mono_race_gate_orientations = ObsTerm(
            func=observations.mono_race_gate_orientations,
            history_length=_MONORACE_OBS_WINDOW,
        )

        mono_race_future_gates_pose_in_current_gate_frame = ObsTerm(
            func=observations.mono_race_future_gates_pose_in_current_gate_frame,
            history_length=_MONORACE_OBS_WINDOW,
        )
        base_ang_vel = ObsTerm(
            func=observations.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(
                std_noise=RACING_RANDOMIZATION.base_ang_vel_noise[0],
                std_bias=RACING_RANDOMIZATION.base_ang_vel_noise[1],
                dropout=0.0,
            ),
            scale= 0.2,
            history_length=_MONORACE_OBS_WINDOW,
        )

        motor_speeds = ObsTerm(
            func=observations.motor_speeds_obs,
            history_length=_MONORACE_OBS_WINDOW,
        )

        speed_limits = ObsTerm(
            func=observations.speed_limits_obs,
        )


    if OBSERVATION_SPACE == "mono_race":
        policy = MonoRaceObservationsCfg()
    else:
        policy = OursObservationsCfg()


@configclass
class RewardsCfg:
    """Racing reward terms; each callable returns unweighted factors in ``mdp.rewards``.

    Weights come from the active ``RACING_REWARD_PRESET``. The RewardManager multiplies every
    term by ``step_dt``.
    """

    gate_passage = RewTerm(func=rewards.reward_gate_passage, weight=RewardWeights.GATE_PASSAGE)
    gate_progress = RewTerm(func=rewards.reward_gate_progress, weight=RewardWeights.GATE_PROGRESS)
    gate_centering = RewTerm(
        func=rewards.reward_gate_centering,
        weight=RewardWeights.GATE_CENTERING,
        params={"reference_speed": 15.0},
    )
    velocity_toward_gate = RewTerm(
        func=rewards.reward_velocity_toward_gate, weight=RewardWeights.VELOCITY_TOWARD_GATE
    )
    velocity_limit = RewTerm(func=rewards.penalty_velocity_limit, weight=RewardWeights.VELOCITY_LIMIT)
    angular_velocity_limit = RewTerm(
        func=rewards.penalty_angular_velocity_limit, weight=RewardWeights.ANGULAR_VELOCITY_LIMIT
    )

    orientation = RewTerm(func=rewards.penalty_orientation, weight=RewardWeights.ORIENTATION)
    angular_velocity = RewTerm(
        func=rewards.penalty_angular_velocity, weight=RewardWeights.ANGULAR_VELOCITY
    )
    action_smoothness = RewTerm(
        func=rewards.penalty_action_smoothness, weight=RewardWeights.ACTION_SMOOTHNESS
    )

    action_magnitude = RewTerm(
        func=shared_features.penalty_action_magnitude,
        weight=RewardWeights.ACTION_MAGNITUDE,
        params={
            "clip_threshold": 1.0,
            "clip_excess_multiplier": RACING_REWARD_PRESET.action_magnitude_clip_excess_multiplier,
        },
    )
    crash = RewTerm(func=rewards.penalty_crash, weight=RewardWeights.CRASH)
    wrong_side = RewTerm(func=rewards.penalty_wrong_side, weight=RewardWeights.WRONG_SIDE)

    # On most tracks this is irrelevant but on some it is necessary (and has to be activated at a later
    # curriculum stage to avoid hindering early learning).
    # wrong_direction = RewTerm(func=rewards.penalty_wrong_direction, weight=RewardWeights.WRONG_DIRECTION)

    perception = RewTerm(func=rewards.penalty_perception, weight=RewardWeights.PERCEPTION)


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
    # Must stay after randomize_robot_mass: that term uses recompute_inertia=True,
    # which would overwrite any inertia scaling applied before it.
    randomize_robot_inertia = EventTerm(
        func=events.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body|base_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # NON-WORKING (no-op): the Anemos/Crazyflie actuators have stiffness = damping = 0
    # (thrust and torque are applied as a body wrench), so scaling them changes nothing.
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
    reset_action_noise = EventTerm(func=shared_features.reset_action_noise, mode="reset")
    randomize_speed_limits = EventTerm(func=shared_features.randomize_speed_limits, mode="reset")


@configclass
class CurriculumCfg:
    """Racing curriculum; route-adaptive terms are disabled by default."""

    staged = CurrTerm(func=apply_staged_curriculum)
    route_progress_rewards = CurrTerm(
        func=AdaptiveRewardScaleFromRouteProgress,
        params={
            "enabled_attr": "adaptive_curriculum_enabled",
            "completion_term": "lap_complete",
            "reward_term_names": (
                "action_smoothness",
                "gate_centering",
                "velocity_limit",
                "angular_velocity_limit",
            ),
            "increase_only_term_names": ("velocity_limit", "angular_velocity_limit"),
            "eval_interval_s": 120.0,
            "low_ratio_threshold": 0.50,
            "high_ratio_threshold": 0.75,
            "scale_step_fraction": 0.10,
            "min_scale": 1.0,
            "max_scale_default": 5.0,
        },
    )
    randomization_progressive = CurrTerm(
        func=AdaptiveRandomizationFromRouteProgress,
        params={
            "enabled_attr": "adaptive_curriculum_enabled",
            "completion_term": "lap_complete",
            "eval_interval_s": 120.0,
            "low_ratio_threshold": 0.50,
            "high_ratio_threshold": 0.75,
            "scale_step_fraction": 0.10,
            "min_noise_scale": 0.0,
            "max_noise_scale": 1.0,
            "initial_noise_scale": 0.0,
            "randomization_profile": "racing",
            "activate_speed_limits_on_increase": True,
        },
    )


@configclass
class RacingQuadcopterEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for manager-based racing quadcopter.

    Owns everything the env class (:class:`RacingQuadcopterRLEnv`) reads directly (scene,
    terminations, plain runtime fields) as well as racing's drone profile, managers, reward
    preset and domain-randomization flags. Racing is the only user of this schema; the AI-GP
    experimental snapshot carries its own frozen copy.
    """

    seed: int = 42

    scene: RacingQuadcopterSceneCfg = RacingQuadcopterSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=False, clone_in_fabric=False)

    terminations: TerminationsCfg = TerminationsCfg()
    observation: GateObservationConfig = GateObservationConfig()
    action_noise: ActionNoiseCfg = ActionNoiseCfg()
    # Episode-conditioned speed limits; appends [linear_limit, angular_limit] to policy obs when
    # enabled (reward preset decides).
    speed_limits: SpeedLimitsCfg = SpeedLimitsCfg(
        enabled=RACING_REWARD_PRESET.speed_limits_enabled,
        linear_range=(2.0, 15.0),
        angular_range=(1.0, 5.0),
    )

    # Enables the shared route-progress reward/noise/dynamics curriculum below (gates curricula
    # that declare ``enabled_attr``; reward preset decides). Speed limits remain independently
    # controlled by ``speed_limits.enabled``.
    adaptive_curriculum_enabled: bool = RACING_REWARD_PRESET.adaptive_curriculum_enabled
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

    # Motor mixing (:math:`B`, :math:`B^{-1}` built once in the env from the task's drone profile).
    motor_k_t: float = PROFILE.motor_k_t
    motor_k_d: float = PROFILE.motor_k_d
    motor_arm_length_m: float = PROFILE.motor_arm_length_m
    motor_omega_max_rad_s: float = PROFILE.motor_omega_max_rad_s

    #: Threshold Δu_thresh (rad/s): motor-speed penalty counts excess only above this per rotor per step.
    motor_speed_delta_thresh_rad_s: float = 0.1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50,
        render_interval=2,
        # The default 2**21 aggregate-pair buffer is short by one pair per
        # environment at 8192 envs, which makes PhysX drop broad-phase
        # interactions. Keep a power-of-two margin for large sweeps.
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

    # ── Racing managers ─────────────────────────────────────────────────────────
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    enable_fpv_camera: bool = False

    # ── Domain randomization flags ─────────────────────────────────────────────
    # All default to False (no domain randomization). Set True individually to
    # enable the corresponding perturbation. They are consumed in __post_init__, so they must be
    # set in the cfg class (or a subclass default); --override / Hydra env.* come too late.
    # COM shifts have no dynamic effect (thrust is applied at the COM) and actuator-gain
    # randomization is a no-op (zero-stiffness actuators).
    randomize_mass: bool = False              # startup: scale robot mass ±10 %
    randomize_com: bool = False               # startup: shift CoM ±15 mm per axis
    randomize_inertia: bool = False           # startup: scale principal inertias ±20 %
    randomize_actuator_gains: bool = False    # startup: scale stiffness/damping ±10 %
    apply_external_disturbances: bool = False # interval: random force/torque + velocity push
    # NON-WORKING: only logged. Observation noise is controlled by each ObsTerm's noise.enable,
    # which the adaptive curriculum ramps (RACING_REWARD_PRESET=current).
    observation_noise_enable: bool = False

    def _apply_dr_flags(self) -> None:
        """Wire domain-randomization flags into sub-configs. Called by every __post_init__."""
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
            self.events.push_robot = None
        # Per-term noise ``enable`` is the runtime authority.  The group stays
        # enabled so Isaac Lab does not discard those term configurations.
        self.observations.policy.enable_corruption = True

    def _maybe_attach_fpv_camera(self) -> None:
        """Spawn the FPV camera only if `enable_fpv_camera` is True. Called from __post_init__.

        Mount uses the Anemos `camera_keypoint` body — the URDF fixed joint already encodes
        the FPV pose (+X optical axis, -30° pitch about Y). Identity offset with
        convention="world" (forward=+X, up=+Z) keeps the URDF as the single source of truth.
        Only valid for Anemos profiles (crazyflie has no such link).
        """
        if not self.enable_fpv_camera:
            return
        if not DRONE_MODEL.startswith("anemos"):
            raise ValueError(
                f"The FPV camera mounts on the Anemos 'camera_keypoint' body, which the "
                f"{DRONE_MODEL!r} profile does not have. Set "
                f"{DRONE_MODEL_ENV_VAR}=anemos_urdf (see fpv_play.sh)."
            )
        self.scene.fpv_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_keypoint/fpv_camera",
            update_period=1 / 30,
            height=int(getattr(self, "fpv_camera_height", 1080)),
            width=int(getattr(self, "fpv_camera_width", 1920)),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.0,
                focus_distance=400.0,
                horizontal_aperture=2.5,
                clipping_range=(0.01, 1000.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )

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
            debug_vis=False,
        )
        self.sim.render_interval = self.decimation
        self._apply_dr_flags()
        self._maybe_attach_fpv_camera()


@configclass
class RacingQuadcopterEnvCfg_PLAY(RacingQuadcopterEnvCfg):
    """Evaluation config: eval_mode on, fixed spawn at gate 0, no randomisation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.eval_mode = True
        self.fixed_spawn = True
        self.stats_warmup_s = 30.0
        print("[INFO] Evaluation mode is ON.")


# ---------------------------------------------------------------------------
# Fixed-track variants
# ---------------------------------------------------------------------------
# Tracks are defined in YAML under ``configs/tracks/``. The path is set on
# ``RacingQuadcopterEnvCfg_FIXED.gate_track_yaml`` and can be overridden via the
# ``TRACK_YAML`` env variable (read in __post_init__ before terrain
# generation, so it works through subprocess launches like the sweep runner).
#
# Legacy hard-coded layouts kept for reference — see configs/tracks/default.yaml
# and configs/tracks/simple.yaml for the YAML equivalents.
# ---------------------------------------------------------------------------
# FIXED_GATES (full circuit, see configs/tracks/default.yaml):
#   GateSpec(position=[7.5,  2.0, 1.2], rotation=[0.0, 0.0, 0.0]),
#   GateSpec(position=[12.5, 4.5, 2.5], rotation=[0.0, 0.0, 65.0]),
#   GateSpec(position=[13.5, 9.0, 4.5], rotation=[0.0, 0.0, 105.0]),
#   GateSpec(position=[10.5, 13.0, 4.5], rotation=[0.0, 0.0, 160.0]),
#   GateSpec(position=[7.5, 13.5, 4.2], rotation=[0.0, 0.0, 180.0]),  # SPLIT-S entry
#   GateSpec(position=[7.5, 12.0, 1.5], rotation=[0.0, 0.0, 0.0]),    # SPLIT-S exit
#   GateSpec(position=[4.0, 10.5, 2.0], rotation=[0.0, 0.0, 270.0]),
#   GateSpec(position=[1.8,  7.5, 1.5], rotation=[0.0, 0.0, 230.0]),
#   GateSpec(position=[3.5,  3.5, 1.8], rotation=[0.0, 0.0, 350.0]),
#   GateSpec(position=[5.5,  2.0, 2.5], rotation=[0.0, 0.0, 25.0]),
#
# FIXED_GATES_simple (4-gate diamond, see configs/tracks/simple.yaml):
#   GateSpec(position=[7.5, 5.0, 4.5], rotation=[0.0, 0.0, 45.0]),
#   GateSpec(position=[7.5, 7.5, 4.5], rotation=[0.0, 0.0, 315.0]),
#   GateSpec(position=[5.0, 7.5, 4.5], rotation=[0.0, 0.0, 225.0]),
#   GateSpec(position=[5.0, 5.0, 4.5], rotation=[0.0, 0.0, 135.0]),


@configclass
class RacingQuadcopterEnvCfg_FIXED(RacingQuadcopterEnvCfg):
    """Training config with a fixed, deterministic racing circuit.

    The track is loaded from ``gate_track_yaml`` (a path to a track YAML, see
    ``configs/tracks/default.yaml`` for the format). The ``TRACK_YAML``
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
            debug_vis=False,
        )
        self.sim.render_interval = self.decimation
        self._apply_dr_flags()
        self._maybe_attach_fpv_camera()


@configclass
class RacingQuadcopterEnvCfg_FIXED_PLAY(RacingQuadcopterEnvCfg_FIXED):
    """Evaluation config for the fixed circuit: eval_mode on, fixed spawn at gate 0, no randomisation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.eval_mode = True
        self.fixed_spawn = True
        self.stats_warmup_s = 30.0
        self.draw_camera_arrow = True
        print("[INFO] Fixed-track evaluation mode is ON.")


@configclass
class RacingQuadcopterEnvCfg_FIXED_PLAY_FPV(RacingQuadcopterEnvCfg_FIXED_PLAY):
    """Single-drone fixed-track evaluation with an FPV camera attached.

    Identical behaviour to ``RacingQuadcopterEnvCfg_FIXED_PLAY`` (eval mode,
    fixed spawn at gate 0, no randomisation) but:

    - ``num_envs = 1``: only one drone in the world. Required because the FPV
      ``CameraCfg`` is cloned per env via ``{ENV_REGEX_NS}``; rendering a
      camera for each of 1024 envs would tank performance, and bypassing the
      regex (single literal path) breaks the scene's sensor-reset loop, which
      assumes one sensor instance per env.
    - **1×1 terrain grid**: a single 15×15 m circuit instead of 4×4=16
      copies. Same fixed gates as the parent.
    - ``enable_fpv_camera = True``: spawns the camera under the drone's
      ``camera_keypoint`` body. Launch with ``--enable_cameras``.

    View the FPV feed in the Isaac Sim viewport's camera dropdown:
    ``/World/envs/env_0/Robot/camera_keypoint/fpv_camera``.

    Race-scenario knobs (env_0 = "main" drone, env_1 = "opponent"):

    - ``opponent_spawn_gate_idx``: opponent spawns at this gate index instead
      of 0, giving it a head start along the track. Clamped to gate_count - 1.
    - ``draw_target_gate_highlight``: glowing outline overlaid on env_0's
      next gate; tracks ``_current_gate_idx`` and re-authors on gate changes.
    """

    # Race-scenario knobs. Overridable via Hydra / CLI on this task only.
    # Palette: mostly-black + dark-purple sky, white pillars, light-blue main, purple opponent,
    # flickering purple↔white target gate.
    draw_target_gate_highlight: bool = True
    target_gate_highlight_color: tuple[float, float, float] = (0.75, 0.15, 1.0)  # deeper purple
    target_gate_highlight_emissive: float = 180.0
    target_gate_highlight_width: float = 0.10                                    # thinner
    # Flicker stays in the purple family — alt color is pale lavender, not pure white.
    target_gate_flicker_freq_hz: float = 10.0
    target_gate_flicker_depth: float = 0.7
    target_gate_flicker_jitter: float = 0.3
    target_gate_flicker_alt_color: tuple[float, float, float] = (0.85, 0.55, 1.0)  # pale purple
    opponent_spawn_gate_idx: int | None = 1

    # Drone glow markers — pulled back so they no longer wash surroundings to white.
    main_glow_color: tuple[float, float, float] | None = (0.3, 0.7, 1.0)        # light blue
    main_glow_radius: float = 0.0                                                # ignored
    main_glow_intensity: float = 0.0                                             # ignored
    main_glow_light_radius: float = 0.03
    main_glow_light_intensity: float = 2.0e7                                     # ~4× softer

    opponent_glow_color: tuple[float, float, float] | None = (0.75, 0.20, 1.0)   # purple
    opponent_glow_radius: float = 0.0                                            # ignored
    opponent_glow_intensity: float = 0.0                                         # ignored
    opponent_glow_light_radius: float = 0.03
    opponent_glow_light_intensity: float = 2.0e7                                 # ~4× softer

    # Drone light trails — thinner + much brighter.
    draw_drone_trails: bool = True
    main_trail_color: tuple[float, float, float] = (0.3, 0.7, 1.0)        # light blue
    opponent_trail_color: tuple[float, float, float] = (0.75, 0.20, 1.0)  # purple
    trail_length: int = 40                                                 # ~1.6 s at 25 Hz
    trail_width: float = 0.015
    trail_emissive: float = 320.0                                          # very strong

    # Stadium perimeter pillars — white alternating with light blue (no orange).
    draw_stadium_pillars: bool = True
    stadium_pillar_count: int = 12
    stadium_pillar_ring_radius: float = 9.5
    stadium_pillar_radius: float = 0.3
    stadium_pillar_height: float = 10.0
    stadium_pillar_colors: tuple[tuple[float, float, float], ...] = (
        (1.0, 1.0, 1.0),    # white
        (0.3, 0.7, 1.0),    # light blue
    )
    stadium_pillar_emissive: float = 120.0

    # Wet reflective ground (neon mirrors off it).
    draw_wet_ground: bool = True
    wet_ground_size: float = 80.0
    wet_ground_color: tuple[float, float, float] = (0.010, 0.010, 0.018)   # near-black cool
    wet_ground_metallic: float = 0.8
    wet_ground_roughness: float = 0.15

    # Holographic floor grid.
    draw_holographic_grid: bool = True
    holo_grid_size: float = 30.0
    holo_grid_spacing: float = 1.5
    holo_grid_width: float = 0.03
    holo_grid_color: tuple[float, float, float] = (0.3, 0.7, 1.0)          # light blue
    holo_grid_emissive: float = 55.0

    # Neon walls between pillars — alternating white + light blue.
    draw_neon_walls: bool = True
    neon_wall_heights: tuple[float, ...] = (0.6, 2.0, 4.0, 6.0, 8.0, 10.0)
    neon_wall_colors: tuple[tuple[float, float, float], ...] = (
        (1.0, 1.0, 1.0),
        (0.3, 0.7, 1.0),
        (1.0, 1.0, 1.0),
        (0.3, 0.7, 1.0),
        (1.0, 1.0, 1.0),
        (0.3, 0.7, 1.0),
    )
    neon_wall_width: float = 0.06
    neon_wall_emissive: float = 90.0

    # Hemispherical dome cap above the pillars — alternating white + light blue.
    draw_dome_cap: bool = True
    dome_cap_radius: float = 9.5
    dome_cap_meridians: int = 12
    dome_cap_latitudes: int = 5
    dome_cap_segments: int = 18
    dome_cap_colors: tuple[tuple[float, float, float], ...] = (
        (1.0, 1.0, 1.0),
        (0.3, 0.7, 1.0),
    )
    dome_cap_width: float = 0.06
    dome_cap_emissive: float = 100.0

    # Sky accents — purple + light blue + white dots (no orange).
    draw_sky_accents: bool = True
    sky_accent_count: int = 220
    sky_accent_palette: tuple[tuple[float, float, float], ...] = (
        (0.7, 0.2, 1.0),    # purple
        (0.45, 0.1, 0.9),   # deeper purple
        (0.3, 0.7, 1.0),    # light blue
        (1.0, 1.0, 1.0),    # white
    )
    sky_accent_radius_range: tuple[float, float] = (0.04, 0.12)
    sky_accent_emissive: float = 180.0

    # White outline overlay on every gate.
    draw_all_gate_outlines: bool = True
    all_gate_outline_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    all_gate_outline_emissive: float = 18.0
    all_gate_outline_width: float = 0.06

    # Party lights — drone-glow-style SphereLights at random arena positions, each
    # flickering on/off on its own schedule and periodically swapping colors. Looks
    # like a chaotic concert/festival light show.
    draw_party_lights: bool = True
    party_light_count: int = 28
    # Radial annulus the lights spawn in — default biases toward the perimeter so the
    # arena walls/dome are populated instead of just the floor.
    party_light_plan_radius_range: tuple[float, float] = (3.0, 9.4)
    party_light_height_range: tuple[float, float] = (1.5, 14.0)
    party_light_palette: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.3, 0.0),    # orange
        (0.3, 0.7, 1.0),    # light blue
        (0.75, 0.20, 1.0),  # purple
        (1.0, 1.0, 1.0),    # white
        (0.0, 1.0, 0.5),    # mint
        (1.0, 0.1, 0.6),    # hot pink
    )
    party_light_body_radius: float = 0.04
    party_light_base_intensity: float = 2.0e7        # ~4× softer than before — no washout
    party_light_freq_range: tuple[float, float] = (0.4, 2.2)             # Hz
    party_light_sharpness_range: tuple[float, float] = (1.5, 4.0)        # higher = sharper on/off
    party_light_color_change_period_range: tuple[float, float] = (2.5, 8.0)  # seconds
    party_light_seed: int = 42

    # FPV camera resolution (set on the CameraCfg; affects render cost). Read in __post_init__
    # when the camera is built, so change it here: --override / Hydra env.* have no effect.
    fpv_camera_width: int = 1920
    fpv_camera_height: int = 1080

    # Automatic FPV video recording (env_0 camera → MP4). Encoded with imageio +
    # libx264 at CRF 14 (visually lossless). Frames are pulled in step() after the
    # camera is updated.
    record_fpv_video: bool = True
    video_output_dir: str = "logs/fpv_videos"
    video_filename: str = ""                       # empty → auto-generated timestamp
    video_fps: int = 0                             # 0 → auto = 1/step_dt → playback matches real-time
    video_length_s: float = 30.0
    video_crf: int = 14                            # x264 CRF: lower = higher quality

    # Cyberpunk atmosphere: mostly black sky tinted dark purple, purple fog.
    enable_cyberpunk_atmosphere: bool = True
    cyber_fog_color: tuple[float, float, float] = (0.05, 0.01, 0.12)       # dark purple
    cyber_fog_intensity: float = 1.0
    cyber_fog_start: float = 2.0
    cyber_fog_end: float = 90.0
    cyber_fog_density: float = 0.022
    cyber_chromatic_strength: float = 0.005
    cyber_dome_intensity: float = 25.0                                     # very dim (mostly black sky)
    cyber_dome_color: tuple[float, float, float] = (0.06, 0.02, 0.12)      # dark purple

    def __post_init__(self) -> None:
        # Must be set before super(), since super().__post_init__ attaches the camera.
        self.enable_fpv_camera = True
        super().__post_init__()
        # Override the 1024-env grid set by parent.
        self.scene.num_envs = 2
        # Rebuild the terrain as a single 1×1 cell instead of 4×4.
        track_path = _os.environ.get(TRACK_YAML_ENV_VAR, self.gate_track_yaml)
        gates = _load_gates_from_yaml(track_path)
        terrain_generator = create_fixed_track_terrain(
            gates, seed=self.seed, num_rows=1, num_cols=1
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

        self.draw_camera_arrow = False
        self.draw_trajectory = False
        self.draw_gate_indicators = False
        self.draw_gate_labels = False

        # Race mode: drones loop the circuit until they crash. Disable lap-complete
        # termination (it fires after one lap in eval mode) and push the episode
        # length far enough out that the timeout never triggers in practice.
        self.terminations.lap_complete = None
        self.terminations.time_out = None
        self.episode_length_s = 1.0e6

        # The adaptive curricula key off the lap_complete termination, which no longer
        # exists here. This is a demo config, so no curriculum should run anyway.
        self.adaptive_curriculum_enabled = False

        print("[INFO] FPV race evaluation mode is ON (main=env_0, opponent=env_1).")


# ---------------------------------------------------------------------------
# Skill evaluation: N fixed circuits, one per
# terrain, each with its own spawn rule (upside-down, normal, recovery, sensor delay).
#
# Each terrain is configured via a SkillTerrainCfg that specifies:
#   - which track YAML to load (from configs/skill_tracks/)
#   - per-terrain spawn overrides (e.g. spawn_upside_down)
#
# Proportions are derived dynamically from len(skill_terrains) so adding or removing a terrain
# requires no manual proportion bookkeeping.
#
# This config is evaluation-only — there is no training variant.
# ---------------------------------------------------------------------------
_SKILL_TRACKS_DIR = _os.path.join(_REPO_ROOT, "configs", "skill_tracks")


@configclass
class SkillTerrainCfg:
    """Configuration for a single skill-evaluation terrain.

    Add new boolean/float fields here to expose further per-terrain settings;
    read spawn settings inside ``reset_skill_eval_spawn`` in ``mdp/events.py``.
    ``obs_delay_steps`` is applied by ``RacingQuadcopterSkillEvalRLEnv.load_managers``.
    """

    track_yaml: str = _os.path.join(_SKILL_TRACKS_DIR, "default_difficult.yaml")
    spawn_upside_down: bool = False
    spawn_recovery: bool = False
    obs_delay_steps: int = 0
    """Number of policy (env) steps to delay the policy observation (0 = no delay).

    Simulates sensor/communication latency: the policy receives the observation
    that was computed ``obs_delay_steps`` steps ago instead of the current one.
    """


@configclass
class RacingQuadcopterSkillEvalEnvCfg(RacingQuadcopterEnvCfg):
    """Skill evaluation: N fixed circuits in N terrains, each with its own spawn logic.

    ``skill_terrains`` is the single source of truth: its length determines the
    number of terrains (proportions are computed as ``1 / len(skill_terrains)``),
    and each entry's fields control both which track to build and how to spawn
    drones when they reset.

    The default configuration has four terrains:
      - Terrain 0: upside-down challenge — drones spawn inverted.
      - Terrain 1: default difficult run — normal deterministic spawn.
      - Terrain 2: recovery challenge — drones spawn with random attitude/velocity.
      - Terrain 3: sensor delay — policy receives observations delayed by 5 policy steps
        (5 × step_dt = 200 ms at the default 1/50 s dt × decimation 2).
    """

    skill_terrains: list = [
        SkillTerrainCfg(
            track_yaml=_os.path.join(_SKILL_TRACKS_DIR, "upside_down_challenge.yaml"),
            spawn_upside_down=True,
        ),
        SkillTerrainCfg(
            track_yaml=_os.path.join(_SKILL_TRACKS_DIR, "default_difficult.yaml"),
            spawn_upside_down=False,
        ),
        SkillTerrainCfg(
            track_yaml=_os.path.join(_SKILL_TRACKS_DIR, "recovery_challenge.yaml"),
            spawn_recovery=True,
        ),
        SkillTerrainCfg(
            track_yaml=_os.path.join(_SKILL_TRACKS_DIR, "sensor_delay_challenge.yaml"),
            obs_delay_steps=5,
        ),
    ]

    # Gate geometry shared across all skill terrains; override per-run if needed.
    skill_gate_size: float = 0.8
    skill_bar_thickness: float = 0.05
    skill_gate_color: tuple[float, float, float] = (1.0, 0.3, 0.0)

    def __post_init__(self) -> None:
        if not self.skill_terrains:
            raise ValueError("skill_terrains must not be empty.")

        track_yamls = [t.track_yaml for t in self.skill_terrains]
        terrain_generator = create_multi_track_terrain(
            track_yamls,
            gate_size=self.skill_gate_size,
            bar_thickness=self.skill_bar_thickness,
            gate_color=self.skill_gate_color,
            seed=self.seed,
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
            debug_vis=True,
        )
        self.sim.render_interval = self.decimation

        # Evaluation defaults.
        self.scene.num_envs = 256
        self.eval_mode = True
        self.force_collision_termination = True
        self.observations.policy.enable_corruption = False
        self.action_noise.enable = False

        # IsaacLab nulls every term_cfg.noise when enable_corruption is False
        # (ObservationManager), but the adaptive curricula bind "noise.enable" on
        # those term cfgs. Disable them here: this is an evaluation env, and the
        # staged curriculum already self-disables under eval_mode.
        self.adaptive_curriculum_enabled = False
        # This __post_init__ does not call super(), so _apply_dr_flags never runs:
        # null every DR event explicitly.
        self.events.randomize_robot_mass = None
        self.events.randomize_robot_actuator_gains = None
        self.events.randomize_base_com = None
        self.events.randomize_robot_inertia = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.stats_warmup_s = 30.0

        # Replace the standard spawn event with the per-terrain dispatcher.
        # reset_skill_eval_spawn reads env._env_col_idx (set by the env class) to
        # route each env to its terrain's spawn logic.
        self.events.reset_racing_spawn = EventTerm(
            func=events.reset_skill_eval_spawn,
            mode="reset",
        )

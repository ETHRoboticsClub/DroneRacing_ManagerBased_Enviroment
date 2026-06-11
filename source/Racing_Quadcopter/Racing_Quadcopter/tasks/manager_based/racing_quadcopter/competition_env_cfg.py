"""Competition env config — fixed 6-gate descending track + custom racing drone.

Reuses the same ``RacingQuadcopterRLEnv`` runtime; only the configuration differs
from ``racing_quadcopter``:

* a deterministic 6-gate course loaded from ``configs/tracks/competition.yaml``
  (sized to a 175 x 28 m tile so the ~136 m / 26 m-descent course fits),
* competition gate geometry: 1.5 x 1.5 m inner aperture, frame extending only
  OUTWARD (0.6 m per side -> 2.7 m outer), 0.26 m deep,
* a custom racing-drone profile (``COMPETITION_PROFILE``), flown with the
  collective-thrust + body-rates action (inner-loop PID -> body torques),
* a (24 + 25*k)-dim observation: pose, body velocities, current + next gate inner corners, and
  last action (49 per-frame channels; the stacked terms carry a k-step history),
* collision termination always active (no curriculum gate) and episode
  termination upon passing the final gate (training and eval),
* a self-contained domain-randomization block (``CompetitionEventsCfg`` plus the
  ``domain_randomization`` switch) — it does not depend on the base flag wiring.
"""

from __future__ import annotations

import numpy as np
import os
import scipy.spatial.transform as tf
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply

from Racing_Quadcopter.assets.robots_cfg import ANEMOS_DRONE_CFG_URDF
from Racing_Quadcopter.terrains import load_gates_from_yaml
from Racing_Quadcopter.terrains.racing_gates_cfg import MeshFixedGatesTerrainCfg

from .mdp import events, observations
from .mdp.actions import QuadcopterRatesActionCfg
from .mdp.noise import drone_noise_model_factory
from .racing_quadcopter_env_cfg import RacingQuadcopterEnvCfg, TRACK_YAML_ENV_VAR, _make_profile

# ---------------------------------------------------------------------------
# Track frame
# ---------------------------------------------------------------------------
# Uniform shift applied when converting the source NED track to the tile's
# corner-origin frame:  tile = NWU + COMPETITION_TRACK_OFFSET, where
# NWU = (N, -E, -D).
#
# DO NOT CHANGE — must stay identical to the offset baked into
# configs/tracks/competition.yaml. The absolute-position observations below are
# reported in the tile (YAML) frame, so this constant is what maps them back to
# the original track coordinates:  NWU = obs_position - COMPETITION_TRACK_OFFSET.
COMPETITION_TRACK_OFFSET: tuple[float, float, float] = (165.0, 12.0, 28.0)

# ---------------------------------------------------------------------------
# Gate geometry: 1.5 m square inner aperture; frame members are 0.6 m wide and
# stretch only OUTWARD from the aperture (outer = 1.5 + 2*0.6 = 2.7 m), 0.26 m deep.
# ---------------------------------------------------------------------------
COMPETITION_GATE_INNER_SIZE = 1.5    # inner aperture side length [m] (passage check + corner obs)
COMPETITION_GATE_OUTER_SIZE = 2.7    # outer frame side length [m]
COMPETITION_GATE_DEPTH = 0.26        # frame depth along the passage axis [m]
_GATE_FRAME_WIDTH = (COMPETITION_GATE_OUTER_SIZE - COMPETITION_GATE_INNER_SIZE) / 2.0  # 0.6 m

COMPETITION_TILE_SIZE = (175.0, 28.0)  # (x, y) tile [m]; contains the full course

COMPETITION_PHYSICS_HZ = 120.0  # PhysX rate fixed by the competition; not configurable

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)
COMPETITION_TRACK_YAML = os.path.join(_REPO_ROOT, "configs", "tracks", "competition.yaml")


def create_competition_gate_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    gate_size: float,
    bar_thickness: float,
    color: tuple[float, float, float] | None = None,
) -> list[trimesh.Trimesh]:
    """Build the competition gate: 4 box members forming a square ring around the aperture.

    Drop-in replacement for ``gate_base.create_gate_frame`` (same signature, so it works as a
    ``MeshFixedGatesTerrainCfg.gate_frame_fn``) but with different geometry: that one centres the
    members on the aperture edges so they intrude into the opening, whereas here each member sits
    fully OUTSIDE the aperture. The inner opening stays exactly ``gate_size`` square and the outer
    silhouette is ``gate_size + 2 * _GATE_FRAME_WIDTH`` square; ``bar_thickness`` sets the member
    depth along the passage (+Y) axis.
    """
    inner_half = gate_size / 2.0
    outer = gate_size + 2.0 * _GATE_FRAME_WIDTH
    member_centre = inner_half + _GATE_FRAME_WIDTH / 2.0  # aperture edge + half member width

    quat_scipy = [orientation[1], orientation[2], orientation[3], orientation[0]]
    rot_mat = tf.Rotation.from_quat(quat_scipy).as_matrix()

    # (local_centre_offset, box_dims) in the gate frame (X right, Y depth, Z up).
    # Top/bottom span the full outer width; left/right fill the inner height between them.
    bar_specs = [
        ("top",    np.array([0.0, 0.0, +member_centre]), np.array([outer, bar_thickness, _GATE_FRAME_WIDTH])),
        ("bottom", np.array([0.0, 0.0, -member_centre]), np.array([outer, bar_thickness, _GATE_FRAME_WIDTH])),
        ("left",   np.array([-member_centre, 0.0, 0.0]), np.array([_GATE_FRAME_WIDTH, bar_thickness, gate_size])),
        ("right",  np.array([+member_centre, 0.0, 0.0]), np.array([_GATE_FRAME_WIDTH, bar_thickness, gate_size])),
    ]

    meshes = []
    for _, local_offset, dims in bar_specs:
        bar = trimesh.creation.box(dims)
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = position + rot_mat @ local_offset
        bar.apply_transform(transform)
        if color is not None:
            bar.visual.face_colors = [int(c * 255) for c in color] + [255]
        meshes.append(bar)

    return meshes


# ---------------------------------------------------------------------------
# Custom racing-drone profile.
# A ~0.85 kg 5"-class racing airframe (Anemos URDF, converted on the fly). The
# control parameters are tuned for that airframe; the env consumes
# ``thrust_to_weight`` + the body-rate PID gains for the collective-thrust /
# body-rates action (no direct motor mixing is used here).
# ---------------------------------------------------------------------------
COMPETITION_PROFILE = _make_profile(
    ANEMOS_DRONE_CFG_URDF,
    body_name="base_link",
    thrust_to_weight=4.1056,
    moment_scale=(1.16, 1.40, 0.020),  # unused by the rates action; kept for completeness
    motor_k_t=8.54858e-6,
    motor_k_d=1e-8,
    motor_arm_length_m=0.110,
    motor_omega_max_rad_s=1000.0,
    rate_command_scale=30.0,           # full stick -> 30 rad/s body-rate setpoint
    pid_scales=0.005,
    pid_p=(45.0, 47.0, 0.64),
    pid_i=(0.0, 0.0, 0.0),
    pid_d=(32.0, 34.0, 0.45),
    pid_f=(0.0, 0.0, 0.0),
)


# ---------------------------------------------------------------------------
# Observations (track frame = the YAML corner-origin tile frame)
# ---------------------------------------------------------------------------

def _track_origin_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env world position of the track frame's origin [N, 3] (cached).

    Each terrain tile holds an identical copy of the track, so the origin is
    recovered per env as (gate-0 world position) - (gate-0 YAML position). Static
    for the whole run.
    """
    cached = getattr(env, "_competition_track_origin_w", None)
    if cached is None:
        gate0_yaml = env.cfg.scene.terrain.terrain_generator.sub_terrains["racing_gates"].gates[0].position
        cached = env._gate_positions[:, 0, :] - torch.tensor(
            gate0_yaml, device=env.device, dtype=torch.float32
        )
        env._competition_track_origin_w = cached
    return cached


def position_track_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Absolute drone position in the track (YAML) frame [N, 3].

    Subtract COMPETITION_TRACK_OFFSET to recover the NWU track coordinates.
    """
    return env._robot.data.root_pos_w - _track_origin_w(env)


def rotation_matrix_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body-to-world rotation matrix, row-major flattened [N, 9]."""
    return matrix_from_quat(env._robot.data.root_quat_w).reshape(env.num_envs, 9)


# Inner-corner offsets in the gate frame (X right, Z up; passage along +Y),
# unit half-aperture: NW(-x,+z), NE(+x,+z), SE(+x,-z), SW(-x,-z) as seen by the
# approaching drone.
_CORNER_SIGNS = ((-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


def gate_inner_corners_track_frame(env: ManagerBasedRLEnv, gate_offset: int = 0) -> torch.Tensor:
    """Inner-corner positions (NW, NE, SE, SW) of the gate at ``current + gate_offset``,
    in the track frame, flattened [N, 12].

    The index is clamped at the final gate (no wrap-around): on a point-to-point
    course the "next" gate of the last gate is the last gate itself.
    """
    idx = torch.minimum(env._current_gate_idx + gate_offset, env._gate_counts - 1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    gate_pos = env._gate_positions[env_ids, idx]        # [N, 3] world
    gate_ori = env._gate_orientations[env_ids, idx]     # [N, 4] wxyz
    half = (env._gate_sizes * 0.5).unsqueeze(-1)        # [N, 1] inner half-aperture

    signs = torch.tensor(_CORNER_SIGNS, device=env.device)         # [4, 2]
    corners_local = torch.zeros(env.num_envs, 4, 3, device=env.device)
    corners_local[:, :, 0] = signs[:, 0] * half                    # X (right)
    corners_local[:, :, 2] = signs[:, 1] * half                    # Z (up)

    quat_exp = gate_ori.unsqueeze(1).expand(-1, 4, -1)             # [N, 4, 4]
    corners_w = gate_pos.unsqueeze(1) + quat_apply(quat_exp, corners_local)
    corners_track = corners_w - _track_origin_w(env).unsqueeze(1)
    return corners_track.reshape(env.num_envs, 12)


# Observation history window. Module-level on purpose: a class-body attribute named
# ``history_length`` would set the inherited ObsGroup field and override every term's window
# (including the gate corners), and any other class-body var is mis-read as an observation term.
_OBS_WINDOW = 3


@configclass
class CompetitionObservationsCfg:
    """Policy observation: pose + velocities + gate inner corners + last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        enable_corruption: bool = False  # toggles to True (activates noise) if domain_randomization is on
        concatenate_terms: bool = True

        # (3) absolute position, track frame (NWU = value - COMPETITION_TRACK_OFFSET)
        position = ObsTerm(
            func=position_track_frame,
            noise=drone_noise_model_factory(std_noise=0.02, std_bias=0.01, dropout=0.0),
            history_length=_OBS_WINDOW,
        )
        # (9) body-to-world rotation matrix
        rotation_matrix = ObsTerm(
            func=rotation_matrix_obs,
            history_length=_OBS_WINDOW,
        )
        # (3) body-frame linear velocity
        base_lin_vel = ObsTerm(
            func=observations.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(std_noise=0.05, std_bias=0.01, dropout=0.0),
            history_length=_OBS_WINDOW,
        )
        # (3) body-frame angular velocity
        base_ang_vel = ObsTerm(
            func=observations.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(std_noise=0.02, std_bias=0.005, dropout=0.0),
            history_length=_OBS_WINDOW,
        )
        # (3) gravity direction in body frame
        projected_gravity = ObsTerm(
            func=observations.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=drone_noise_model_factory(std_noise=0.02, std_bias=0.005, dropout=0.0),
            history_length=_OBS_WINDOW,
        )
        # (12) inner corners of the current gate (NW, NE, SE, SW), track frame
        current_gate_corners = ObsTerm(
            func=gate_inner_corners_track_frame,
            params={"gate_offset": 0},
            history_length=1,  # Keep at 1, breaks otherwise (positions jump with gate passages)
        )
        # (12) inner corners of the next gate, track frame
        next_gate_corners = ObsTerm(
            func=gate_inner_corners_track_frame,
            params={"gate_offset": 1},
            history_length=1,  # Keep at 1, breaks otherwise (positions jump with gate passages)
        )
        # (4) previous action
        last_action = ObsTerm(
            func=observations.last_action,
            history_length=_OBS_WINDOW,
        )

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

def final_gate_passed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True on the step the drone passes the LAST gate of the track. Shape [N].

    After a valid passage of gate ``counts - 1``, ``_current_gate_idx`` wraps to 0
    while ``_gate_passed_buf`` is still 1 — that combination only occurs on the
    final-gate passage step. Active in training AND eval (unlike the base
    ``lap_complete``, which is eval-only).
    """
    return (env._gate_passed_buf > 0.5) & (env._current_gate_idx == 0)


def upside_down(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage-independent upside-down termination (the competition has no curriculum).

    The shared ``terminations.upside_down`` only fires during curriculum stage 0; since
    the competition runs a single fixed regime, this fires whenever the body +Z tips
    past the threshold, in both training and eval. Shape [N].
    """
    return env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@configclass
class CompetitionCurriculumCfg:
    """No staged curriculum — the competition trains in a single fixed regime.

    Removing the staged terms decouples the env from ``_curriculum_stage``: there is
    no stage-0/1/2 progression, no reward-weight boosting, and no recovery spawns turning
    on mid-training. Collision termination is instead made always-on directly (via the
    trivial curriculum_* thresholds set in __post_init__), independent of this manager.
    """


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _set_spawned_at_origin(env: ManagerBasedRLEnv, env_ids: torch.Tensor, value: bool) -> None:
    """Mark whether ``env_ids`` spawned at the offset origin (vs a per-gate box spawn).

    Lazily allocates the per-env bool buffer. Read by the stats pipeline to count metrics
    only for origin-spawned drones (see ``cfg.stats_origin_only``).
    """
    buf = getattr(env, "_spawned_at_origin", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._spawned_at_origin = buf
    buf[env_ids] = value


def _spawn_at_offset_origin(env: ManagerBasedRLEnv, env_ids: torch.Tensor, speed: float = 0.0) -> None:
    """Place ``env_ids`` at the track's (offset) origin.

    NWU origin = tile-frame COMPETITION_TRACK_OFFSET (so the ``position`` obs reads the
    offset and NWU = obs - offset = 0), facing down-track (gate-0 heading), with gate 0 as
    the first target. ``_spawn_gate_idx`` is 0, so these count as from-start runs.

    ``speed`` = 0 spawns at rest (the cold eval / true competition start). ``speed`` > 0
    gives a "warm" start: an initial velocity straight toward gate 0 (down-track), with a
    per-env magnitude in [0.5, 1.0] * speed — used for training so the policy isn't learning
    to arrest a fall from a dead stop 23 m out.
    """
    n = len(env_ids)
    env._spawn_gate_idx[env_ids] = 0
    env._current_gate_idx[env_ids] = 0
    env._gates_passed[env_ids] = 0
    _set_spawned_at_origin(env, env_ids, True)

    offset = torch.tensor(COMPETITION_TRACK_OFFSET, device=env.device, dtype=torch.float32)
    start_pos = _track_origin_w(env)[env_ids] + offset       # world position of the (offset) origin
    start_quat = env._gate_orientations[env_ids, 0]          # face down-track, like gate 0
    root_vel = torch.zeros(n, 6, device=env.device)
    if speed > 0.0:
        # Gate-0 forward (passage axis) in world = the down-track direction toward gate 0.
        fwd = quat_apply(start_quat, env._y_local.expand(n, -1))
        mag = speed * (0.5 + 0.5 * torch.rand(n, device=env.device))   # [0.5, 1.0] * speed
        root_vel[:, :3] = mag.unsqueeze(-1) * fwd
    events._write_spawn_state(env, env_ids, start_pos, start_quat, root_vel)


def reset_competition_origin_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Test-mode spawn: every drone at the (offset) origin. Used by ``CompetitionEnvCfg_PLAY``.

    Cross-env collisions are handled by the GPU broad-phase, so overlapping drones are fine.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    _spawn_at_offset_origin(env, env_ids)


def reset_competition_training_spawn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Training spawn: power-law over [offset-origin, gate 0 … gate N-1], origin first.

    The offset origin is prepended as the highest-weighted slot of the same power-law that
    ranks the gates (``cfg.spawn_gate_bias``): weight(slot k) ∝ k**bias for k = N+1 (origin)
    down to 1 (last gate). Implemented as: spawn at the origin with probability
    (N+1)**bias / Σ_{k=1}^{N+1} k**bias, else use the existing per-gate spawn
    (``_apply_uniform_initialization``, whose internal gate power-law integrates to the
    remaining mass — so the combined distribution is exactly the prepended one).
    """
    if env_ids is None or len(env_ids) == 0:
        return
    # Origin slot disabled → plain per-gate uniform spawn (the known-working start).
    if not bool(getattr(env.cfg, "origin_spawn_in_training", True)):
        _set_spawned_at_origin(env, env_ids, False)
        events._apply_uniform_initialization(env, env_ids)
        return
    n = len(env_ids)
    bias = float(getattr(env.cfg, "spawn_gate_bias", 1.0))
    counts = env._gate_counts[env_ids].float()                       # N gates per env
    max_slots = int((env._gate_counts[env_ids].max() + 1).item())    # N+1 (origin + gates)
    ks = torch.arange(1, max_slots + 1, device=env.device).float()
    kpow = ks.pow(bias)
    valid = ks.unsqueeze(0) <= (counts + 1.0).unsqueeze(1)           # [n, max_slots]: k <= N+1
    denom = torch.where(valid, kpow.unsqueeze(0), torch.zeros_like(valid, dtype=kpow.dtype)).sum(dim=1)
    p_origin = (counts + 1.0).pow(bias) / denom                      # [n]

    use_origin = torch.rand(n, device=env.device) < p_origin
    origin_ids = env_ids[use_origin]
    gate_ids = env_ids[~use_origin]
    if len(origin_ids) > 0:
        # Warm origin: initial velocity toward gate 0 so it isn't a dead-stop start.
        _spawn_at_offset_origin(env, origin_ids, speed=float(getattr(env.cfg, "origin_spawn_speed", 8.0)))
    if len(gate_ids) > 0:
        _set_spawned_at_origin(env, gate_ids, False)
        events._apply_uniform_initialization(env, gate_ids)


@configclass
class CompetitionEventsCfg:
    """Self-contained reset + domain-randomization pipeline for the competition env.

    Reset terms always run. The randomization terms (mass, actuator gains, external
    force/torque) are set to None by __post_init__ when ``domain_randomization`` is False.
    """

    # --- reset (always on) ---
    reset_scene_to_default = EventTerm(func=events.reset_scene_to_default, mode="reset")
    # Training spawn: warm offset-origin (top-weighted slot, moving toward gate 0) + per-gate
    # power-law. The Play cfg overrides this with reset_competition_origin_spawn (cold, at rest).
    reset_racing_spawn = EventTerm(func=reset_competition_training_spawn, mode="reset")

    # --- domain randomization (toggled in __post_init__) ---
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
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (-0.02, 0.02),
            "torque_range": (-0.02, 0.02),
        },
    )


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------

@configclass
class CompetitionEnvCfg(RacingQuadcopterEnvCfg):
    """Training config: fixed competition course, custom drone, always-on collisions."""

    gate_track_yaml: str = COMPETITION_TRACK_YAML

    # Standalone observation + event pipelines (override the base configs).
    observations: CompetitionObservationsCfg = CompetitionObservationsCfg()
    events: CompetitionEventsCfg = CompetitionEventsCfg()
    # No staged curriculum: fixed-regime env, fully decoupled from _curriculum_stage.
    curriculum: CompetitionCurriculumCfg = CompetitionCurriculumCfg()

    # Single switch for all domain randomization (mass / actuator gains / external
    # disturbances / observation noise). Set False (or the Play cfg) to disable.
    domain_randomization: bool = True

    # Spawn-gate distribution (power-law exponent on rank-from-end; see
    # events._sample_spawn_gate). 1.0 = linear → 6:1 first:last on this 6-gate track.
    spawn_gate_bias: float = 1.0

    # Count ALL stats (avg/top speed, gates/ep, lap%, lap time, and the TB Episode_Metrics)
    # only for drones that spawned at the offset origin — the real competition start. In eval
    # every drone spawns there, so this only filters training (origin slot vs gate-box spawns).
    stats_origin_only: bool = True

    # Include the (warm) offset-origin as the top-weighted training spawn slot; set False to
    # train on the plain per-gate uniform spawn only.
    origin_spawn_in_training: bool = True

    # Warm-origin training spawn: initial speed (m/s) toward gate 0 for drones that draw the
    # origin slot (per-env magnitude [0.5, 1.0] * this), so it isn't a dead-stop start. Eval
    # spawns stay cold (at rest). 0 = cold origin.
    origin_spawn_speed: float = 8.0

    # Policy (action) rate [Hz]. Physics runs at the competition-fixed COMPETITION_PHYSICS_HZ;
    # decimation = COMPETITION_PHYSICS_HZ / control_hz, so control_hz must divide it evenly.
    # The rates-PID integrates at the physics step, so changing this may need PID re-tuning.
    control_hz: float = 60.0   # decimation = 120 / 60 = 2

    def __post_init__(self) -> None:
        # --- custom racing drone ---
        self.scene.robot = COMPETITION_PROFILE.robot_cfg
        self.motor_k_t = COMPETITION_PROFILE.motor_k_t
        self.motor_k_d = COMPETITION_PROFILE.motor_k_d
        self.motor_arm_length_m = COMPETITION_PROFILE.motor_arm_length_m
        self.motor_omega_max_rad_s = COMPETITION_PROFILE.motor_omega_max_rad_s

        # --- collective-thrust + body-rates action (inner-loop PID), no moments ---
        self.actions.moments = None
        self.actions.rates = QuadcopterRatesActionCfg(
            asset_name="robot",
            thrust_to_weight=COMPETITION_PROFILE.thrust_to_weight,
            action_dim=4,
            body_name=COMPETITION_PROFILE.body_name,
            pid_scales=COMPETITION_PROFILE.pid_scales,
            pid_p=COMPETITION_PROFILE.pid_p,
            pid_i=COMPETITION_PROFILE.pid_i,
            pid_d=COMPETITION_PROFILE.pid_d,
            pid_f=COMPETITION_PROFILE.pid_f,
            rate_command_scale=COMPETITION_PROFILE.rate_command_scale,
        )

        # --- always-on collision termination ---
        # collision_termination fires when (contact > threshold) AND the per-env
        # curriculum gate is open. Trivially-satisfiable thresholds keep that gate
        # open for every env from episode 0, so hard contacts always terminate.
        self.crash_termination = True
        self.curriculum_min_gates_passed = 0
        self.curriculum_min_mean_velocity = 0.0
        self.curriculum_max_center_offset = 1.0e9

        # --- terminate upon passing the final gate (training and eval) ---
        self.terminations.lap_complete = DoneTerm(func=final_gate_passed, time_out=True)
        # Stage-independent upside-down termination (base one only fires at curriculum stage 0).
        self.terminations.upside_down = DoneTerm(func=upside_down, time_out=False)

        # --- fixed competition terrain (large tile holds the long descending course) ---
        track_path = os.environ.get(TRACK_YAML_ENV_VAR, self.gate_track_yaml)
        gates = load_gates_from_yaml(track_path)
        terrain_generator = TerrainGeneratorCfg(
            seed=self.seed,
            size=COMPETITION_TILE_SIZE,
            difficulty_range=(0.0, 0.0),  # ignored by fixed_gates_terrain
            border_width=0.5,
            num_rows=4,
            num_cols=4,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "racing_gates": MeshFixedGatesTerrainCfg(
                    proportion=1.0,
                    seed=self.seed,
                    gates=gates,
                    gate_size=COMPETITION_GATE_INNER_SIZE,   # passage check + corner obs use this
                    bar_thickness=COMPETITION_GATE_DEPTH,     # frame depth (see gate_frame_fn)
                    gate_frame_fn=create_competition_gate_frame,
                ),
            },
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

        # Longer course -> longer episode.
        self.episode_length_s = 30.0

        # --- physics (competition-fixed) / control rates ---
        decimation = int(round(COMPETITION_PHYSICS_HZ / self.control_hz))
        assert decimation >= 1 and abs(COMPETITION_PHYSICS_HZ - decimation * self.control_hz) < 1e-6, (
            f"control_hz ({self.control_hz}) must divide COMPETITION_PHYSICS_HZ "
            f"({COMPETITION_PHYSICS_HZ}) evenly; got decimation={COMPETITION_PHYSICS_HZ / self.control_hz}."
        )
        self.sim.dt = 1.0 / COMPETITION_PHYSICS_HZ
        self.decimation = decimation
        self.sim.render_interval = decimation
        self.scene.contact_sensor.update_period = self.sim.dt  # refresh contacts every physics step

        # No trajectory polyline overlay.
        self.draw_trajectory = False

        # --- standalone domain randomization (no base _apply_dr_flags) ---
        if not self.domain_randomization:
            self.events.randomize_robot_mass = None
            self.events.randomize_robot_actuator_gains = None
            self.events.base_external_force_torque = None
        self.observations.policy.enable_corruption = self.domain_randomization


@configclass
class CompetitionEnvCfg_PLAY(CompetitionEnvCfg):
    """Evaluation config: eval_mode on, all drones spawn at the track origin, no randomization."""

    domain_randomization: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1024
        self.eval_mode = True
        self.fixed_spawn = True
        self.stats_warmup_s = 5.0
        # Test mode spawns at the (offset) track origin instead of behind gate 0.
        self.events.reset_racing_spawn = EventTerm(func=reset_competition_origin_spawn, mode="reset")
        print("[INFO] Competition evaluation mode is ON.")

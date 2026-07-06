"""Competition env config — fixed 6-gate descending track + custom racing drone.

Reuses the same ``RacingQuadcopterRLEnv`` runtime; only the configuration differs
from ``racing_quadcopter``:

* a deterministic 6-gate course loaded from ``configs/tracks/competition.yaml``
  (sized to a 175 x 28 m tile so the ~136 m / 26 m-descent course fits),
* competition gate geometry: 1.5 x 1.5 m inner aperture, frame extending only
  OUTWARD (0.6 m per side -> 2.7 m outer), 0.26 m deep,
* a custom racing-drone profile (``COMPETITION_PROFILE``), flown with the
  collective-thrust + second-order body-rates action,
* a (24 + 25*k)-dim observation: pose, body velocities, current + next gate inner corners, and
  last action (49 per-frame channels; the stacked terms carry a k-step history),
* collision termination always active (no curriculum gate) and episode
  termination upon passing the final gate (training and eval),
* a self-contained domain-randomization block (``CompetitionEventsCfg`` plus the
  ``domain_randomization`` switch) — it does not depend on the base flag wiring.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
import os
import scipy.spatial.transform as tf
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

from Racing_Quadcopter.assets.robots_cfg import ANEMOS_DRONE_CFG_URDF
from Racing_Quadcopter.terrains import load_gates_from_yaml
from Racing_Quadcopter.terrains.racing_gates_cfg import MeshFixedGatesTerrainCfg

from .mdp import events, observations
from .mdp.actions import QuadcopterRatesActionCfg, QuadcopterSecondOrderRatesActionCfg
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
COMPETITION_TARGET_HOVER_ACTION = 0.276
COMPETITION_TARGET_THRUST_TO_WEIGHT = 1.0 / COMPETITION_TARGET_HOVER_ACTION # 2.0 / (1.0 + COMPETITION_TARGET_HOVER_ACTION)

COMPETITION_MAX_BODY_RATE_COMMAND = 6.5 # rad/s

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
# control parameters are tuned for that airframe; the competition env consumes
# ``thrust_to_weight`` + body-rate command limits for the collective-thrust /
# second-order body-rates action (no direct motor mixing is used here).
# ---------------------------------------------------------------------------
COMPETITION_PROFILE = _make_profile(
    ANEMOS_DRONE_CFG_URDF,
    body_name="base_link",
    thrust_to_weight=COMPETITION_TARGET_THRUST_TO_WEIGHT,
    moment_scale=(1.16, 1.40, 0.020),  # unused by the rates action; kept for completeness
    motor_k_t=8.54858e-6,
    motor_k_d=1e-8,
    motor_arm_length_m=0.110,
    motor_omega_max_rad_s=1000.0,
    rate_command_scale=COMPETITION_MAX_BODY_RATE_COMMAND,          
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


def gate_inner_corners_body_frame(env: ManagerBasedRLEnv, gate_offset: int = 0) -> torch.Tensor:
    """Inner-corner positions (NW, NE, SE, SW) of the gate at ``current + gate_offset``,
    expressed in drone body frame and flattened [N, 12].

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
    drone_pos_w = env._robot.data.root_pos_w
    drone_quat_w = env._robot.data.root_quat_w
    corners_rel_w = corners_w - drone_pos_w.unsqueeze(1)
    drone_quat_exp = drone_quat_w.unsqueeze(1).expand(-1, 4, -1)
    corners_body = quat_apply_inverse(drone_quat_exp, corners_rel_w)
    return corners_body.reshape(env.num_envs, 12)

# Observation history window. Module-level on purpose: a class-body attribute named
# ``history_length`` would set the inherited ObsGroup field and override every term's window
# (including the gate corners), and any other class-body var is mis-read as an observation term.
_OBS_WINDOW = 1


@configclass
class CompetitionObservationsCfg:
    """Policy observation: pose + velocities + gate inner corners + last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        enable_corruption: bool = False  # toggles to True (activates noise) if domain_randomization is on
        concatenate_terms: bool = True

        # absolute position removed; current gate corners are provided directly in body frame
        # (9) body-to-world rotation matrix
        # rotation_matrix = ObsTerm(
        #     func=rotation_matrix_obs,
        #     history_length=_OBS_WINDOW,
        # )
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
        # (12) inner corners of the current gate (NW, NE, SE, SW), body frame
        current_gate_corners = ObsTerm(
            func=gate_inner_corners_body_frame,
            params={"gate_offset": 0},
            history_length=1,  # Keep at 1, breaks otherwise (positions jump with gate passages)
        )
        # (12) inner corners of the next gate, body frame
        # next_gate_corners = ObsTerm(
        #     func=gate_inner_corners_body_frame,
        #     params={"gate_offset": 1},
        #     history_length=1,  # Keep at 1, breaks otherwise (positions jump with gate passages)
        # )
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

class adaptive_reward_scale_from_lap_completion_ratio(ManagerTermBase):
    """Windowed curriculum controller that scales selected reward-term weights by name."""

    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reward_term_names = tuple(cfg.params["reward_term_names"])
        self._reward_term_cfgs = {
            term_name: env.reward_manager.get_term_cfg(term_name) for term_name in self._reward_term_names
        }
        self._base_weights = {term_name: float(term_cfg.weight) for term_name, term_cfg in self._reward_term_cfgs.items()}
        self._term_scales = {term_name: 1.0 for term_name in self._reward_term_names}
        self._window_lap_complete_resets = 0.0
        self._window_total_resets = 0.0
        self._last_eval_step = int(env.common_step_counter)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        lap_complete_term: str = "lap_complete",
        reward_term_names: tuple[str, ...] = ("action_smoothness", "gate_centering"),
        eval_interval_s: float = 120.0,
        low_ratio_threshold: float = 0.50,
        high_ratio_threshold: float = 0.75,
        scale_step_fraction: float = 0.10,
        min_scale: float = 1.0,
        max_scale_default: float = 5.0,
        max_scale_by_term: dict[str, float] | None = None,
    ) -> dict[str, float]:
        # Keep run-time params and constructor-time term set aligned.
        if tuple(reward_term_names) != self._reward_term_names:
            raise ValueError(
                "reward_term_names changed after initialization. "
                f"Initialized with {self._reward_term_names}, received {tuple(reward_term_names)}."
            )

        # Accumulate reset statistics for this call.
        env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
        if env_ids_tensor.numel() > 0:
            lap_complete_mask = env.termination_manager.get_term(lap_complete_term)[env_ids_tensor]
            self._window_lap_complete_resets += float(lap_complete_mask.float().sum().item())
            self._window_total_resets += float(env_ids_tensor.numel())

        # Evaluate at fixed wall-time in simulation seconds (using RL-step timebase).
        interval_steps = max(1, int(round(float(eval_interval_s) / float(env.step_dt))))
        curr_step = int(env.common_step_counter)
        updated = 0.0
        ratio = (
            self._window_lap_complete_resets / self._window_total_resets
            if self._window_total_resets > 0.0
            else 0.0
        )
        reported_total_resets = self._window_total_resets
        reported_lap_complete_resets = self._window_lap_complete_resets
        if (curr_step - self._last_eval_step) >= interval_steps:
            step_frac = max(0.0, float(scale_step_fraction))
            lo = float(min_scale)
            decision = "hold"
            if ratio < float(low_ratio_threshold):
                scale_factor = max(0.0, 1.0/(1+step_frac))
                decision = "decrease"
            elif ratio > float(high_ratio_threshold):
                scale_factor = 1.0 + step_frac
                decision = "increase"
            else:
                scale_factor = 1.0

            max_by_term = {} if max_scale_by_term is None else {k: float(v) for k, v in max_scale_by_term.items()}
            for term_name in self._reward_term_names:
                prev_scale = self._term_scales[term_name]
                hi = max(lo, max_by_term.get(term_name, float(max_scale_default)))
                new_scale = float(max(lo, min(hi, prev_scale * scale_factor)))
                self._term_scales[term_name] = new_scale
                term_cfg = self._reward_term_cfgs[term_name]
                term_cfg.weight = self._base_weights[term_name] * new_scale
                env.reward_manager.set_term_cfg(term_name, term_cfg)

            updated = 1.0

            term_scale_summary = ", ".join(f"{k}={v:.3f}" for k, v in self._term_scales.items())
            print(
                "[CompetitionCurriculum] Window update: "
                f"lap_complete_ratio={ratio:.3f}, "
                f"window_lap_complete_resets={reported_lap_complete_resets:.0f}, "
                f"window_total_resets={reported_total_resets:.0f}, "
                f"decision={decision}, "
                f"reward_scales={{ {term_scale_summary} }}"
            )

            self._window_lap_complete_resets = 0.0
            self._window_total_resets = 0.0
            self._last_eval_step = curr_step

        out = {
            "lap_complete_ratio_window": ratio,
            "window_total_resets": reported_total_resets,
            "window_lap_complete_resets": reported_lap_complete_resets,
            "updated": updated,
        }
        for term_name in self._reward_term_names:
            out[f"reward_scale/{term_name}"] = self._term_scales[term_name]
            out[f"reward_weight/{term_name}"] = float(self._reward_term_cfgs[term_name].weight)
        return out


class adaptive_randomization_from_lap_completion_ratio(ManagerTermBase):
    """Windowed curriculum controller that applies generic, config-defined parameter scaling."""

    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._window_lap_complete_resets = 0.0
        self._window_total_resets = 0.0
        self._last_eval_step = int(env.common_step_counter)
        self._target_specs = tuple(cfg.params["targets"])
        self._bindings: list[dict[str, object]] = []
        self._initialize_bindings(env, self._target_specs)
        self._noise_scale = float(cfg.params["initial_noise_scale"])
        self._apply_scale_to_bindings(self._noise_scale)

    @staticmethod
    def _get_attr_path(root_obj: object, attr_path: str) -> object:
        obj = root_obj
        for name in attr_path.split("."):
            obj = getattr(obj, name)
        return obj

    @staticmethod
    def _set_attr_path(root_obj: object, attr_path: str, value: object) -> None:
        parts = attr_path.split(".")
        obj = root_obj
        for name in parts[:-1]:
            obj = getattr(obj, name)
        setattr(obj, parts[-1], value)

    @staticmethod
    def _resolve_target_root(env: ManagerBasedRLEnv, spec: dict[str, object]) -> tuple[object, callable | None]:
        target_type = str(spec["target_type"])
        if target_type == "env_cfg":
            return env.cfg, None
        if target_type == "reward_term_cfg":
            term_name = str(spec["term_name"])
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            return term_cfg, lambda cfg_obj: env.reward_manager.set_term_cfg(term_name, cfg_obj)
        if target_type == "action_term_cfg":
            term_name = str(spec["term_name"])
            # IMPORTANT: bind to the live action term cfg (ActionManager deep-copies cfg at init),
            # otherwise mutating env.cfg.actions.<term> won't affect the running term.
            action_term = env.action_manager.get_term(term_name)
            root_obj = action_term.cfg
            return root_obj, None
        if target_type == "observation_group_cfg":
            group_name = str(spec["group_name"])
            obs_cfg = env.observation_manager.cfg
            if isinstance(obs_cfg, dict):
                if group_name not in obs_cfg:
                    raise ValueError(
                        f"Observation group '{group_name}' not found. Available groups: {tuple(obs_cfg.keys())}."
                    )
                return obs_cfg[group_name], None
            if not hasattr(obs_cfg, group_name):
                raise ValueError(
                    f"Observation group '{group_name}' not found. "
                    f"Available groups: {tuple(k for k in obs_cfg.__dict__.keys() if not k.startswith('_'))}."
                )
            return getattr(obs_cfg, group_name), None
        if target_type == "observation_term_cfg":
            group_name = str(spec["group_name"])
            term_name = str(spec["term_name"])
            group_term_names = env.observation_manager._group_obs_term_names.get(group_name, [])
            if term_name not in group_term_names:
                raise ValueError(
                    f"Observation term '{term_name}' not found in group '{group_name}'. "
                    f"Available terms: {tuple(group_term_names)}."
                )
            term_idx = group_term_names.index(term_name)
            return env.observation_manager._group_obs_term_cfgs[group_name][term_idx], None
        raise ValueError(f"Unknown randomization target_type '{target_type}'.")

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return isinstance(value, (tuple, list))

    @staticmethod
    def _lerp_scalar(start: float, target: float, scale: float) -> float:
        return float(start) + float(scale) * (float(target) - float(start))

    def _scale_value(self, spec: dict[str, object], scale: float, start_value: object, target_value: object) -> object:
        mode = spec["mode"]
        value: object
        if isinstance(target_value, bool):
            threshold = float(spec["bool_threshold"])
            value = bool(target_value) if float(scale) >= threshold else bool(start_value)
        elif self._is_sequence(target_value):
            start_seq = tuple(float(v) for v in (start_value if self._is_sequence(start_value) else target_value))
            target_seq = tuple(float(v) for v in target_value)
            if mode != "lerp":
                raise ValueError(f"Unsupported mode '{mode}' for sequence target.")
            value = tuple(self._lerp_scalar(s, t, scale) for s, t in zip(start_seq, target_seq, strict=True))
        elif isinstance(target_value, (int, float)):
            s = float(start_value)
            t = float(target_value)
            if mode == "lerp":
                value = self._lerp_scalar(s, t, scale)
            elif mode == "mul":
                value = t * float(scale)
            else:
                raise ValueError(f"Unsupported numeric mode '{mode}'.")
        else:
            raise TypeError(f"Unsupported target value type: {type(target_value)}")

        if "clamp_min" in spec and isinstance(value, (int, float)):
            value = max(float(spec["clamp_min"]), float(value))
        if "clamp_max" in spec and isinstance(value, (int, float)):
            value = min(float(spec["clamp_max"]), float(value))
        if "dtype" in spec and str(spec["dtype"]) == "int":
            value = int(round(float(value)))
        return value

    def _initialize_bindings(self, env: ManagerBasedRLEnv, target_specs: tuple[dict[str, object], ...]) -> None:
        for idx, spec in enumerate(target_specs):
            root_obj, post_apply_hook = self._resolve_target_root(env, spec)
            attr_path = str(spec["attr_path"])
            current_value = self._get_attr_path(root_obj, attr_path)
            target_value = spec["target_value"]
            start_value = spec["start_value"]
            if start_value is None:
                if isinstance(target_value, bool):
                    start_value = False
                elif self._is_sequence(target_value):
                    start_value = tuple(0.0 for _ in target_value)
                elif isinstance(target_value, (int, float)):
                    start_value = 0.0
                else:
                    start_value = target_value
            self._bindings.append(
                {
                    "name": str(spec.get("name", f"target_{idx}")),
                    "spec": spec,
                    "root_obj": root_obj,
                    "post_apply_hook": post_apply_hook,
                    "attr_path": attr_path,
                    "start_value": start_value,
                    "target_value": target_value,
                    "last_value": current_value,
                }
            )

    def _apply_scale_to_bindings(self, scale: float) -> None:
        for binding in self._bindings:
            spec = binding["spec"]
            root_obj = binding["root_obj"]
            attr_path = binding["attr_path"]
            value = self._scale_value(spec, scale, binding["start_value"], binding["target_value"])
            self._set_attr_path(root_obj, attr_path, value)
            post_apply_hook = binding["post_apply_hook"]
            if post_apply_hook is not None:
                post_apply_hook(root_obj)
            binding["last_value"] = value

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        lap_complete_term: str = "lap_complete",
        eval_interval_s: float = 120.0,
        low_ratio_threshold: float = 0.50,
        high_ratio_threshold: float = 0.75,
        scale_step_fraction: float = 0.10,
        min_noise_scale: float = 0.05,
        max_noise_scale: float = 1.0,
        initial_noise_scale: float = 0.20,
        targets: Tuple[Dict[str, Any]] = tuple()
    ) -> dict[str, float]:
        del initial_noise_scale  # consumed in __init__
        del targets # consumed in __init__
        env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
        if env_ids_tensor.numel() > 0:
            lap_complete_mask = env.termination_manager.get_term(lap_complete_term)[env_ids_tensor]
            self._window_lap_complete_resets += float(lap_complete_mask.float().sum().item())
            self._window_total_resets += float(env_ids_tensor.numel())

        interval_steps = max(1, int(round(float(eval_interval_s) / float(env.step_dt))))
        curr_step = int(env.common_step_counter)
        updated = 0.0
        ratio = (
            self._window_lap_complete_resets / self._window_total_resets
            if self._window_total_resets > 0.0
            else 0.0
        )
        reported_total_resets = self._window_total_resets
        reported_lap_complete_resets = self._window_lap_complete_resets

        if (curr_step - self._last_eval_step) >= interval_steps:
            step_frac = max(0.0, float(scale_step_fraction))
            lo = float(min_noise_scale)
            hi = float(max_noise_scale)
            if hi < lo:
                hi = lo

            prev_scale = self._noise_scale
            decision = "hold"
            # Additive growth avoids the absorbing-zero bug of pure multiplicative updates.
            additive_scale_update = step_frac * (1.0 - self._noise_scale)
            if ratio < float(low_ratio_threshold):
                self._noise_scale -= max(0.0, additive_scale_update)
                decision = "decrease"
            elif ratio > float(high_ratio_threshold):
                self._noise_scale += additive_scale_update
                decision = "increase"
            self._noise_scale = float(max(lo, min(hi, self._noise_scale)))
            self._apply_scale_to_bindings(self._noise_scale)
            updated = 1.0

            sampled = []
            sample_names = {
                "action_noise_enable",
                "action_noise_gaussian_std",
                "action_noise_bias_range",
                "obs_base_lin_vel_enable",
                "obs_base_lin_vel_std",
                "obs_base_lin_vel_bias_std",
                "obs_base_ang_vel_enable",
                "obs_base_ang_vel_std",
                "obs_base_ang_vel_bias_std",
                "obs_gravity_enable",
                "obs_gravity_std",
                "obs_gravity_bias_std",
            }
            for binding in self._bindings:
                if binding["name"] in sample_names:
                    sampled.append(f"{binding['name']}={binding['last_value']}")
            sample_summary = ", ".join(sampled)
            print(
                "[CompetitionNoiseCurriculum] Window update: "
                f"lap_complete_ratio={ratio:.3f}, "
                f"window_lap_complete_resets={reported_lap_complete_resets:.0f}, "
                f"window_total_resets={reported_total_resets:.0f}, "
                f"decision={decision}, "
                f"noise_scale={prev_scale:.3f}->{self._noise_scale:.3f}, "
                f"targets=[{sample_summary}]"
            )

            self._window_lap_complete_resets = 0.0
            self._window_total_resets = 0.0
            self._last_eval_step = curr_step

        out = {
            "lap_complete_ratio_window": ratio,
            "window_total_resets": reported_total_resets,
            "window_lap_complete_resets": reported_lap_complete_resets,
            "updated": updated,
            "randomization_noise_scale": self._noise_scale,
        }
        for binding in self._bindings:
            value = binding["last_value"]
            if isinstance(value, bool):
                out[f"target/{binding['name']}"] = float(value)
            elif isinstance(value, (int, float)):
                out[f"target/{binding['name']}"] = float(value)
        return out


@configclass
class CompetitionCurriculumCfg:
    """Competition curriculum: periodic reward-scale adaptation from lap-completion ratio."""

    precision_after_lap_completion_windowed = CurrTerm(
        func=adaptive_reward_scale_from_lap_completion_ratio,
        params={
            "lap_complete_term": "lap_complete",
            "reward_term_names": ("action_smoothness", "gate_centering"),
            "eval_interval_s": 120.0,
            "low_ratio_threshold": 0.50,
            "high_ratio_threshold": 0.75,
            "scale_step_fraction": 2.0,
            "min_scale": 1.0,
            "max_scale_default": 5.0,
            "max_scale_by_term": {
                "action_smoothness": 400.0,
                "gate_centering": 40.0,
            },
        },
    )
    randomization_progressive = CurrTerm(
        func=adaptive_randomization_from_lap_completion_ratio,
        params={
            "lap_complete_term": "lap_complete",
            "eval_interval_s": 40.0,
            "low_ratio_threshold": 0.5,
            "high_ratio_threshold": 0.75,
            "scale_step_fraction": 0.08,
            "min_noise_scale": 0.0,
            "max_noise_scale": 1.0,
            "initial_noise_scale": 0.0,
            "targets": (
                {
                    "name": "action_noise_enable",
                    "target_type": "env_cfg",
                    "attr_path": "action_noise.enable",
                    "mode": "lerp",
                    "start_value": False,
                    "target_value": True,
                    "bool_threshold": 1.0e-6,
                },
                {
                    "name": "obs_base_lin_vel_enable",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "base_lin_vel",
                    "attr_path": "noise.enable",
                    "mode": "lerp",
                    "start_value": False,
                    "target_value": True,
                    "bool_threshold": 1.0e-6,
                },
                {
                    "name": "obs_base_ang_vel_enable",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "base_ang_vel",
                    "attr_path": "noise.enable",
                    "mode": "lerp",
                    "start_value": False,
                    "target_value": True,
                    "bool_threshold": 1.0e-6,
                },
                {
                    "name": "obs_gravity_enable",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "projected_gravity",
                    "attr_path": "noise.enable",
                    "mode": "lerp",
                    "start_value": False,
                    "target_value": True,
                    "bool_threshold": 1.0e-6,
                },
                {
                    "name": "action_noise_gaussian_std",
                    "target_type": "env_cfg",
                    "attr_path": "action_noise.gaussian_std",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.02,
                },
                {
                    "name": "action_noise_bias_range",
                    "target_type": "env_cfg",
                    "attr_path": "action_noise.bias_range",
                    "mode": "lerp",
                    "start_value": (0.0, 0.0),
                    "target_value": (-0.05, 0.05),
                },
                {
                    "name": "action_noise_lpf_alpha_range",
                    "target_type": "env_cfg",
                    "attr_path": "action_noise.lpf_alpha_range",
                    "mode": "lerp",
                    "start_value": (0.9, 1.0),
                    "target_value": (0.8, 1.0),
                },
                {
                    "name": "rates_max_delay",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "max_delay",
                    "mode": "lerp",
                    "start_value": 0,
                    "target_value": 2,
                    "dtype": "int",
                    "clamp_min": 0,
                },
                {
                    "name": "rates_filter_wn_rand_frac",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "rate_filter_natural_frequency_randomization_frac",
                    "mode": "lerp",
                    "start_value": (0.0, 0.0, 0.0),
                    "target_value": (0.3, 0.3, 0.3),
                },
                {
                    "name": "rates_filter_zeta_rand_frac",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "rate_filter_damping_randomization_frac",
                    "mode": "lerp",
                    "start_value": (0.0, 0.0, 0.0),
                    "target_value": (0.3, 0.3, 0.3),
                },
                {
                    "name": "rates_filter_gain_rand_frac",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "rate_filter_command_gain_randomization_frac",
                    "mode": "lerp",
                    "start_value": (0.0, 0.0, 0.0),
                    "target_value": (0.2, 0.2, 0.2),
                },
                {
                    "name": "rates_thrust_to_weight_rand_frac",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "thrust_to_weight_randomization_frac",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.05,
                },
                {
                    "name": "rates_aero_randomization_frac",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "aero_randomization_frac",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.1,
                },
                {
                    "name": "rates_linear_drag_coeff",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "linear_drag_coeff",
                    "mode": "lerp",
                    "start_value": (0.0, 0.0, 0.0),
                    "target_value": (0.08, 0.08, 0.12),
                },
                {
                    "name": "rates_quadratic_drag_coeff",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "quadratic_drag_coeff",
                    "mode": "lerp",
                    "start_value": (0.0, 0.0, 0.0),
                    "target_value": (0.010, 0.010, 0.018),
                },
                {
                    "name": "rates_angular_drag_coeff",
                    "target_type": "action_term_cfg",
                    "term_name": "rates",
                    "attr_path": "angular_drag_coeff",
                    "mode": "lerp",
                    "start_value": (0.0, 0.0, 0.0),
                    "target_value": (0.0020, 0.0020, 0.0012),
                },
                {
                    "name": "obs_base_lin_vel_std",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "base_lin_vel",
                    "attr_path": "noise.noise_cfg.std",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.2,
                },
                {
                    "name": "obs_base_lin_vel_bias_std",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "base_lin_vel",
                    "attr_path": "noise.bias_noise_cfg.std",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 1.0,
                },
                {
                    "name": "obs_base_ang_vel_std",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "base_ang_vel",
                    "attr_path": "noise.noise_cfg.std",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.05,
                },
                {
                    "name": "obs_base_ang_vel_bias_std",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "base_ang_vel",
                    "attr_path": "noise.bias_noise_cfg.std",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.2,
                },
                {
                    "name": "obs_gravity_std",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "projected_gravity",
                    "attr_path": "noise.noise_cfg.std",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.02,
                },
                {
                    "name": "obs_gravity_bias_std",
                    "target_type": "observation_term_cfg",
                    "group_name": "policy",
                    "term_name": "projected_gravity",
                    "attr_path": "noise.bias_noise_cfg.std",
                    "mode": "lerp",
                    "start_value": 0.0,
                    "target_value": 0.04,
                },
            ),
        },
    )


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
            "mass_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_base_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (-0.03, 0.03)},
        },
    )
    randomize_robot_inertia = EventTerm(
        func=events.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "inertia_distribution_params": (0.6, 1.6),
            "operation": "scale",
            "distribution": "uniform",
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
            "force_range": (-0.8, 0.8),
            "torque_range": (-0.04, 0.04),
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
    # The second-order rate model integrates at the physics step.
    control_hz: float = 60.0   # decimation = 120 / 60 = 2

    def __post_init__(self) -> None:
        # --- custom racing drone ---
        self.scene.robot = COMPETITION_PROFILE.robot_cfg
        self.motor_k_t = COMPETITION_PROFILE.motor_k_t
        self.motor_k_d = COMPETITION_PROFILE.motor_k_d
        self.motor_arm_length_m = COMPETITION_PROFILE.motor_arm_length_m
        self.motor_omega_max_rad_s = COMPETITION_PROFILE.motor_omega_max_rad_s

        # --- collective-thrust + PID-less second-order body-rates action ---
        self.actions.moments = None
        # self.actions.rates = QuadcopterRatesActionCfg(
        self.actions.rates = QuadcopterSecondOrderRatesActionCfg (
            asset_name="robot",
            thrust_to_weight=COMPETITION_PROFILE.thrust_to_weight,
            action_dim=4,
            min_delay=0,
            max_delay=2,
            body_name=COMPETITION_PROFILE.body_name,
            pid_scales=COMPETITION_PROFILE.pid_scales,
            pid_p=COMPETITION_PROFILE.pid_p,
            pid_i=COMPETITION_PROFILE.pid_i,
            pid_d=COMPETITION_PROFILE.pid_d,
            pid_f=COMPETITION_PROFILE.pid_f,
            rate_command_scale=COMPETITION_PROFILE.rate_command_scale,
            pid_gain_randomization_frac=None,
            thrust_to_weight_randomization_frac=0.05,
            aero_randomization_enable=True,
            aero_randomization_frac=0.1,
            linear_drag_coeff=(0.08, 0.08, 0.12),
            quadratic_drag_coeff=(0.010, 0.010, 0.018),
            angular_drag_coeff=(0.0020, 0.0020, 0.0012),
            
            rate_filter_enable=True,
            rate_filter_natural_frequency_hz=(7.5, 7.5, 4.5),
            rate_filter_natural_frequency_randomization_frac=(0.1, 0.1, 0.1),
            rate_filter_damping_ratio=(0.75, 0.75, 0.90),
            rate_filter_damping_randomization_frac=(0.1, 0.1, 0.1),
            rate_filter_command_gain=(1.0, 1.0, 1.0),
            rate_filter_command_gain_randomization_frac=(0.1, 0.1, 0.1),
            rate_filter_max_command_rad_s=COMPETITION_PROFILE.rate_command_scale,
        )

        # Action-level perturbations model command transport, calibration drift, and actuator
        # uncertainty before the target simulator's hidden rate controller receives commands.
        self.action_noise.enable = self.domain_randomization
        self.action_noise.per_episode_bias = True
        self.action_noise.gaussian_std = 0.015
        self.action_noise.bias_range = (-0.03, 0.03)
        self.action_noise.per_episode_lpf_alpha = True
        self.action_noise.lpf_alpha = 0.85
        self.action_noise.lpf_alpha_range = (0.85, 1.0)
        self.action_noise.clip_actions = True

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
            self.events.randomize_base_com = None
            self.events.randomize_robot_inertia = None
            self.events.randomize_robot_actuator_gains = None
            self.events.base_external_force_torque = None
            self.action_noise.enable = False
            self.actions.rates.min_delay = 0
            self.actions.rates.max_delay = 0
            self.actions.rates.pid_gain_randomization_frac = None
            self.actions.rates.thrust_to_weight_randomization_frac = None
            self.actions.rates.rate_filter_natural_frequency_randomization_frac = None
            self.actions.rates.rate_filter_damping_randomization_frac = None
            self.actions.rates.rate_filter_command_gain_randomization_frac = None
            self.actions.rates.aero_randomization_frac = None
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
        # disable curriculum
        self.curriculum.precision_after_lap_completion_windowed = None
        self.curriculum.randomization_progressive = None
        # Test mode spawns at the (offset) track origin instead of behind gate 0.
        self.events.reset_racing_spawn = EventTerm(func=reset_competition_origin_spawn, mode="reset")
        print("[INFO] Competition evaluation mode is ON.")

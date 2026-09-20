# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Skill-evaluation multi-track config: N fixed circuits, one per terrain.

Each terrain is configured via a :class:`SkillTerrainCfg` that specifies:
  - which track YAML to load (from ``configs/skill_tracks/``)
  - per-terrain spawn overrides (e.g. ``spawn_upside_down``)

Proportions are derived dynamically from ``len(skill_terrains)`` so adding or
removing a terrain requires no manual proportion bookkeeping.

This config is **evaluation-only** — there is no training variant.
"""

from __future__ import annotations

import os as _os

from Racing_Quadcopter.terrains import create_multi_track_terrain

from .mdp import events
# All isaaclab symbols re-imported from the parent cfg module so this file has no
# direct isaaclab imports (avoids IDE path-resolution warnings in environments where
# isaaclab is installed as an editable package not on Pylance's search path).
from .racing_quadcopter_env_cfg import (
    EventTerm,
    RacingQuadcopterEnvCfg,
    TerrainImporterCfg,
    _REPO_ROOT,
    configclass,
    sim_utils,
)

_SKILL_TRACKS_DIR = _os.path.join(_REPO_ROOT, "configs", "skill_tracks")


@configclass
class SkillTerrainCfg:
    """Configuration for a single skill-evaluation terrain.

    Add new boolean/float fields here to expose further per-terrain settings;
    read them inside ``reset_skill_eval_spawn`` in ``mdp/events.py``.
    """

    track_yaml: str = _os.path.join(_SKILL_TRACKS_DIR, "default_difficult.yaml")
    spawn_upside_down: bool = False
    spawn_recovery: bool = False
    obs_delay_steps: int = 0
    """Number of simulation steps to delay the policy observation (0 = no delay).

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
      - Terrain 3: sensor delay — policy receives observations delayed by 5 steps (~50 ms).
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
        self.events.randomize_robot_mass = None
        self.events.randomize_robot_actuator_gains = None
        self.stats_warmup_s = 30.0

        # Replace the standard spawn event with the per-terrain dispatcher.
        # reset_skill_eval_spawn reads env._env_col_idx (set by the env class) to
        # route each env to its terrain's spawn logic.
        self.events.reset_racing_spawn = EventTerm(
            func=events.reset_skill_eval_spawn,
            mode="reset",
        )

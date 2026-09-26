Changelog
---------

Unreleased (2026-09-26)
~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* ``RACING_REWARD_PRESET`` environment variable (``may`` default, ``current``) to select the
  racing reward set.
* ``ai_gp_quadcopter`` task: two frozen, self-contained snapshots of the AI Grand Prix
  competition course, ``june11`` (default) and ``experimental``, selected with
  ``AI_GP_VARIANT`` / ``AI_GP_TRACK_YAML``.
* One shared ``mdp/`` package for racing and hovering MDP terms (observations, actions,
  rewards, events, terminations, curricula).
* ``docs/INDEX.md``: entry point that explains the code layout and links every document.
* NON-WORKING / STALE notes on configs, sweeps and scripts that do not run as written.

Changed
^^^^^^^

* **Breaking:** renamed the competition task to AI-GP: task IDs
  ``Isaac-Racing-Quadcopter-Competition-*`` → ``Isaac-AI-GP-Quadcopter-*``; environment
  variables ``COMPETITION_VARIANT`` / ``COMPETITION_TRACK_YAML`` →
  ``AI_GP_VARIANT`` / ``AI_GP_TRACK_YAML``; log directories ``competition_*`` → ``ai_gp_*``;
  ``configs/tracks/competition.yaml`` → ``configs/tracks/ai_gp.yaml``.
* **Breaking:** racing's default reward preset is now ``may`` (the May 2026 weights that train
  reliably); the previous shared weights are kept as the ``current`` preset. Racing checkpoints
  trained on the old defaults need ``RACING_REWARD_PRESET=current``.
* ``racing_quadcopter/`` consolidated from 7 ``.py`` files plus a ``vis/`` package down to 4
  files: ``__init__.py``, ``racing_quadcopter_env.py`` (now also holds
  ``RacingQuadcopterMultiTrackRLEnv`` and ``RacingQuadcopterSkillEvalRLEnv``),
  ``racing_quadcopter_env_cfg.py`` (now also holds the former ``gate_racing_base_cfg.py``,
  ``racing_randomization.py`` and the SkillEval config), and ``trajectory_vis.py`` (moved out of
  ``vis/``).
* Hovering's MDP terms moved into the shared ``mdp/`` package, with renames:
  ``last_action`` → ``last_applied_action``, ``penalty_crash`` →
  ``penalty_out_of_bounds_or_upside_down``, ``collect_episode_metrics_before_reset`` →
  ``collect_waypoint_episode_metrics_before_reset``.
* Hovering skrl ``rewards_shaper_scale`` 0.01 → 1.0 (weights are direct values, matching
  rsl_rl) and PhysX aggregate-pair buffer raised to match racing.
* Hovering Play grades waypoint collection with the trained thresholds (0.3 m/s, 1 rad/s,
  tilt 0.3) instead of the curriculum start values.

Fixed
^^^^^

* Crash penalty now also fires on upside-down (previously contact-only despite the docs).
* Evaluation (``eval_mode``) now always terminates the episode on collision.
* Fixed-spawn evaluation grid now ranks drones per terrain cell (previously could spawn up to
  78 m behind gate 0).
* Ground-spawn (and AI-GP experimental origin-spawn) nose heading now points along the
  gate passage axis (was 90° off).
* SkillEval no longer runs domain-randomization events during evaluation (its config skips
  ``super().__post_init__``, so the randomization flags never switched them off).
* Raw-action buffers are actually cleared on reset.
* ``randomize_rigid_body_inertia`` now passes the full tensor (previously crashed on partial
  resets).
* Noise curriculum now backs off symmetrically after a collapse (previously struggled to lower
  high noise).
* ``sweep.py`` now sets the ``TRACK_YAML`` variable the envs actually read (it previously
  exported ``RACING_GATE_TRACK_YAML``, so inline sweep tracks were silently ignored and CI
  regression sweeps always flew ``default.yaml``).
* ``sweep.py`` ``DIRECT_*`` keys now map 1:1 to the reward weight (previously scaled ×100
  through a since-removed ``MANAGER_SCALE``).
* ``actuator_model`` in the rates action config is now validated (``direct`` or
  ``betaflight_sitl``) instead of silently accepting typos.
* ``play.sh`` no longer passes the unsupported ``--video_interval`` flag; ``train.sh`` also
  sets ``HOVERING_DRONE_MODEL``.
* ``--experiment_name`` is applied in ``scripts/rsl_rl/cli_args.py``, matching upstream
  Isaac Lab.
* Gate roll/pitch docstring corrected in ``terrains/racing_gates_cfg.py`` (roll and pitch were
  swapped).

Removed
^^^^^^^

* ``QuadcopterSecondOrderRatesAction`` (+ its action-term config) removed from the shared
  ``mdp/actions.py``; it was unused outside the AI-GP experimental snapshot, which keeps its own
  copy.
* ``stats_origin_only`` / ``_spawned_at_origin`` spawn-statistics branches removed from
  ``mdp/events.py`` and ``mdp/gate_utils.py``.
* The "official route start" prefix removed from the shared ``mdp/route_progress.py`` (the
  AI-GP experimental snapshot keeps its own copy with this feature).
* ``gate_frame_fn`` hook removed from ``terrains/racing_gates_cfg.py`` and
  ``terrains/fixed_gates.py``.
* Two unused ``domain_randomization`` logging entries removed from the racing environment class.

0.1.0 (2026-04-05)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based racing quadcopter (standalone extension; no isaaclab_tasks dependency)."""

from __future__ import annotations

import os
from collections.abc import Sequence
import time

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors import ContactSensor

from Racing_Quadcopter.terrains import get_gate_registry
from Racing_Quadcopter.tasks.manager_based.action_diagnostics import ActionDiagnostics
from Racing_Quadcopter.tasks.manager_based.route_progress import (
    begin_racing_routes,
    initialize_route_progress,
)
from Racing_Quadcopter.tasks.manager_based.shared_features import (
    initialize_action_noise,
    initialize_speed_limits,
    validate_shared_feature_events,
)
from .mdp.curriculums import init_two_stage_curriculum
from .mdp.events import collect_episode_metrics_before_reset
from .mdp.gate_utils import update_racing_state
from .mdp.actions import build_quad_mixer_b_inv
from .mdp.stats import init_stats, step_stats
from .racing_quadcopter_env_cfg import RacingQuadcopterEnvCfg, RewardWeights
from .vis.trajectory_vis import (
    add_gate_sphere_lights_usd,
    apply_emissive_to_robots,
    draw_gate_indicators_usd,
    draw_gate_labels_usd,
    draw_gate_outlines_usd,
    draw_trajectory_usd,
    setup_dark_scene,
)


class RacingQuadcopterRLEnv(ManagerBasedRLEnv):
    """Gate racing environment with procedural gate terrain and manager-based RL."""

    cfg: RacingQuadcopterEnvCfg

    def __init__(self, cfg: RacingQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        # Type-hint only; both are populated during load_managers → _prepare_racing_task.
        self._robot: Articulation
        self._contact_sensor: ContactSensor
        self._mixer_b_inv: torch.Tensor
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self):
        # _prepare_racing_task must come BEFORE super().load_managers() because the
        # reward/obs/termination functions reference env buffers (e.g. _gate_positions,
        # _actions_buf) at manager construction time to infer output shapes.
        self._prepare_racing_task()
        super().load_managers()
        init_two_stage_curriculum(self)
        # filter_collisions is only supported on CPU; GPU broad-phase handles it automatically.
        if str(self.device) == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.terrain.prim_path])

    def _prepare_racing_task(self) -> None:
        """Allocate all racing-specific GPU buffers before managers are built."""
        validate_shared_feature_events(self.cfg)
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene.sensors["contact_sensor"]
        self._setup_gate_tracking()
        initialize_route_progress(self, self._gate_positions.shape[1] + 1)

        initialize_speed_limits(self)

        init_stats(self)
        self._action_config_log_interval_s = 120.0
        self._last_action_config_log_wall_time = 0.0

        # [N, 4] normalized thrust+moment actions. _prev_actions used for smoothness penalty.
        self._actions_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions_buf)
        # Raw policy output (before action manager processing) and one-step lag.
        self._policy_raw_actions = torch.zeros_like(self._actions_buf)
        self._prev_policy_raw_actions = torch.zeros_like(self._actions_buf)
        self._action_diagnostics = ActionDiagnostics(self)
        initialize_action_noise(self, action_dim=4)

        # Per-drone motor speeds [N, 4] (one element per impeller); updated elsewhere as needed.
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_motor_speeds = torch.zeros_like(self._motor_speeds)

        # X-frame mixer B^{-1} [4, 4] from cfg motor params (computed once per env instance).
        mixer_b_inv_np = build_quad_mixer_b_inv(
            self.cfg.motor_k_t,
            self.cfg.motor_k_d,
            self.cfg.motor_arm_length_m,
        )
        self._mixer_b_inv = torch.as_tensor(mixer_b_inv_np, device=self.device, dtype=torch.float32)

        # Float 0/1 per env: set by check_gate_passages each step.
        self._gate_passed_buf = torch.zeros(self.num_envs, device=self.device)

        # _crashed_buf: contact OR upside-down → drives crash penalty.
        # _collision_termination_buf: contact-force only, used for early termination.
        # They differ because upside-down fires the penalty but not the termination
        # until the curriculum threshold is met.
        self._crashed_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Set by check_gate_passages: drone crossed the current gate from the wrong direction.
        self._wrong_crossing_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Gate-frame basis vectors on device; reused in every quat_apply call.
        # Gate convention: +Y = forward (passage), +X = right, +Z = up.
        self._y_local = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self._x_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self._z_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._gate_step_indices = torch.arange(
            self.cfg.observation.num_next_gates,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(self.num_envs, -1)

        # ``camera_keypoint`` link (URDF) — preferred frame for perception reward; else base (body/base_link/``base``).
        try:
            _cam_ids, _ = self._robot.find_bodies("camera_keypoint")
        except ValueError:
            try:
                _cam_ids, _ = self._robot.find_bodies("body|base_link|^base$")
            except ValueError as err:
                raise RuntimeError(
                    "Perception reward needs a `camera_keypoint` body or a single base link matching "
                    "`body`, `base_link`, or `base`. "
                    f"Available bodies: {self._robot.body_names}"
                ) from err
        if isinstance(_cam_ids, torch.Tensor):
            _n = int(_cam_ids.numel())
        else:
            _n = len(_cam_ids)
        if _n != 1:
            raise RuntimeError(
                "Perception reward requires exactly one `camera_keypoint` body, or exactly one base link "
                f"(body/base_link/base). Matched {_n} bodies. Check the robot URDF/USD."
            )
        self._camera_keypoint_body_idx: int = (
            int(_cam_ids[0].item()) if isinstance(_cam_ids, torch.Tensor) else int(_cam_ids[0])
        )

        # All body IDs monitored for contact; ".*" captures every link.
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*")

        # Running episode stats (reset per env at episode end).
        self._episode_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_collision_count = torch.zeros(self.num_envs, device=self.device)

        # Lap completion stats: accumulated time and count for completed laps this episode.
        # _last_lap_step tracks the step count when the previous lap ended (or episode start),
        # so each lap's duration is delta steps × step_dt — not cumulative time from episode start.
        self._episode_lap_time_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_lap_count = torch.zeros(self.num_envs, device=self.device)
        self._last_lap_step = torch.zeros(self.num_envs, device=self.device)

        # Top-speed metric. _run_max_speed is each env's peak speed since spawn; it is
        # folded into the global running max _top_speed ONLY when that env completes a
        # lap (so speeds from runs that crash/time-out are never counted). Reset per env
        # at episode end; _top_speed persists for the whole run.
        self._run_max_speed = torch.zeros(self.num_envs, device=self.device)
        self._top_speed = torch.zeros((), device=self.device)

        # Per-passage quality stats accumulated each gate crossing.
        self._passage_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self._passage_centering_sum = torch.zeros(self.num_envs, device=self.device)
        self._passage_count = torch.zeros(self.num_envs, device=self.device)

        # Curriculum gate: collision termination enabled only once the agent is competent.
        # Starts False so early training always runs to timeout regardless of crashes.
        self._collision_detection_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


        # Two-stage curriculum stage index (0 = stay-alive, 1 = full racing objective).
        # In eval mode curriculum never advances; use a large sentinel (1000) so that
        # all stage gates are satisfied:  upside_down (== 0) stays OFF, collision
        # termination (>= 2) stays ON — matching fully-trained stage-3 behaviour.
        self._curriculum_stage: int = 1000 if getattr(self.cfg, "eval_mode", False) else 0

        # Snapshots of last episode metrics; used by curriculum check every step.
        # Kept separate from live accumulators because those are zeroed at episode start.
        self._last_episode_gates_passed = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_mean_velocity = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_center_offset = torch.zeros(self.num_envs, device=self.device)


    def _setup_gate_tracking(self) -> None:
        """Build world-frame gate position/orientation tables for all envs.

        Fetches terrain geometry, delegates data loading and env→cell mapping to
        ``_load_cell_gate_data`` (override in subclasses for different terrain types),
        then fills the shared gate buffers and calls the vis helpers.

        Centre-offset correction:
            Gate positions are generated in a corner-origin local frame (origin at mesh
            corner). TerrainGenerator places each cell so its CENTER sits at terrain_origin,
            so we subtract (size_x/2, size_y/2, 0) before adding the origin.

        Padding:
            Different cells may have different gate counts. All tensors are padded to
            max_gates; _gate_counts[env_id] stores the true count so modular indexing
            never reads a padding slot.
        """
        terrain = self.scene.terrain
        if terrain is None or terrain.cfg.terrain_generator is None:
            raise RuntimeError("Racing quadcopter requires a generator terrain with gates.")
        terrain_cfg = terrain.cfg.terrain_generator

        # Shift from corner-origin (mesh builder) to centre-origin (TerrainGenerator).
        centre_offset = np.array([-terrain_cfg.size[0] / 2.0, -terrain_cfg.size[1] / 2.0, 0.0])

        terrain_origins = terrain.terrain_origins
        if terrain_origins is None:
            raise RuntimeError("Terrain origins are None — set use_terrain_origins on TerrainImporterCfg.")
        if isinstance(terrain_origins, np.ndarray):
            terrain_origins = torch.from_numpy(terrain_origins).float().to(self.device)
        else:
            terrain_origins = terrain_origins.to(self.device)
        num_rows, num_cols = terrain_origins.shape[:2]

        per_cell_pos, per_cell_ori, per_cell_sizes, gate_datas, env_cell_idx, env_origins = \
            self._load_cell_gate_data(terrain_origins, centre_offset)

        max_gates = max(p.shape[0] for p in per_cell_pos)
        self._num_gates = max_gates
        self._gate_sizes = torch.zeros(self.num_envs, device=self.device)

        # Shape [N, max_gates, 3/4]; padded slots are zero.
        self._gate_positions = torch.zeros(self.num_envs, max_gates, 3, device=self.device)
        self._gate_orientations = torch.zeros(self.num_envs, max_gates, 4, device=self.device)
        self._gate_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for env_id in range(self.num_envs):
            c = int(env_cell_idx[env_id])
            origin = env_origins[env_id]
            n = per_cell_pos[c].shape[0]
            self._gate_positions[env_id, :n] = per_cell_pos[c] + origin
            self._gate_orientations[env_id, :n] = per_cell_ori[c].clone()
            self._gate_counts[env_id] = int(n)
            self._gate_sizes[env_id] = per_cell_sizes[c]
            if n < max_gates:
                self._gate_positions[env_id, n:] = 0.0
                self._gate_orientations[env_id, n:] = 0.0

        if os.getenv("DEBUG_TERRAIN") == "1":
            try:
                print("DEBUG_TERRAIN: gate_positions_env0_sample=", self._gate_positions[0, :3].cpu().numpy())
                if self.num_envs > 1:
                    print("DEBUG_TERRAIN: gate_positions_env1_sample=", self._gate_positions[1, :3].cpu().numpy())
                centroids = self._gate_positions.mean(dim=1).cpu().numpy()
                uniq_cent = np.unique(np.round(centroids, 4), axis=0)
                print("DEBUG_TERRAIN: unique_env_centroids=", uniq_cent.shape[0], "of", self.num_envs)
                try:
                    gates_nonzero = (self._gate_positions.abs().sum(dim=2) > 0.0).sum(dim=1).cpu().numpy()
                    print("DEBUG_TERRAIN: nonzero_gate_counts_sample=", gates_nonzero[:5])
                    print("DEBUG_TERRAIN: first_gate_positions_sample=", self._gate_positions[:5, 0].cpu().numpy())
                except Exception:
                    pass
            except Exception:
                pass

        # Wraps modulo _gate_counts[env_id] on each passage; never reads padding slots.
        self._current_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Gate index the drone spawned in front of this episode; needed to detect lap completion
        # when spawning mid-circuit (a lap = passing _gate_counts gates from any start gate).
        self._spawn_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Cumulative passages this episode; used for lap_complete and logging.
        self._gates_passed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Previous-step drone position in world frame; required for line-crossing detection
        # and for the gate-progress reward (distance reduction over one step).
        self._prev_drone_pos = torch.zeros(self.num_envs, 3, device=self.device)

        origins_np = terrain_origins.cpu().numpy()
        emissive = self.cfg.demo_emissive_intensity if self.cfg.demo_mode else 0.0
        if self.cfg.draw_trajectory:
            draw_trajectory_usd(
                gate_datas=gate_datas,
                terrain_origins_np=origins_np,
                centre_offset=centre_offset,
                num_rows=num_rows,
                num_cols=num_cols,
                emissive_intensity=emissive,
            )
        if self.cfg.draw_gate_indicators:
            draw_gate_indicators_usd(
                gate_datas=gate_datas,
                terrain_origins_np=origins_np,
                centre_offset=centre_offset,
                num_rows=num_rows,
                num_cols=num_cols,
                emissive_intensity=emissive,
            )
        if self.cfg.draw_gate_labels:
            draw_gate_labels_usd(
                gate_datas=gate_datas,
                terrain_origins_np=origins_np,
                centre_offset=centre_offset,
                num_rows=num_rows,
                num_cols=num_cols,
                emissive_intensity=emissive,
            )
        if self.cfg.demo_mode:
            setup_dark_scene()
            draw_gate_outlines_usd(
                gate_datas=gate_datas,
                terrain_origins_np=origins_np,
                centre_offset=centre_offset,
                num_rows=num_rows,
                num_cols=num_cols,
                color=self.cfg.demo_gate_color,
                emissive_intensity=self.cfg.demo_gate_outline_emissive,
                width=self.cfg.demo_gate_outline_width,
            )
            add_gate_sphere_lights_usd(
                gate_datas=gate_datas,
                terrain_origins_np=origins_np,
                centre_offset=centre_offset,
                num_rows=num_rows,
                num_cols=num_cols,
                color=self.cfg.demo_gate_color,
                intensity=self.cfg.demo_gate_light_intensity,
                radius=self.cfg.demo_gate_light_radius,
            )
            apply_emissive_to_robots(
                color=self.cfg.demo_drone_glow_color,
                intensity=self.cfg.demo_drone_glow_intensity,
            )

        self._log_action_configuration(reason="startup")
        self._last_action_config_log_wall_time = time.time()

    def _load_cell_gate_data(
        self,
        terrain_origins: torch.Tensor,
        centre_offset: np.ndarray,
    ) -> tuple[list, list, list, list, np.ndarray, torch.Tensor]:
        """Load per-cell gate data and compute the env→cell mapping.

        Default implementation for procedural / single-circuit terrains: reads
        from the module-level ``gate_registry`` (populated as a side-effect of
        terrain generation) and resolves env→cell via the hand-rolled row-major
        formula ``cell_idx = (row * num_cols + col) % n_cells``.

        Override in subclasses that source gate data differently (e.g.
        ``RacingQuadcopterMultiTrackRLEnv`` reads from sub_terrain cfgs and
        uses ``scene.env_origins`` for an exact env→cell lookup).

        Args:
            terrain_origins: ``[num_rows, num_cols, 3]`` world-frame cell centres.
            centre_offset:   ``[3]`` shift from corner-origin to centre-origin frame.

        Returns:
            per_cell_pos   : ``list[Tensor[G, 3]]`` gate positions, centre-origin frame.
            per_cell_ori   : ``list[Tensor[G, 4]]`` gate quaternions (wxyz).
            per_cell_sizes : ``list[float]`` gate aperture per cell.
            gate_datas     : ``list[GateData]`` in cell-major order for vis helpers.
            env_cell_idx   : ``np.ndarray[num_envs]`` per-env index into per_cell_*.
            env_origins    : ``Tensor[num_envs, 3]`` per-env world-frame terrain origin.
        """
        registry = get_gate_registry()
        if not registry:
            raise RuntimeError(
                "Gate registry is empty — terrain must call racing_gates_terrain / fixed_gates_terrain."
            )

        gate_datas_reg = list(registry.values())
        per_cell_pos: list[torch.Tensor] = []
        per_cell_ori: list[torch.Tensor] = []
        per_cell_sizes: list[float] = []
        for gd in gate_datas_reg:
            local_pos = gd.positions + centre_offset
            per_cell_pos.append(torch.from_numpy(local_pos).float().to(self.device))
            per_cell_ori.append(torch.from_numpy(gd.orientations).float().to(self.device))
            per_cell_sizes.append(float(gd.gate_size))

        num_rows, num_cols = terrain_origins.shape[:2]
        n_cells = len(per_cell_pos)

        if os.getenv("DEBUG_TERRAIN") == "1":
            try:
                keys = list(registry.keys())
                print("DEBUG_TERRAIN: gate_registry_len=", len(keys))
                print("DEBUG_TERRAIN: gate_registry_keys_sample=", keys[:5])
            except Exception:
                pass
            try:
                to_cpu = terrain_origins.cpu().numpy()
                print("DEBUG_TERRAIN: terrain_origins_shape=", to_cpu.shape)
                uniq = np.unique(np.round(to_cpu.reshape(-1, 3), 5), axis=0)
                print("DEBUG_TERRAIN: unique_origins_count=", uniq.shape[0])
                print("DEBUG_TERRAIN: terrain_origins_sample=", to_cpu.reshape(-1, 3)[:5])
            except Exception:
                pass

        env_cell_idx = np.empty(self.num_envs, dtype=int)
        env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        for env_id in range(self.num_envs):
            row = (env_id // num_cols) % num_rows
            col = env_id % num_cols
            cell_idx = (row * num_cols + col) % n_cells
            env_cell_idx[env_id] = cell_idx
            env_origins[env_id] = terrain_origins[row, col]

        # gate_datas in cell-major order (row * num_cols + col) for vis helpers.
        gate_datas = [gate_datas_reg[(r * num_cols + c) % n_cells]
                      for r in range(num_rows) for c in range(num_cols)]

        return per_cell_pos, per_cell_ori, per_cell_sizes, gate_datas, env_cell_idx, env_origins

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Keep a copy of raw network actions before the action manager transforms/clips them.
        self._prev_policy_raw_actions[:] = self._policy_raw_actions
        self._policy_raw_actions[:] = action.to(self.device)

        # Step 1: clamp to [-1,1], write _actions_buf, copy to _prev_actions.
        self.action_manager.process_action(self._policy_raw_actions)

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Steps 2a-e: decimation loop — apply forces, advance physics, refresh sensors.
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # Step 3: gate passage detection, crash buffer refresh, episode velocity stats.
        # Must run after the final scene.update() so all sensor data is current.
        update_racing_state(self)

        step_stats(self)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if os.getenv("DEBUG_REWARDS") == "1":
            with torch.no_grad():
                cfg : RacingQuadcopterEnvCfg = self.cfg # for type hints
                per_term_weighted : dict[str, torch.Tensor] = {}
                for reward_term_cfg_name in cfg.rewards.__annotations__:
                    reward_term_cfg = getattr(cfg.rewards, reward_term_cfg_name, None)
                    fn = reward_term_cfg.func
                    w = reward_term_cfg.weight
                    per_term_weighted[reward_term_cfg_name] = fn(self) * w
                stacked = torch.stack(list(per_term_weighted.values()), dim=0)
                summed = stacked.sum(dim=0)
                print("DEBUG_REWARDS: dt=", float(self.step_dt), "reward_buf_sample=", self.reward_buf[:5].cpu().numpy())
                print("DEBUG_REWARDS: summed_unscaled_sample=", summed[:5].cpu().numpy())
                for k, v in per_term_weighted.items():
                    try:
                        w = getattr(cfg.rewards, k).weight
                        print(f"DEBUG_REWARDS: term={k} mean={v.mean().item():.6e} weight={w:.6e} weighted_mean={(v*float(w)).mean().item():.6e}")
                    except Exception:
                        pass

        # Update _prev_drone_pos AFTER rewards: reward_gate_progress reads it as the
        # "old" position to compute distance reduction over this step. Updating it first
        # would make prev_dist == curr_dist → zero progress signal every step.
        self._prev_drone_pos[:] = self._robot.data.root_pos_w

        # reset_terminated covers all non-timeout terminations (collision, upside-down,
        # runaway velocity) — not just contact collisions.
        self._episode_collision_count += self.reset_terminated.float()

        # First obs pass: only for the recorder (needs pre-reset state). Does NOT
        # update history to avoid double-advancing the history buffer.
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self._action_diagnostics.update_if_due()
        self._maybe_log_action_configuration()

        # Second obs pass: the "real" obs returned to the policy, with history update.
        # Runs after resets so freshly-spawned envs contribute correct observations.
        self.obs_buf = self.observation_manager.compute(update_history=True)

        self._prev_motor_speeds[:] = self._motor_speeds

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int] | None):
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)

        # Must run BEFORE super()._reset_idx: the super call triggers the EventManager
        # reset mode which zeros _gates_passed, _current_gate_idx, etc. via
        # reset_racing_spawn. Calling collect_episode_metrics_before_reset after that
        # would always log zero gates passed.
        gate_log = collect_episode_metrics_before_reset(self, env_ids_tensor)

        super()._reset_idx(env_ids)
        begin_racing_routes(self, env_ids_tensor)

        if os.getenv("DEBUG_TERRAIN") == "1":
            try:
                ids_cpu = env_ids_tensor.cpu().numpy()
                cur_idx = self._current_gate_idx[env_ids_tensor].cpu().numpy()
                counts = self._gate_counts[env_ids_tensor].cpu().numpy()
                first_pos = self._gate_positions[env_ids_tensor, 0].cpu().numpy()
                print("DEBUG_TERRAIN: reset_env_ids=", ids_cpu[:10])
                print("DEBUG_TERRAIN: current_gate_idx_after_reset_sample=", cur_idx[:10])
                print("DEBUG_TERRAIN: gate_counts_after_reset_sample=", counts[:10])
                print("DEBUG_TERRAIN: first_gate_pos_after_reset_sample=", first_pos[:5])
            except Exception as e:
                print("DEBUG_TERRAIN: reset debug error", e)

        if gate_log:
            self.extras["log"].update(gate_log)
            self._on_episode_metrics_collected(gate_log)

        # SKRL only logs scalar tensors from extras["log"]; managers emit floats.
        if self.extras.get("log"):
            log_tensors: dict[str, torch.Tensor] = {}
            for key, value in self.extras["log"].items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    log_tensors[key] = value.detach().reshape(()).to(device=self.device, dtype=torch.float32)
                elif isinstance(value, (float, int, bool)):
                    log_tensors[key] = torch.tensor(float(value), device=self.device, dtype=torch.float32)
            self.extras["log"] = log_tensors

        # Keep raw-action lag buffer aligned with reset state to avoid artificial first-step spikes.
        self._policy_raw_actions[env_ids_tensor].zero_()
        self._prev_policy_raw_actions[env_ids_tensor].zero_()
        self._action_diagnostics.reset(env_ids_tensor)

    def _on_episode_metrics_collected(self, gate_log: dict) -> None:
        """Hook called after each reset once `gate_log` has been merged into ``extras["log"]``.

        Default is a no-op. Subclasses (e.g. ``RacingQuadcopterSkillEvalRLEnv``)
        override this to write the metrics directly to a tensorboard writer
        without relying on the agent runner having a configured ``log_dir``.
        """
        return

    @staticmethod
    def _format_config_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (tuple, list)):
            return "(" + ", ".join(RacingQuadcopterRLEnv._format_config_value(v) for v in value) + ")"
        return str(value)

    def _log_action_configuration(self, reason: str) -> None:
        """Print and log the current action-noise / corruption / randomization config."""
        action_noise_cfg = getattr(self.cfg, "action_noise", None)
        observations_cfg = getattr(self.cfg, "observations", None)
        policy_obs_cfg = getattr(observations_cfg, "policy", None)
        actions_cfg = getattr(self.cfg, "actions", None)
        rates_cfg = getattr(actions_cfg, "rates", None)
        if hasattr(self, "action_manager"):
            try:
                rates_cfg = self.action_manager.get_term("rates").cfg
            except Exception:
                pass
        moments_cfg = getattr(actions_cfg, "moments", None)
        action_cfg = rates_cfg if rates_cfg is not None else moments_cfg
        action_type = "rates" if rates_cfg is not None else ("moments" if moments_cfg is not None else "none")

        policy_term_cfgs = {}
        if hasattr(self, "observation_manager"):
            term_names = self.observation_manager._group_obs_term_names.get("policy", [])
            term_cfgs = self.observation_manager._group_obs_term_cfgs.get("policy", [])
            policy_term_cfgs = {name: cfg for name, cfg in zip(term_names, term_cfgs)}

        def _obs_noise_attr(term_name: str, attr: str, default: object = "n/a") -> object:
            term_cfg = policy_term_cfgs.get(term_name, None)
            if term_cfg is None:
                return default
            noise_cfg = getattr(term_cfg, "noise", None)
            if noise_cfg is None:
                return default
            obj = noise_cfg
            for name in attr.split("."):
                obj = getattr(obj, name, None)
                if obj is None:
                    return default
            return obj

        fields: dict[str, object] = {
            "domain_randomization": getattr(self.cfg, "domain_randomization", "n/a"),
            "observation_noise_enable": getattr(self.cfg, "observation_noise_enable", "n/a"),
            "observations.policy.enable_corruption": getattr(policy_obs_cfg, "enable_corruption", "n/a"),
            "obs_cfg.policy.base_lin_vel.noise.enable": _obs_noise_attr("base_lin_vel", "enable"),
            "obs_cfg.policy.base_lin_vel.noise.std": _obs_noise_attr("base_lin_vel", "noise_cfg.std"),
            "obs_cfg.policy.base_lin_vel.bias.std": _obs_noise_attr("base_lin_vel", "bias_noise_cfg.std"),
            "obs_cfg.policy.base_ang_vel.noise.enable": _obs_noise_attr("base_ang_vel", "enable"),
            "obs_cfg.policy.base_ang_vel.noise.std": _obs_noise_attr("base_ang_vel", "noise_cfg.std"),
            "obs_cfg.policy.base_ang_vel.bias.std": _obs_noise_attr("base_ang_vel", "bias_noise_cfg.std"),
            "obs_cfg.policy.projected_gravity.noise.enable": _obs_noise_attr("projected_gravity", "enable"),
            "obs_cfg.policy.projected_gravity.noise.std": _obs_noise_attr("projected_gravity", "noise_cfg.std"),
            "obs_cfg.policy.projected_gravity.bias.std": _obs_noise_attr("projected_gravity", "bias_noise_cfg.std"),
            "action_noise.enable": getattr(action_noise_cfg, "enable", "n/a"),
            "action_noise.per_episode_bias": getattr(action_noise_cfg, "per_episode_bias", "n/a"),
            "action_noise.gaussian_std": getattr(action_noise_cfg, "gaussian_std", "n/a"),
            "action_noise.bias_range": getattr(action_noise_cfg, "bias_range", "n/a"),
            "action_noise.lpf_alpha": getattr(action_noise_cfg, "lpf_alpha", "n/a"),
            "action_noise.per_episode_lpf_alpha": getattr(action_noise_cfg, "per_episode_lpf_alpha", "n/a"),
            "action_noise.lpf_alpha_range": getattr(action_noise_cfg, "lpf_alpha_range", "n/a"),
            "action_noise.clip_actions": getattr(action_noise_cfg, "clip_actions", "n/a"),
            "action_type": action_type,
        }
        if action_cfg is not None:
            fields["action_cfg.max_delay"] = getattr(action_cfg, "max_delay", "n/a")

        if rates_cfg is not None:
            fields.update(
                {
                    "action_cfg.pid_gain_randomization_frac": getattr(rates_cfg, "pid_gain_randomization_frac", "n/a"),
                    "action_cfg.thrust_to_weight_randomization_frac": getattr(
                        rates_cfg, "thrust_to_weight_randomization_frac", "n/a"
                    ),
                    "action_cfg.rate_filter_natural_frequency_randomization_frac": getattr(
                        rates_cfg, "rate_filter_natural_frequency_randomization_frac", "n/a"
                    ),
                    "action_cfg.rate_filter_damping_randomization_frac": getattr(
                        rates_cfg, "rate_filter_damping_randomization_frac", "n/a"
                    ),
                    "action_cfg.rate_filter_command_gain_randomization_frac": getattr(
                        rates_cfg, "rate_filter_command_gain_randomization_frac", "n/a"
                    ),
                    "action_cfg.aero_randomization_enable": getattr(rates_cfg, "aero_randomization_enable", "n/a"),
                    "action_cfg.aero_randomization_frac": getattr(rates_cfg, "aero_randomization_frac", "n/a"),
                }
            )

        summary = " ".join(f"{key}={self._format_config_value(value)}" for key, value in fields.items())
        print(f"[ACTION_CFG][{reason}] {summary}")

        log_dict = self.extras.setdefault("log", {})
        scalar_fields = {
            "Config/domain_randomization": getattr(self.cfg, "domain_randomization", None),
            "Config/observation_noise_enable": getattr(self.cfg, "observation_noise_enable", None),
            "Config/observations_policy_enable_corruption": getattr(policy_obs_cfg, "enable_corruption", None),
            "Config/obs_cfg_policy_base_lin_vel_noise_enable": _obs_noise_attr("base_lin_vel", "enable", None),
            "Config/obs_cfg_policy_base_lin_vel_noise_std": _obs_noise_attr("base_lin_vel", "noise_cfg.std", None),
            "Config/obs_cfg_policy_base_lin_vel_bias_std": _obs_noise_attr("base_lin_vel", "bias_noise_cfg.std", None),
            "Config/obs_cfg_policy_base_ang_vel_noise_enable": _obs_noise_attr("base_ang_vel", "enable", None),
            "Config/obs_cfg_policy_base_ang_vel_noise_std": _obs_noise_attr("base_ang_vel", "noise_cfg.std", None),
            "Config/obs_cfg_policy_base_ang_vel_bias_std": _obs_noise_attr("base_ang_vel", "bias_noise_cfg.std", None),
            "Config/obs_cfg_policy_projected_gravity_noise_enable": _obs_noise_attr("projected_gravity", "enable", None),
            "Config/obs_cfg_policy_projected_gravity_noise_std": _obs_noise_attr("projected_gravity", "noise_cfg.std", None),
            "Config/obs_cfg_policy_projected_gravity_bias_std": _obs_noise_attr("projected_gravity", "bias_noise_cfg.std", None),
            "Config/action_noise_enable": getattr(action_noise_cfg, "enable", None),
            "Config/action_noise_per_episode_bias": getattr(action_noise_cfg, "per_episode_bias", None),
            "Config/action_noise_gaussian_std": getattr(action_noise_cfg, "gaussian_std", None),
            "Config/action_noise_lpf_alpha": getattr(action_noise_cfg, "lpf_alpha", None),
            "Config/action_noise_per_episode_lpf_alpha": getattr(action_noise_cfg, "per_episode_lpf_alpha", None),
            "Config/action_noise_clip_actions": getattr(action_noise_cfg, "clip_actions", None),
            "Config/action_cfg_max_delay": getattr(action_cfg, "max_delay", None) if action_cfg is not None else None,
            "Config/action_type_is_rates": 1.0 if rates_cfg is not None else 0.0,
            "Config/action_type_is_moments": 1.0 if moments_cfg is not None else 0.0,
        }
        if rates_cfg is not None:
            scalar_fields.update(
                {
                    "Config/action_cfg_pid_gain_randomization_frac": getattr(
                        rates_cfg, "pid_gain_randomization_frac", None
                    ),
                    "Config/action_cfg_thrust_to_weight_randomization_frac": getattr(
                        rates_cfg, "thrust_to_weight_randomization_frac", None
                    ),
                    "Config/action_cfg_rate_filter_natural_frequency_randomization_frac": getattr(
                        rates_cfg, "rate_filter_natural_frequency_randomization_frac", None
                    ),
                    "Config/action_cfg_rate_filter_damping_randomization_frac": getattr(
                        rates_cfg, "rate_filter_damping_randomization_frac", None
                    ),
                    "Config/action_cfg_rate_filter_command_gain_randomization_frac": getattr(
                        rates_cfg, "rate_filter_command_gain_randomization_frac", None
                    ),
                    "Config/action_cfg_aero_randomization_enable": getattr(rates_cfg, "aero_randomization_enable", None),
                }
            )

        for key, value in scalar_fields.items():
            if value is None:
                continue
            if isinstance(value, bool):
                log_dict[key] = torch.tensor(float(value), device=self.device, dtype=torch.float32)
            elif isinstance(value, (int, float)):
                log_dict[key] = torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def _maybe_log_action_configuration(self) -> None:
        now = time.time()
        if now - self._last_action_config_log_wall_time < self._action_config_log_interval_s:
            return
        self._last_action_config_log_wall_time = now
        self._log_action_configuration(reason="heartbeat")

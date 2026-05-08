# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based racing quadcopter (standalone extension; no isaaclab_tasks dependency)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors import ContactSensor

from Racing_Quadcopter.terrains import get_gate_registry
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
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene.sensors["contact_sensor"]
        self._setup_gate_tracking()

        # Per-env speed cap; re-sampled each episode if cfg.randomize_velocity_limit.
        self._velocity_limit = torch.full((self.num_envs,), self.cfg.velocity_limit, device=self.device)

        init_stats(self)

        # [N, 4] normalized thrust+moment actions. _prev_actions used for smoothness penalty.
        self._actions_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions_buf)
        # Action-noise per-env bias (sampled at reset if enabled)
        self._action_noise_bias = torch.zeros(self.num_envs, 4, device=self.device)

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
        # Step 1: clamp to [-1,1], write _actions_buf, copy to _prev_actions.
        self.action_manager.process_action(action.to(self.device))

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

        # Sample per-episode biases for action noise (if enabled).
        # These buffers were pre-allocated in _prepare_racing_task.
        n = env_ids_tensor.shape[0]

        action_noise_cfg = getattr(self.cfg, "action_noise", None)
        if action_noise_cfg and getattr(action_noise_cfg, "enable", False) and getattr(action_noise_cfg, "per_episode_bias", False):
            a, b = action_noise_cfg.bias_range
            self._action_noise_bias[env_ids_tensor] = torch.rand(n, 4, device=self.device) * (b - a) + a
        else:
            self._action_noise_bias[env_ids_tensor] = 0.0

        if gate_log:
            self.extras["log"].update(gate_log)

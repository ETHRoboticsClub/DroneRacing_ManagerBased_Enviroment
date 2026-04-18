# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based racing quadcopter (standalone extension; no isaaclab_tasks dependency)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors import ContactSensor
from Racing_Quadcopter.terrains.racing_gates import get_gate_registry
from .mdp.events import collect_episode_metrics_before_reset
from .mdp.rewards import update_racing_state
from .mdp import rewards as racing_rewards
import os
from .racing_quadcopter_env_cfg import RacingQuadcopterEnvCfg


class RacingQuadcopterRLEnv(ManagerBasedRLEnv):
    """Gate racing with terrain meshes."""

    cfg: RacingQuadcopterEnvCfg

    def __init__(self, cfg: RacingQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        self._robot: Articulation
        self._contact_sensor: ContactSensor
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self):
        self._prepare_racing_task()
        super().load_managers()
        if str(self.device) == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.terrain.prim_path])

    def _prepare_racing_task(self) -> None:
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene.sensors["contact_sensor"]
        self._setup_gate_tracking()

        self._velocity_limit = torch.full((self.num_envs,), self.cfg.velocity_limit, device=self.device)
        self._actions_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions_buf)
        self._gate_passed_buf = torch.zeros(self.num_envs, device=self.device)
        self._crashed_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._y_local = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self._x_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self._z_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*")

        self._episode_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_collision_count = torch.zeros(self.num_envs, device=self.device)
        self._passage_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self._passage_centering_sum = torch.zeros(self.num_envs, device=self.device)
        self._passage_count = torch.zeros(self.num_envs, device=self.device)

        self._collision_detection_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._last_episode_gates_passed = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_mean_velocity = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_center_offset = torch.zeros(self.num_envs, device=self.device)

    def _setup_gate_tracking(self) -> None:
        registry = get_gate_registry()
        if not registry:
            raise RuntimeError(
                "Gate registry is empty — use terrain that calls racing_gates_terrain (MeshRacingGatesTerrainCfg)."
            )

        # Registry may contain one entry per generated sub-terrain cell.
        gate_datas = list(registry.values())
        terrain = self.scene.terrain
        if terrain is None or terrain.cfg.terrain_generator is None:
            raise RuntimeError("Racing quadcopter requires a generator terrain with gates.")
        terrain_cfg = terrain.cfg.terrain_generator
        centre_offset = np.array([-terrain_cfg.size[0] / 2.0, -terrain_cfg.size[1] / 2.0, 0.0])
        # Build per-cell base gate positions/orientations if registry provides multiple entries
        per_cell_base_pos = []
        per_cell_base_ori = []
        gate_size = None
        for gd in gate_datas:
            local_gate_pos = gd.positions + centre_offset
            per_cell_base_pos.append(torch.from_numpy(local_gate_pos).float().to(self.device))
            per_cell_base_ori.append(torch.from_numpy(gd.orientations).float().to(self.device))
            gate_size = gd.gate_size if gate_size is None else gate_size

        if len(per_cell_base_pos) == 0:
            raise RuntimeError("Gate registry is empty — use terrain that calls racing_gates_terrain (MeshRacingGatesTerrainCfg).")

        # Cells may have differing numbers of gates. Use the maximum across cells
        # and pad shorter cells with zeros so tensors have fixed shape.
        per_cell_counts = [p.shape[0] for p in per_cell_base_pos]
        max_gates = max(per_cell_counts)
        self._num_gates = max_gates
        self._gate_size = gate_size

        terrain_origins = terrain.terrain_origins
        if terrain_origins is None:
            raise RuntimeError("Terrain origins are None — set use_terrain_origins on TerrainImporterCfg.")
        if isinstance(terrain_origins, np.ndarray):
            terrain_origins = torch.from_numpy(terrain_origins).float().to(self.device)
        else:
            terrain_origins = terrain_origins.to(self.device)
        num_rows, num_cols = terrain_origins.shape[:2]

        # Optional debug: check whether terrains/gates are per-subterrain (per-cell)
        if os.getenv("DEBUG_TERRAIN") == "1":
            try:
                keys = list(registry.keys())
                print("DEBUG_TERRAIN: gate_registry_len=", len(keys))
                print("DEBUG_TERRAIN: gate_registry_keys_sample=", keys[:5])
            except Exception:
                pass
            try:
                # show terrain_origins stats
                to_cpu = terrain_origins.cpu().numpy()
                print("DEBUG_TERRAIN: terrain_origins_shape=", to_cpu.shape)
                # show unique origins count (rounded)
                uniq = np.unique(np.round(to_cpu.reshape(-1, 3), 5), axis=0)
                print("DEBUG_TERRAIN: unique_origins_count=", uniq.shape[0])
                print("DEBUG_TERRAIN: terrain_origins_sample=", to_cpu.reshape(-1, 3)[:5])
            except Exception:
                pass

        self._gate_positions = torch.zeros(self.num_envs, max_gates, 3, device=self.device)
        self._gate_orientations = torch.zeros(self.num_envs, max_gates, 4, device=self.device)
        # Per-environment gate counts (actual number of gates before padding)
        self._gate_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Map registry entries to terrain grid cells. We assume insertion order of the
        # registry matches the TerrainGenerator's generation order (row-major).
        num_cells = num_rows * num_cols
        for env_id in range(self.num_envs):
            row = (env_id // num_cols) % num_rows
            col = env_id % num_cols
            origin = terrain_origins[row, col]

            cell_idx = row * num_cols + col
            # if registry has fewer/more entries than cells, wrap or clamp
            cell_idx = cell_idx % len(per_cell_base_pos)
            cell_base_pos = per_cell_base_pos[cell_idx]
            cell_base_ori = per_cell_base_ori[cell_idx]

            n = cell_base_pos.shape[0]
            # assign existing gates
            self._gate_positions[env_id, :n] = cell_base_pos + origin
            self._gate_orientations[env_id, :n] = cell_base_ori.clone()
            # record how many gates this env actually has (for safe wrapping)
            self._gate_counts[env_id] = int(n)
            # pad remaining slots with zeros for positions and orientations
            if n < max_gates:
                self._gate_positions[env_id, n:] = 0.0
                self._gate_orientations[env_id, n:] = 0.0

        if os.getenv("DEBUG_TERRAIN") == "1":
            try:
                # print first few env gate centers to verify per-env differences
                print("DEBUG_TERRAIN: gate_positions_env0_sample=", self._gate_positions[0, :3].cpu().numpy())
                if self.num_envs > 1:
                    print("DEBUG_TERRAIN: gate_positions_env1_sample=", self._gate_positions[1, :3].cpu().numpy())
                # count distinct per-env gate centroid positions (rough check)
                centroids = self._gate_positions.mean(dim=1).cpu().numpy()
                uniq_cent = np.unique(np.round(centroids, 4), axis=0)
                print("DEBUG_TERRAIN: unique_env_centroids=", uniq_cent.shape[0], "of", self.num_envs)
                # per-env non-zero gate counts and first-gate positions
                try:
                    gates_nonzero = (self._gate_positions.abs().sum(dim=2) > 0.0).sum(dim=1).cpu().numpy()
                    print("DEBUG_TERRAIN: nonzero_gate_counts_sample=", gates_nonzero[:5])
                    print("DEBUG_TERRAIN: first_gate_positions_sample=", self._gate_positions[:5, 0].cpu().numpy())
                except Exception:
                    pass
            except Exception:
                pass

        self._current_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gates_passed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_drone_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        update_racing_state(self)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Optional debug: print per-term unweighted factors and weighted sums
        if os.getenv("DEBUG_REWARDS") == "1":
            # compute unweighted factors directly from mdp.rewards functions
            with torch.no_grad():
                terms = {
                    "gate_passage": racing_rewards.reward_gate_passage(self),
                    "gate_progress": racing_rewards.reward_gate_progress(self),
                    "gate_centering": racing_rewards.reward_gate_centering(self),
                    "velocity_toward_gate": racing_rewards.reward_velocity_toward_gate(self),
                    "velocity_limit": racing_rewards.penalty_velocity_limit(self),
                    "orientation": racing_rewards.penalty_orientation(self),
                    "angular_velocity": racing_rewards.penalty_angular_velocity(self),
                    "action_smoothness": racing_rewards.penalty_action_smoothness(self),
                    "action_magnitude": racing_rewards.penalty_action_magnitude(self),
                    "crash": racing_rewards.penalty_crash(self),
                    "wrong_side": racing_rewards.penalty_wrong_side(self),
                }

                weights = {
                    "gate_passage": racing_rewards.WEIGHT_GATE_PASSAGE,
                    "gate_progress": racing_rewards.WEIGHT_GATE_PROGRESS,
                    "gate_centering": racing_rewards.WEIGHT_GATE_CENTERING,
                    "velocity_toward_gate": racing_rewards.WEIGHT_VELOCITY_TOWARD_GATE,
                    "velocity_limit": racing_rewards.WEIGHT_VELOCITY_LIMIT,
                    "orientation": racing_rewards.WEIGHT_ORIENTATION,
                    "angular_velocity": racing_rewards.WEIGHT_ANGULAR_VELOCITY,
                    "action_smoothness": racing_rewards.WEIGHT_ACTION_SMOOTHNESS,
                    "action_magnitude": racing_rewards.WEIGHT_ACTION_MAGNITUDE,
                    "crash": racing_rewards.WEIGHT_CRASH,
                    "wrong_side": racing_rewards.WEIGHT_WRONG_SIDE,
                }

                per_term_weighted = {}
                for k, v in terms.items():
                    w = weights.get(k, 1.0)
                    per_term_weighted[k] = (v * float(w))

                # Sum weighted terms per-env
                stacked = torch.stack(list(per_term_weighted.values()), dim=0)
                summed = stacked.sum(dim=0)

                print("DEBUG_REWARDS: dt=", float(self.step_dt), "reward_buf_sample=", self.reward_buf[:5].cpu().numpy())
                print("DEBUG_REWARDS: summed_unscaled_sample=", summed[:5].cpu().numpy())
                # print per-term means for quick inspection
                for k, v in terms.items():
                    try:
                        print(f"DEBUG_REWARDS: term={k} mean={v.mean().item():.6e} weight={weights[k]:.6e} weighted_mean={(v*float(weights[k])).mean().item():.6e}")
                    except Exception:
                        pass

        self._prev_drone_pos[:] = self._robot.data.root_pos_w

        self._episode_collision_count += self.reset_terminated.float()

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

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int] | None):
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)

        gate_log = collect_episode_metrics_before_reset(self, env_ids_tensor)

        super()._reset_idx(env_ids)

        # DEBUG: after reset, log current gate idx and counts for these envs
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

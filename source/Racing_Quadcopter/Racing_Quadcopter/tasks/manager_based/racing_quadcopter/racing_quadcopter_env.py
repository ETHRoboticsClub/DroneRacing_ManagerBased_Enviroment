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

        gate_data = next(iter(registry.values()))
        terrain = self.scene.terrain
        if terrain is None or terrain.cfg.terrain_generator is None:
            raise RuntimeError("Racing quadcopter requires a generator terrain with gates.")
        terrain_cfg = terrain.cfg.terrain_generator
        centre_offset = np.array([-terrain_cfg.size[0] / 2.0, -terrain_cfg.size[1] / 2.0, 0.0])
        local_gate_pos = gate_data.positions + centre_offset

        base_gate_pos = torch.from_numpy(local_gate_pos).float().to(self.device)
        base_gate_ori = torch.from_numpy(gate_data.orientations).float().to(self.device)
        self._gate_size = gate_data.gate_size

        num_gates = base_gate_pos.shape[0]
        self._num_gates = num_gates

        terrain_origins = terrain.terrain_origins
        if terrain_origins is None:
            raise RuntimeError("Terrain origins are None — set use_terrain_origins on TerrainImporterCfg.")
        if isinstance(terrain_origins, np.ndarray):
            terrain_origins = torch.from_numpy(terrain_origins).float().to(self.device)
        else:
            terrain_origins = terrain_origins.to(self.device)
        num_rows, num_cols = terrain_origins.shape[:2]

        self._gate_positions = torch.zeros(self.num_envs, num_gates, 3, device=self.device)
        self._gate_orientations = base_gate_ori.unsqueeze(0).expand(self.num_envs, -1, -1).clone()

        for env_id in range(self.num_envs):
            row = (env_id // num_cols) % num_rows
            col = env_id % num_cols
            origin = terrain_origins[row, col]
            self._gate_positions[env_id] = base_gate_pos + origin

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

        if gate_log:
            self.extras["log"].update(gate_log)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based hovering / waypoint-following env (standalone extension)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn

from .hovering_quadcopter_env_cfg import HoveringQuadcopterEnvCfg, WAYPOINTS_ENV_VAR
from .mdp.events import collect_episode_metrics_before_reset
from .mdp.waypoint_io import load_waypoints_from_yaml
from .mdp.waypoint_utils import update_waypoint_state
from .terrain import broadcast_yaml_track, generate_random_tracks
from .vis import draw_waypoints


class HoveringQuadcopterRLEnv(ManagerBasedRLEnv):
    """Hovering env with per-terrain-cell waypoint tracks and dwell-counting collection."""

    cfg: HoveringQuadcopterEnvCfg

    def __init__(self, cfg: HoveringQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        self._robot: Articulation
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self):
        # _prepare_waypoint_task allocates per-env GPU buffers that reward /
        # observation callables read at manager-construction time.
        self._prepare_waypoint_task()
        super().load_managers()
        if str(self.device) == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.terrain.prim_path])

    # ------------------------------------------------------------------
    # Task preparation
    # ------------------------------------------------------------------
    def _prepare_waypoint_task(self) -> None:
        self._robot = self.scene["robot"]

        self._actions_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions_buf)
        self._prev_prev_actions = torch.zeros_like(self._actions_buf)
        self._action_noise_bias = torch.zeros_like(self._actions_buf)
        # Raw policy output (pre-clamp, pre-noise) — read by penalty_oversaturation.
        self._unclamped_actions_buf = torch.zeros_like(self._actions_buf)

        self._setup_waypoint_tracks()

        # Per-env state for proximity hysteresis / dwell counting.
        self._current_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collection_timer = torch.zeros(self.num_envs, device=self.device)
        self._collected_step_flag = torch.zeros(self.num_envs, device=self.device)
        self._all_collected_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Aliased name kept for parity with the racing event helpers we re-use.
        self._gates_passed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._episode_collected_count = torch.zeros(self.num_envs, device=self.device)
        self._curr_dist_to_waypoint = torch.zeros(self.num_envs, device=self.device)
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)
        base_link_ids, _ = self._robot.find_bodies("base_link")
        self._base_link_body_idx = int(base_link_ids[0])

    def _setup_waypoint_tracks(self) -> None:
        """Generate per-cell tracks (random by default, YAML if ``WAYPOINTS`` set)
        and broadcast them onto the per-env world-frame buffers.
        """
        terrain = self.scene.terrain
        if terrain is None or terrain.cfg.terrain_generator is None:
            raise RuntimeError("Hovering env requires a generator terrain.")
        terrain_cfg = terrain.cfg.terrain_generator

        terrain_origins = terrain.terrain_origins
        if terrain_origins is None:
            raise RuntimeError("Terrain origins are None — set use_terrain_origins.")
        if isinstance(terrain_origins, np.ndarray):
            terrain_origins_t = torch.from_numpy(terrain_origins).float().to(self.device)
        else:
            terrain_origins_t = terrain_origins.to(self.device)
        num_rows, num_cols = terrain_origins_t.shape[:2]
        num_cells = num_rows * num_cols

        if getattr(self.cfg, "eval_cardinal_spawn", False):
            self._setup_cardinal_eval_layout(terrain_origins_t, num_rows, num_cols)
            return

        # TerrainGenerator places each cell so its CENTRE sits at terrain_origin,
        # but the cell mesh is built corner-origin → subtract size/2 to bridge.
        centre_offset = np.array([-terrain_cfg.size[0] / 2.0, -terrain_cfg.size[1] / 2.0, 0.0])

        # YAML override has highest priority.
        yaml_path = os.environ.get(WAYPOINTS_ENV_VAR, self.cfg.waypoint_track_yaml)
        if yaml_path:
            waypoints_local, tc_yaml = load_waypoints_from_yaml(yaml_path)
            tc = float(tc_yaml) if tc_yaml is not None else float(self.cfg.collection_time_s)
            tracks = broadcast_yaml_track(waypoints_local, tc, num_cells)
            print(f"[INFO] Hovering env: loaded fixed waypoint track from {yaml_path} (T_c={tc:.2f} s).")
        else:
            tracks = generate_random_tracks(
                base_seed=self.cfg.seed,
                num_cells=num_cells,
                num_waypoints_range=self.cfg.num_waypoints_range,
                waypoint_xy_range=self.cfg.waypoint_xy_range,
                waypoint_z_range=self.cfg.waypoint_z_range,
                min_waypoint_spacing=self.cfg.min_waypoint_spacing,
                cell_size=self.cfg.cell_size,
                num_zero_tc_cells=self.cfg.num_zero_tc_cells,
                collection_time_range_s=self.cfg.collection_time_range_s,
            )

        max_waypoints = max(t.waypoints.shape[0] for t in tracks)
        self._waypoint_positions = torch.zeros(self.num_envs, max_waypoints, 3, device=self.device)
        self._waypoint_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._collection_time = torch.zeros(self.num_envs, device=self.device)

        # Pre-stage every cell's track into device tensors once, then index per env.
        cell_positions: list[torch.Tensor] = []
        for track in tracks:
            local = torch.from_numpy(track.waypoints + centre_offset).float().to(self.device)
            cell_positions.append(local)

        for env_id in range(self.num_envs):
            row = (env_id // num_cols) % num_rows
            col = env_id % num_cols
            cell_idx = (row * num_cols + col) % num_cells
            origin = terrain_origins_t[row, col]  # [3] world frame
            local = cell_positions[cell_idx]      # [G, 3] centred local frame
            g = local.shape[0]
            self._waypoint_positions[env_id, :g] = local + origin
            self._waypoint_counts[env_id] = g
            self._collection_time[env_id] = tracks[cell_idx].collection_time_s

        # Visualization: red sphere per unique waypoint across all 64 cells.
        if self.cfg.draw_waypoints:
            all_positions = []
            for row in range(num_rows):
                for col in range(num_cols):
                    cell_idx = (row * num_cols + col) % num_cells
                    origin = terrain_origins_t[row, col].cpu().numpy()
                    local = tracks[cell_idx].waypoints + centre_offset
                    all_positions.append(local + origin)
            stacked = np.concatenate(all_positions, axis=0)
            # Stash the markers handle so the prim stays alive.
            self._waypoint_markers = draw_waypoints(stacked, device=self.device)

    # ------------------------------------------------------------------
    # Cardinal-spawn evaluation layout (mirrors test_controller_offline.py)
    # ------------------------------------------------------------------
    # Cell ordering → cardinal direction the drone spawns from, relative to
    # the cell-local goal at [0, 0, 1]. World frame is FLU (+X east, +Y north).
    _CARDINAL_OFFSETS: tuple[tuple[float, float, float], ...] = (
        (+1.0, 0.0, 0.0),  # cell 0 — east   (drone +X of goal)
        (-1.0, 0.0, 0.0),  # cell 1 — west   (drone −X of goal)
        (0.0, +1.0, 0.0),  # cell 2 — north  (drone +Y of goal)
        (0.0, -1.0, 0.0),  # cell 3 — south  (drone −Y of goal)
    )
    _CARDINAL_LABELS: tuple[str, ...] = ("east", "west", "north", "south")

    def _setup_cardinal_eval_layout(
        self,
        terrain_origins_t: torch.Tensor,
        num_rows: int,
        num_cols: int,
    ) -> None:
        """Place exactly one waypoint per cell at world position
        ``terrain_origin + [0, 0, 1]`` and assign each env one of the four
        cardinal spawn offsets based on ``env_id % 4``.

        Requires ``num_rows * num_cols == 4`` (any 1×4 / 4×1 / 2×2 layout works
        — the env→cell mapping below uses the same scheme as the random track
        path so cell_idx is well-defined).
        """
        num_cells = num_rows * num_cols
        assert num_cells == 4, (
            f"eval_cardinal_spawn requires exactly 4 terrain cells (got "
            f"{num_rows}×{num_cols}={num_cells})."
        )

        self._waypoint_positions = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._waypoint_counts = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._collection_time = torch.full(
            (self.num_envs,), float(self.cfg.collection_time_s), device=self.device
        )
        self._eval_spawn_offset = torch.zeros(self.num_envs, 3, device=self.device)

        offsets_t = torch.tensor(self._CARDINAL_OFFSETS, device=self.device, dtype=torch.float32)
        goal_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Cache per-cell world goal for vis below.
        cell_goals_world: list[torch.Tensor] = []
        for cell_idx in range(num_cells):
            row = cell_idx // num_cols
            col = cell_idx % num_cols
            cell_goals_world.append(terrain_origins_t[row, col] + goal_local)

        for env_id in range(self.num_envs):
            row = (env_id // num_cols) % num_rows
            col = env_id % num_cols
            cell_idx = (row * num_cols + col) % num_cells
            self._waypoint_positions[env_id, 0] = cell_goals_world[cell_idx]
            self._eval_spawn_offset[env_id] = offsets_t[cell_idx]

        print(
            "[INFO] Hovering eval (cardinal): "
            + ", ".join(
                f"cell{i}={self._CARDINAL_LABELS[i]} @ "
                f"{cell_goals_world[i].cpu().numpy().round(3).tolist()}"
                for i in range(num_cells)
            )
        )

        if self.cfg.draw_waypoints:
            stacked = torch.stack(cell_goals_world, dim=0).cpu().numpy()
            self._waypoint_markers = draw_waypoints(stacked, device=self.device)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Snapshot the raw policy output BEFORE the action term clamps and
        # applies noise/lag, so penalty_oversaturation can compare the raw
        # request against its clamped version.
        action_dev = action.to(self.device)
        self._unclamped_actions_buf[:] = action_dev
        self.action_manager.process_action(action_dev)
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

        # Refresh per-env waypoint state before terminations / rewards.
        update_waypoint_state(self)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

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

        # Snapshot metrics before super() triggers reset events that zero them.
        log_extra = collect_episode_metrics_before_reset(self, env_ids_tensor)

        super()._reset_idx(env_ids)

        if log_extra:
            self.extras.setdefault("log", {})
            self.extras["log"].update(log_extra)

        # SKRL expects scalar tensors in extras["log"]; coerce floats.
        if self.extras.get("log"):
            log_tensors: dict[str, torch.Tensor] = {}
            for key, value in self.extras["log"].items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    log_tensors[key] = value.detach().reshape(()).to(device=self.device, dtype=torch.float32)
                elif isinstance(value, (float, int, bool)):
                    log_tensors[key] = torch.tensor(float(value), device=self.device, dtype=torch.float32)
            self.extras["log"] = log_tensors
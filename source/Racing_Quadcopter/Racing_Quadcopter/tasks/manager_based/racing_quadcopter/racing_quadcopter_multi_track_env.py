# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-track racing env: one fixed circuit per terrain.

Only ``_load_cell_gate_data`` is overridden. All buffer allocation, gate-tracking
state initialisation, and vis logic live in the parent's ``_setup_gate_tracking``
and are inherited unchanged.

The parent's default ``_load_cell_gate_data`` uses a hand-rolled env→cell formula
that does not match TerrainImporter's curriculum-mode assignment. This override
uses ``scene.env_origins`` (closest-cell match) which is correct for both modes,
and reads gate specs directly from the sub_terrain cfgs so no registry lookup is
needed.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from Racing_Quadcopter.terrains import GateData, MeshFixedGatesTerrainCfg, gate_specs_to_gate_data

from .racing_quadcopter_env import RacingQuadcopterRLEnv


class RacingQuadcopterMultiTrackRLEnv(RacingQuadcopterRLEnv):
    """Variant where each terrain hosts a different fixed circuit.

    Configure with :class:`RacingQuadcopterEnvCfg_MULTI_TRACK` (terrain built via
    ``create_multi_track_terrain``). All other env behaviour is inherited unchanged.
    """

    def _load_cell_gate_data(
        self,
        terrain_origins: torch.Tensor,
        centre_offset: np.ndarray,
    ) -> tuple[list, list, list, list, np.ndarray, torch.Tensor]:
        """Read gate specs from sub_terrain cfgs; map envs to terrains via env_origins."""
        terrain_cfg = self.scene.terrain.cfg.terrain_generator
        sub_cfgs = list(terrain_cfg.sub_terrains.values())

        if not sub_cfgs:
            raise RuntimeError("Multi-track env requires at least one sub_terrain.")
        for cfg in sub_cfgs:
            if not isinstance(cfg, MeshFixedGatesTerrainCfg):
                raise TypeError(
                    f"Expected MeshFixedGatesTerrainCfg sub_terrains, got {type(cfg).__name__}."
                )

        num_rows, num_cols = terrain_origins.shape[:2]
        if num_cols != len(sub_cfgs):
            raise RuntimeError(
                f"terrain has {num_cols} columns but {len(sub_cfgs)} sub_terrains configured."
            )

        per_cell_pos: list[torch.Tensor] = []
        per_cell_ori: list[torch.Tensor] = []
        per_cell_sizes: list[float] = []
        per_col_gate_data: list[GateData] = []
        for cfg in sub_cfgs:
            if not cfg.gates:
                raise ValueError("MeshFixedGatesTerrainCfg.gates must not be empty.")
            gd = gate_specs_to_gate_data(cfg.gates, cfg.gate_size, cfg.bar_thickness)
            per_col_gate_data.append(gd)
            local_pos = gd.positions + centre_offset
            per_cell_pos.append(torch.from_numpy(local_pos).float().to(self.device))
            per_cell_ori.append(torch.from_numpy(gd.orientations).float().to(self.device))
            per_cell_sizes.append(float(cfg.gate_size))

        # Resolve env→cell from TerrainImporter's actual placement (closest cell centre).
        env_origins_world = self.scene.env_origins.to(self.device)       # [N, 3]
        flat_cells = terrain_origins.reshape(-1, 3)                       # [R*C, 3]
        dists = ((env_origins_world.unsqueeze(1) - flat_cells.unsqueeze(0)) ** 2).sum(dim=-1)
        cell_lookup = dists.argmin(dim=1)                                 # [N]
        env_col = (cell_lookup % num_cols).cpu().numpy()                  # index into per_cell_* (= terrain index)
        env_origins = flat_cells[cell_lookup]                             # [N, 3] matching world origins

        if os.getenv("DEBUG_TERRAIN") == "1":
            print(f"DEBUG_TERRAIN: multi_track num_rows={num_rows} num_cols={num_cols}")
            uniq, counts = np.unique(env_col, return_counts=True)
            print(f"DEBUG_TERRAIN: env_col_distribution={dict(zip(uniq.tolist(), counts.tolist()))}")

        # gate_datas in cell-major order for vis (row * num_cols + col).
        # With num_rows=1, this is just per_col_gate_data.
        gate_datas = [per_col_gate_data[c] for _ in range(num_rows) for c in range(num_cols)]

        return per_cell_pos, per_cell_ori, per_cell_sizes, gate_datas, env_col, env_origins


class RacingQuadcopterSkillEvalRLEnv(RacingQuadcopterMultiTrackRLEnv):
    """Skill-evaluation env: multi-track layout with per-terrain spawn logic.

    Extends :class:`RacingQuadcopterMultiTrackRLEnv` by storing the per-env terrain
    assignment as ``self._env_col_idx`` after gate tracking is set up. The custom
    reset event ``reset_skill_eval_spawn`` (registered in
    :class:`RacingQuadcopterSkillEvalEnvCfg`) reads this buffer to dispatch each
    resetting env to the spawn logic defined in its terrain's :class:`SkillTerrainCfg`.
    """

    def _load_cell_gate_data(
        self,
        terrain_origins: torch.Tensor,
        centre_offset: np.ndarray,
    ) -> tuple[list, list, list, list, np.ndarray, torch.Tensor]:
        result = super()._load_cell_gate_data(terrain_origins, centre_offset)
        _, _, _, _, env_col, _ = result
        # Store as a long tensor so it can be used for indexing in the reset event.
        self._env_col_idx = torch.from_numpy(env_col.astype(int)).long().to(self.device)
        return result

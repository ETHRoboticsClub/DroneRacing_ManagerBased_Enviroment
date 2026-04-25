# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter thrust + moment action term."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class QuadcopterThrustAction(ActionTerm):
    """Apply vertical thrust and body moments from normalized actions [-1, 1]."""

    cfg: QuadcopterThrustActionCfg
    _asset: Articulation

    def __init__(self, cfg: QuadcopterThrustActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, self.cfg.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._thrust = torch.zeros(env.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(env.num_envs, 1, 3, device=self.device)
        self._body_id = self._asset.find_bodies("body")[0]
        masses = self._asset.root_physx_view.get_masses()[0].sum()
        self._gravity_mag = torch.tensor(env.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (masses * self._gravity_mag).item()

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # Keep a copy of previous command for smoothness penalties and actuator-lag filtering.
        prev_cmd = self._raw_actions.clone()

        cmd = actions.clamp(-1.0, 1.0)

        # Conventional action-space domain randomization: additive noise, bias, and lag.
        noise_cfg = getattr(self._env.cfg, "action_noise", None)
        if noise_cfg is not None and getattr(noise_cfg, "enable", False):
            std = float(getattr(noise_cfg, "gaussian_std", 0.0))
            if std > 0.0:
                cmd = cmd + torch.randn_like(cmd, device=self.device) * std

            if getattr(noise_cfg, "per_episode_bias", False) and hasattr(self._env, "_action_noise_bias"):
                cmd = cmd + self._env._action_noise_bias

            alpha = float(getattr(noise_cfg, "lpf_alpha", 1.0))
            if alpha < 1.0:
                cmd = alpha * cmd + (1.0 - alpha) * prev_cmd

            if getattr(noise_cfg, "clip_actions", True):
                cmd = cmd.clamp(-1.0, 1.0)

        if hasattr(self._env, "_prev_actions"):
            self._env._prev_actions[:] = prev_cmd

        self._raw_actions[:] = cmd

        if hasattr(self._env, "_actions_buf"):
            self._env._actions_buf[:] = self._raw_actions
        thrust_cmd = (self._raw_actions[:, 0] + 1.0) * 0.5 # normalize the thrust command to be between 0 and 1
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * thrust_cmd
        self._moment[:, 0, :] = self.cfg.moment_scale * self._raw_actions[:, 1:]
        self._processed_actions[:] = self._raw_actions

    def apply_actions(self):
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._thrust, torques=self._moment, body_ids=self._body_id
        )


@configclass
class QuadcopterThrustActionCfg(ActionTermCfg):
    """Configuration for :class:`QuadcopterThrustAction`."""

    class_type: type[ActionTerm] = QuadcopterThrustAction
    asset_name: str = "robot"

    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    action_dim: int = 4

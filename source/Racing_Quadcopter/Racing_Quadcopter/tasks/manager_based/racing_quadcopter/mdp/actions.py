# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter thrust + moment action term."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.buffers.delay_buffer import DelayBuffer

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
        self._delayed_actions = torch.zeros_like(self._raw_actions)
        self._thrust = torch.zeros(env.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(env.num_envs, 1, 3, device=self.device)
        self._body_id = self._asset.find_bodies(self.cfg.body_name)[0]
        masses = self._asset.root_physx_view.get_masses()[0].sum()
        self._gravity_mag = torch.tensor(env.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (masses * self._gravity_mag).item()
        self.actions_delay_buffer = DelayBuffer(self.cfg.max_delay, env.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or env_ids == slice(None):
            num_envs = self._env.num_envs
        else:
            num_envs = len(env_ids)

        if self.cfg.max_delay > 0:
            time_lags = torch.randint(
                low=self.cfg.min_delay,
                high=self.cfg.max_delay + 1,
                size=(num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.actions_delay_buffer.set_time_lag(time_lags, env_ids)
        else:
            self.actions_delay_buffer.set_time_lag(0, env_ids)

        self.actions_delay_buffer.reset(env_ids)

    def process_actions(self, actions: torch.Tensor):
        # Save previous applied (delayed) action for smoothness penalty before overwriting.
        if hasattr(self._env, "_prev_actions"):
            if hasattr(self._env, "_actions_buf"):
                self._env._prev_actions[:] = self._env._actions_buf
            else:
                self._env._prev_actions[:] = self._raw_actions

        # Keep a copy of the previous raw command for LPF filtering.
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

        self._raw_actions[:] = cmd
        self._delayed_actions[:] = self.actions_delay_buffer.compute(self._raw_actions)

        if hasattr(self._env, "_actions_buf"):
            self._env._actions_buf[:] = self._delayed_actions
        thrust_cmd = (self._delayed_actions[:, 0] + 1.0) * 0.5 # normalize the thrust command to be between 0 and 1
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * thrust_cmd
        self._moment[:, 0, :] = self.cfg.moment_scale * self._delayed_actions[:, 1:]
        self._processed_actions[:] = self._delayed_actions

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
    min_delay: int = 0
    max_delay: int = 0
    body_name: str = "body"  # base_link for anemos (gets overridden by env cfg)

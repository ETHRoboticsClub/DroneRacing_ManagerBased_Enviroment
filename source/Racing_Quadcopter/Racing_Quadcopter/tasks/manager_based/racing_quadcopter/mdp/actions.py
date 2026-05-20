# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
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


def build_quad_mixer_b_inv(k_t: float, k_d: float, arm_length_m: float) -> np.ndarray:
    """Invertible X-frame mixer B matrix
    
    x: front, y: left, z: up
    M1: (+l, +l) CCW
    M2: (+l, -l) CW
    M3: (-l, +l) CW
    M4: (-l, -l) CCW

    Parameters ``k_t``, ``k_d``, ``arm_length_m`` match ``DroneProfile`` / env cfg.
    Rows are thrust, roll moment, pitch moment, yaw drag torque (see project notes / ``docs/drone.pdf``).
    """
    ktl = k_t * arm_length_m / np.sqrt(2.0)
    b = np.array(
        [
            [k_t, k_t, k_t, k_t],
            [ktl, -ktl, ktl, -ktl],
            [-ktl, -ktl, ktl, ktl],
            [k_d, -k_d, -k_d, k_d],
        ],
        dtype=np.float64,
    )
    #
    return np.linalg.inv(b)


class QuadcopterMomentsAction(ActionTerm):
    """Apply vertical thrust and body moments from normalized actions [-1, 1]."""

    cfg: QuadcopterMomentsActionCfg
    _asset: Articulation

    def __init__(self, cfg: QuadcopterMomentsActionCfg, env: ManagerBasedEnv):
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
        self._moment_scale = torch.tensor(self.cfg.moment_scale, device=self.device, dtype=torch.float32)
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
        self._moment[:, 0, :] = self._moment_scale * self._delayed_actions[:, 1:]
        self._processed_actions[:] = self._delayed_actions

        # [T, τ]^T = B u^{∘2}  →  u^{∘2} = B^{-1} [T, τ]^T,  u = sqrt( clamp(u^{∘2}, 0) )
        if hasattr(self._env, "_mixer_b_inv") and hasattr(self._env, "_motor_speeds"):
            T_col = self._thrust[:, 0, 2]
            tau_b = self._moment[:, 0, :]
            wrench = torch.cat([T_col.unsqueeze(-1), tau_b], dim=-1).to(
                dtype=self._env._mixer_b_inv.dtype
            )
            omega_sq = wrench @ self._env._mixer_b_inv.T
            self._env._motor_speeds[:] = torch.sqrt(torch.clamp(omega_sq, min=0.0))

    def apply_actions(self):
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._thrust, torques=self._moment, body_ids=self._body_id
        )


@configclass
class QuadcopterMomentsActionCfg(ActionTermCfg):
    """Configuration for :class:`QuadcopterMomentsAction`."""

    class_type: type[ActionTerm] = QuadcopterMomentsAction
    asset_name: str = "robot"

    thrust_to_weight: float = 1.9
    moment_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)  # (roll, pitch, yaw) N·m
    action_dim: int = 4
    min_delay: int = 0
    max_delay: int = 0
    body_name: str = "body"  # base_link for anemos (gets overridden by env cfg)


class QuadcopterRatesAction(ActionTerm):
    """Thrust on body Z plus inner-loop PID+F on body rates.

    Per-axis gains follow Betaflight-style numerators × per-term scales from
    :class:`QuadcopterRatesActionCfg`. Optional ``pid_gain_randomization_frac`` rescales
    coefficients independently per env on each reset.
    Torque uses :math:`\\tau = K_p e + K_i\\int e\\,dt - K_d \\dot\\omega + K_f \\dot\\omega^{des}`
    with measured body rate :math:`\\omega` and rate setpoint :math:`\\omega^{des}`.

    Uses clamped actions only (no action delay buffer or env ``action_noise`` path).
    """

    cfg: QuadcopterRatesActionCfg
    _asset: Articulation

    def __init__(self, cfg: QuadcopterRatesActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, self.cfg.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._delayed_actions = torch.zeros_like(self._raw_actions)
        self._thrust = torch.zeros(env.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(env.num_envs, 1, 3, device=self.device)
        self._omega_prev = torch.zeros(env.num_envs, 3, device=self.device)
        self._omega_des_prev = torch.zeros(env.num_envs, 3, device=self.device)
        self._rate_integral = torch.zeros(env.num_envs, 3, device=self.device)
        self._rate_pid_hold_derivative = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
        self._body_id = self._asset.find_bodies(self.cfg.body_name)[0]
        masses = self._asset.root_physx_view.get_masses()[0].sum()
        self._gravity_mag = torch.tensor(env.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (masses * self._gravity_mag).item()

        def _axis_coeff(name: str, scale: float) -> torch.Tensor:
            t = torch.tensor(getattr(cfg, name), device=self.device, dtype=torch.float32)
            return t * scale * cfg.pid_scales

        self._kp_base = _axis_coeff("pid_p", cfg.pterm_scale)
        self._ki_base = _axis_coeff("pid_i", cfg.iterm_scale)
        self._kd_base = _axis_coeff("pid_d", cfg.dterm_scale)
        self._kf_base = _axis_coeff("pid_f", cfg.feedforward_scale)

        n = env.num_envs
        self._kp_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._ki_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._kd_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._kf_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._resample_pid_gain_randomization(None)
        self.actions_delay_buffer = DelayBuffer(self.cfg.max_delay, env.num_envs, device=self.device)

    def _resample_pid_gain_randomization(self, env_ids: Sequence[int] | None) -> None:
        """Uniform multipliers in ``[1-f, 1+f]`` per env × axis for each PIDF term (``f`` from cfg)."""
        f = self.cfg.pid_gain_randomization_frac
        if f is None:
            return
        low, high = 1.0 - float(f), 1.0 + float(f)
        eids = (
            torch.arange(self._env.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None or env_ids == slice(None)
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        k = int(eids.shape[0])
        dev, dt = self.device, torch.float32
        for t in (self._kp_rand, self._ki_rand, self._kd_rand, self._kf_rand):
            t[eids] = torch.empty((k, 3), device=dev, dtype=dt).uniform_(low, high)

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
            self._omega_prev.zero_()
            self._omega_des_prev.zero_()
            self._rate_integral.zero_()
            self._rate_pid_hold_derivative.fill_(True)
        else:
            eids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._omega_prev[eids] = 0.0
            self._omega_des_prev[eids] = 0.0
            self._rate_integral[eids] = 0.0
            self._rate_pid_hold_derivative[eids] = True

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

        self._resample_pid_gain_randomization(env_ids)

    def process_actions(self, actions: torch.Tensor):
        # Save previous applied (delayed) action for smoothness penalty before overwriting.
        if hasattr(self._env, "_prev_actions"):
            if hasattr(self._env, "_actions_buf"):
                self._env._prev_actions[:] = self._env._actions_buf
            else:
                self._env._prev_actions[:] = self._raw_actions

        prev_cmd = self._raw_actions.clone()
        cmd = actions.clamp(-1.0, 1.0)

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

        thrust_cmd = (self._delayed_actions[:, 0] + 1.0) * 0.5
        #self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * thrust_cmd

        # Hardcoded, TODO: UNDO LATER!
        self._thrust[:, 0, 2] = 34.192 * thrust_cmd

        omega_meas = self._asset.data.root_ang_vel_b
        omega_des = self._delayed_actions[:, 1:] * self.cfg.rate_command_scale
        err = omega_des - omega_meas

        dt = self._env.cfg.sim.dt

        self._rate_integral = self._rate_integral + err * dt
        ilim = self.cfg.rate_integral_limit
        if ilim is not None and ilim > 0.0:
            self._rate_integral.clamp_(-ilim, ilim)

        omega_dot = torch.zeros_like(omega_meas)
        omega_des_dot = torch.zeros_like(omega_meas)
        mask_deriv = ~self._rate_pid_hold_derivative
        omega_dot[mask_deriv] = (omega_meas[mask_deriv] - self._omega_prev[mask_deriv]) / dt
        omega_des_dot[mask_deriv] = (omega_des[mask_deriv] - self._omega_des_prev[mask_deriv]) / dt
        self._omega_prev[:] = omega_meas
        self._omega_des_prev[:] = omega_des
        self._rate_pid_hold_derivative[:] = False

        tau = (
            self._kp_rand * self._kp_base * err
            + self._ki_rand * self._ki_base * self._rate_integral
            - self._kd_rand * self._kd_base * omega_dot
            + self._kf_rand * self._kf_base * omega_des_dot
        )

        tlim = self.cfg.torque_limit
        if tlim is not None and tlim > 0.0:
            tau = tau.clamp(-tlim, tlim)

        self._moment[:, 0, :] = tau
        self._processed_actions[:] = self._delayed_actions

        if hasattr(self._env, "_mixer_b_inv") and hasattr(self._env, "_motor_speeds"):
            T_col = self._thrust[:, 0, 2]
            tau_b = self._moment[:, 0, :]
            wrench = torch.cat([T_col.unsqueeze(-1), tau_b], dim=-1).to(
                dtype=self._env._mixer_b_inv.dtype
            )
            omega_sq = wrench @ self._env._mixer_b_inv.T
            self._env._motor_speeds[:] = torch.sqrt(torch.clamp(omega_sq, min=0.0))

    def apply_actions(self):
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._thrust, torques=self._moment, body_ids=self._body_id
        )


@configclass
class QuadcopterRatesActionCfg(ActionTermCfg):
    """Configuration for :class:`QuadcopterRatesAction`.

    Integer tuples ``(roll, pitch, yaw)`` match Betaflight ``pidf_t`` (``P``, ``I``, ``D``, ``F``).
    Defaults for ``pid_*`` follow ``PID_ROLL_DEFAULT``, ``PID_PITCH_DEFAULT``, ``PID_YAW_DEFAULT``.
    Each ``pid_*`` axis value is multiplied by the matching ``*term_scale`` field below (Betaflight
    maps integers to floats with the same names in ``pid.h``).
    """

    class_type: type[ActionTerm] = QuadcopterRatesAction
    asset_name: str = "robot"

    thrust_to_weight: float = 1.9
    action_dim: int = 4
    min_delay: int = 0
    max_delay: int = 0
    body_name: str = "body"  # base_link for anemos (gets overridden by env cfg)

    #: Maps normalized rate sticks ``[-1, 1]`` to desired body angular rates [rad/s].
    rate_command_scale: float = 20.0 #This has to be defined based on the drone body rates limits

    #: Source for default values: ``https://github.com/betaflight/betaflight/blob/master/src/main/flight/pid.h``
    pterm_scale: float = 0.032029
    iterm_scale: float = 0.244381
    dterm_scale: float = 0.000529
    feedforward_scale: float = 0.013754

    #: Global scale applied to all PIDF coefficients after ``*term_scale`` (sim tuning knob).
    pid_scales: float = 0.0001
    #: Betaflight-style PIDF numerators per axis ``(roll, pitch, yaw)``; each term uses its ``*term_scale``.
    pid_p: tuple[float, float, float] = (45.0, 47.0, 45.0)
    pid_i: tuple[float, float, float] = (80.0, 84.0, 80.0)
    pid_d: tuple[float, float, float] = (30.0, 34.0, 0.0)
    pid_f: tuple[float, float, float] = (120.0, 125.0, 120.0)

    #: Domain randomization: on each env reset, sample independent uniform multipliers in
    #: ``[1 - f, 1 + f]`` for each PIDF term and each axis (e.g. ``0.2`` → ±20% around nominal).
    #: Keep ``f < 1`` so multipliers stay positive. ``None`` disables (multipliers remain 1).
    pid_gain_randomization_frac: float | None = None

    #: Clamp each axis of the integrated rate-error state (anti-windup helper).
    rate_integral_limit: float | None = 2.0
    #: Per-axis torque magnitude limit [N·m]; ``None`` disables.
    torque_limit: float | None = None

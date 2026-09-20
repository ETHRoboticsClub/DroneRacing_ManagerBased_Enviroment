# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import json
import numpy as np
import torch
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.buffers.delay_buffer import DelayBuffer
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_SITL_ROTOR_POSITIONS_BODY = (
    (-0.081108, 0.086485, 0.044629),
    (0.099477, 0.067468, 0.044629),
    (-0.073632, -0.075213, 0.044629),
    (0.086702, -0.054547, 0.044629),
)
_SITL_ROTOR_DIRECTIONS = (-1.0, 1.0, 1.0, -1.0)
_SITL_MASS_KG = 0.8351086449771665
_SITL_MAX_TOTAL_THRUST_N = 32.7584728
_SITL_YAW_TORQUE_COEFFICIENT = 0.00125
_SITL_YAW_DAMPING_COEFFICIENT = 0.006


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
        buffer_max_lag = max(int(self.cfg.max_delay), int(self.cfg.delay_buffer_max_lag))
        self.actions_delay_buffer = DelayBuffer(buffer_max_lag, env.num_envs, device=self.device)

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
        ids = (
            torch.arange(self._env.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None or isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._delayed_actions[ids] = 0.0
        self._thrust[ids] = 0.0
        self._moment[ids] = 0.0
        if env_ids is None or isinstance(env_ids, slice):
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
            if hasattr(self._env, "_prev_prev_actions") and hasattr(self._env, "_actions_buf"):
                self._env._prev_prev_actions[:] = self._env._prev_actions
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

            alpha_tensor = getattr(self._env, "_action_noise_lpf_alpha", None)
            if isinstance(alpha_tensor, torch.Tensor):
                alpha = alpha_tensor
            else:
                alpha = float(getattr(noise_cfg, "lpf_alpha", 1.0))
            if isinstance(alpha, torch.Tensor):
                cmd = alpha * cmd + (1.0 - alpha) * prev_cmd
            elif alpha < 1.0:
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
    # Reserve capacity for progressive curricula while keeping the active delay at zero.
    delay_buffer_max_lag: int = 2
    body_name: str = "body"  # base_link for anemos (gets overridden by env cfg)


class QuadcopterRatesAction(ActionTerm):
    """Thrust on body Z plus inner-loop PID+F on body rates.

    Per-axis gains follow Betaflight-style numerators × per-term scales from
    :class:`QuadcopterRatesActionCfg`. Optional ``pid_gain_randomization_frac`` rescales
    coefficients independently per env on each reset.
    Torque (rate loop, derivative on measurement — no setpoint kick on D):

        e = ω_des − ω
        τ = Kp·e + Ki·∫e dt − Kd·ω_dot_meas + Kf·(ω_des − ω_des_prev)/Δt

    ``pid_*`` numerators map to N·m via ``pid_scales`` and ``*term_scale`` (set
    ``dterm_scale=1`` when ``pid_d`` holds physical Kd from ``design_rate_pd_gains.py``).

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
        self._thrust_to_weight_base = float(cfg.thrust_to_weight)

        def _axis_coeff(name: str, scale: float) -> torch.Tensor:
            t = torch.tensor(getattr(cfg, name), device=self.device, dtype=torch.float32)
            return t * scale * cfg.pid_scales

        self._kp_base = _axis_coeff("pid_p", cfg.pterm_scale)
        self._ki_base = _axis_coeff("pid_i", cfg.iterm_scale)
        self._kd_base = _axis_coeff("pid_d", cfg.dterm_scale)
        self._kf_base = _axis_coeff("pid_f", cfg.feedforward_scale)

        self._sitl_enabled = self.cfg.actuator_model == "betaflight_sitl"
        self._sitl_max_thrust_per_motor = float(self.cfg.sitl_max_total_thrust) / 4.0
        self._sitl_rotor_positions = torch.tensor(
            self.cfg.sitl_rotor_positions_body,
            device=self.device,
            dtype=torch.float32,
        )
        self._sitl_rotor_directions = torch.tensor(
            self.cfg.sitl_rotor_directions,
            device=self.device,
            dtype=torch.float32,
        )
        self._sitl_allocation = self._build_sitl_allocation_matrix()
        self._sitl_allocation_pinv = torch.linalg.pinv(self._sitl_allocation)
        self._sitl_throttle_curve_x, self._sitl_throttle_curve_y = self._load_sitl_throttle_curve()

        n = env.num_envs
        self._kp_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._ki_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._kd_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._kf_rand = torch.ones(n, 3, device=self.device, dtype=torch.float32)
        self._thrust_to_weight = torch.full(
            (n,), self._thrust_to_weight_base, device=self.device, dtype=torch.float32
        )
        self._linear_drag = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        self._quadratic_drag = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        self._angular_drag = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        self._resample_pid_gain_randomization(None)
        self._resample_thrust_randomization(None)
        self._resample_aero_randomization(None)
        buffer_max_lag = max(int(self.cfg.max_delay), int(self.cfg.delay_buffer_max_lag))
        self.actions_delay_buffer = DelayBuffer(buffer_max_lag, env.num_envs, device=self.device)

    def _env_ids_tensor(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        return (
            torch.arange(self._env.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None or isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )

    def _build_sitl_allocation_matrix(self) -> torch.Tensor:
        """Maps per-motor thrusts to body wrench [Tz, tau_x, tau_y, tau_z]."""
        x = self._sitl_rotor_positions[:, 0]
        y = self._sitl_rotor_positions[:, 1]
        return torch.stack(
            (
                torch.ones(4, device=self.device, dtype=torch.float32),
                y,
                -x,
                self._sitl_rotor_directions * float(self.cfg.sitl_yaw_torque_coefficient),
            ),
            dim=0,
        )

    def _load_sitl_throttle_curve(self) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.cfg.sitl_calibration_path
        points = [(0.0, 0.0), (1.0, 1.0)]
        if path:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = Path.cwd() / p
            with p.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            points = data.get("throttle_to_motor_output", points)
        if len(points) < 2:
            raise ValueError("SITL throttle calibration requires at least two points")
        points = sorted((float(x), float(y)) for x, y in points)
        xs = torch.tensor([p[0] for p in points], device=self.device, dtype=torch.float32)
        ys = torch.tensor([p[1] for p in points], device=self.device, dtype=torch.float32)
        if torch.any(xs[1:] <= xs[:-1]):
            raise ValueError("SITL throttle calibration x values must be strictly increasing")
        return xs, ys

    def _sitl_throttle_to_motor_output(self, throttle: torch.Tensor) -> torch.Tensor:
        x = self._sitl_throttle_curve_x
        y = self._sitl_throttle_curve_y
        throttle = throttle.clamp(float(x[0]), float(x[-1]))
        idx = torch.searchsorted(x, throttle, right=True).clamp(1, x.numel() - 1)
        x0 = x[idx - 1]
        x1 = x[idx]
        y0 = y[idx - 1]
        y1 = y[idx]
        alpha = (throttle - x0) / (x1 - x0)
        return y0 + alpha * (y1 - y0)

    def _resample_pid_gain_randomization(self, env_ids: Sequence[int] | None) -> None:
        """Uniform multipliers in ``[1-f, 1+f]`` per env × axis for each PIDF term (``f`` from cfg)."""
        f = self.cfg.pid_gain_randomization_frac
        if f is None:
            return
        low, high = 1.0 - float(f), 1.0 + float(f)
        eids = self._env_ids_tensor(env_ids)
        k = int(eids.shape[0])
        dev, dt = self.device, torch.float32
        for t in (self._kp_rand, self._ki_rand, self._kd_rand, self._kf_rand):
            t[eids] = torch.empty((k, 3), device=dev, dtype=dt).uniform_(low, high)

    @staticmethod
    def _sample_axis_values(
        base: tuple[float, float, float],
        frac: float | tuple[float, float, float] | None,
        count: int,
        device: torch.device,
        min_value: float,
    ) -> torch.Tensor:
        base_t = torch.tensor(base, device=device, dtype=torch.float32).view(1, 3).expand(count, -1)
        if frac is None:
            return base_t.clamp_min(min_value)
        frac_t = torch.as_tensor(frac, device=device, dtype=torch.float32)
        if frac_t.ndim == 0:
            frac_t = frac_t.repeat(3)
        frac_t = frac_t.view(1, 3).expand(count, -1)
        noise = torch.empty((count, 3), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
        return (base_t * (1.0 + frac_t * noise)).clamp_min(min_value)

    def _resample_thrust_randomization(self, env_ids: Sequence[int] | None) -> None:
        eids = self._env_ids_tensor(env_ids)
        f = self.cfg.thrust_to_weight_randomization_frac
        if f is None or float(f) <= 0.0:
            self._thrust_to_weight[eids] = self._thrust_to_weight_base
            return
        low, high = 1.0 - float(f), 1.0 + float(f)
        multipliers = torch.empty(len(eids), device=self.device, dtype=torch.float32).uniform_(low, high)
        self._thrust_to_weight[eids] = self._thrust_to_weight_base * multipliers

    def _resample_aero_randomization(self, env_ids: Sequence[int] | None) -> None:
        eids = self._env_ids_tensor(env_ids)
        k = len(eids)
        if not self.cfg.aero_randomization_enable:
            self._linear_drag[eids] = 0.0
            self._quadratic_drag[eids] = 0.0
            self._angular_drag[eids] = 0.0
            return
        self._linear_drag[eids] = self._sample_axis_values(
            self.cfg.linear_drag_coeff,
            self.cfg.aero_randomization_frac,
            k,
            self.device,
            min_value=0.0,
        )
        self._quadratic_drag[eids] = self._sample_axis_values(
            self.cfg.quadratic_drag_coeff,
            self.cfg.aero_randomization_frac,
            k,
            self.device,
            min_value=0.0,
        )
        self._angular_drag[eids] = self._sample_axis_values(
            self.cfg.angular_drag_coeff,
            self.cfg.aero_randomization_frac,
            k,
            self.device,
            min_value=0.0,
        )

    def _compute_delayed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Shared action preprocessing path: clamp/noise/bias/LPF + delay buffer."""
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

            alpha_tensor = getattr(self._env, "_action_noise_lpf_alpha", None)
            if isinstance(alpha_tensor, torch.Tensor):
                alpha = alpha_tensor
            else:
                alpha = float(getattr(noise_cfg, "lpf_alpha", 1.0))
            if isinstance(alpha, torch.Tensor):
                cmd = alpha * cmd + (1.0 - alpha) * prev_cmd
            elif alpha < 1.0:
                cmd = alpha * cmd + (1.0 - alpha) * prev_cmd

            if getattr(noise_cfg, "clip_actions", True):
                cmd = cmd.clamp(-1.0, 1.0)

        self._raw_actions[:] = cmd
        self._delayed_actions[:] = self.actions_delay_buffer.compute(self._raw_actions)

        if hasattr(self._env, "_actions_buf"):
            self._env._actions_buf[:] = self._delayed_actions

        return self._delayed_actions

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
        ids = self._env_ids_tensor(env_ids)
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._delayed_actions[ids] = 0.0
        self._thrust[ids] = 0.0
        self._moment[ids] = 0.0
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
        self._resample_thrust_randomization(env_ids)
        self._resample_aero_randomization(env_ids)

    def process_actions(self, actions: torch.Tensor):
        delayed_actions = self._compute_delayed_actions(actions)
        thrust_cmd = (delayed_actions[:, 0] + 1.0) * 0.5
        # Betaflight SITL parity caps throttle (cfg default None = no cap).
        if self.cfg.thrust_command_limit is not None:
            thrust_cmd = thrust_cmd.clamp(max=float(self.cfg.thrust_command_limit))
        self._thrust.zero_()
        self._moment.zero_()
        self._thrust[:, 0, 2] = self._thrust_to_weight * self._robot_weight * thrust_cmd

        omega_meas = self._asset.data.root_ang_vel_b
        omega_des = delayed_actions[:, 1:] * self.cfg.rate_command_scale
        err = omega_des - omega_meas

        # process_actions runs once per policy/control step, not once per
        # physics substep, so PID integration and derivatives use control dt.
        dt = self._env.step_dt

        self._rate_integral = self._rate_integral + err * dt
        ilim = self.cfg.rate_integral_limit
        if ilim is not None and ilim > 0.0:
            self._rate_integral.clamp_(-ilim, ilim)

        # D on measured angular acceleration from physics (controller body frame):
        #   body_ang_acc_w (world) -> selected body frame via body_quat_w[:, body_id]
        alpha_w = self._asset.data.body_ang_acc_w[:, self._body_id, :]
        if alpha_w.ndim == 3:
            alpha_w = alpha_w.squeeze(1)
        body_quat_w = self._asset.data.body_quat_w[:, self._body_id]
        omega_dot_meas = quat_apply_inverse(body_quat_w, alpha_w)



        use_deriv = ~self._rate_pid_hold_derivative
        omega_des_dot = torch.zeros_like(omega_des)
        omega_des_dot[use_deriv] = (omega_des[use_deriv] - self._omega_des_prev[use_deriv]) / dt

        self._omega_des_prev[:] = omega_des
        self._rate_pid_hold_derivative[:] = False

        kp = self._kp_rand * self._kp_base
        ki = self._ki_rand * self._ki_base
        kd = self._kd_rand * self._kd_base
        kf = self._kf_rand * self._kf_base
        tau = kp * err + ki * self._rate_integral - kd * omega_dot_meas + kf * omega_des_dot


        tlim = self.cfg.torque_limit
        if tlim is not None and tlim > 0.0:
            tau = tau.clamp(-tlim, tlim)

        if self._sitl_enabled:
            motor_cmd = self._sitl_throttle_to_motor_output(thrust_cmd)
            base_motor_thrust = self._sitl_max_thrust_per_motor * motor_cmd.square()
            desired_wrench = torch.cat(
                ((base_motor_thrust * 4.0).unsqueeze(-1), tau),
                dim=-1,
            )
            motor_thrusts = desired_wrench @ self._sitl_allocation_pinv.T
            motor_thrusts = motor_thrusts.clamp(0.0, self._sitl_max_thrust_per_motor)
            realized_wrench = motor_thrusts @ self._sitl_allocation.T
            yaw_damping = -float(self.cfg.sitl_yaw_damping_coefficient) * omega_meas[:, 2]
            self._thrust[:, 0, 2] = realized_wrench[:, 0]
            self._moment[:, 0, 0] = realized_wrench[:, 1]
            self._moment[:, 0, 1] = realized_wrench[:, 2]
            self._moment[:, 0, 2] = realized_wrench[:, 3] + yaw_damping
            if hasattr(self._env, "_motor_speeds"):
                self._env._motor_speeds[:] = torch.sqrt(
                    motor_thrusts / self._sitl_max_thrust_per_motor
                )
        else:
            # self._thrust[:, 0, 2] was already set from thrust_cmd above.
            self._moment[:, 0, :] = tau
            if self.cfg.aero_randomization_enable:
                lin_vel_b = self._asset.data.root_lin_vel_b
                ang_vel_b = self._asset.data.root_ang_vel_b
                self._thrust[:, 0, :] += (
                    -self._linear_drag * lin_vel_b
                    - self._quadratic_drag * lin_vel_b.abs() * lin_vel_b
                )
                self._moment[:, 0, :] += -self._angular_drag * ang_vel_b
        self._processed_actions[:] = delayed_actions

        if not self._sitl_enabled and hasattr(self._env, "_mixer_b_inv") and hasattr(self._env, "_motor_speeds"):
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
    # Reserve capacity for progressive curricula while keeping the active delay at zero.
    delay_buffer_max_lag: int = 2
    body_name: str = "body"  # base_link for anemos (gets overridden by env cfg)

    #: Maps normalized rate sticks ``[-1, 1]`` to desired body angular rates [rad/s].
    rate_command_scale: float = 20.0 #This has to be defined based on the drone body rates limits

    #: ``direct`` keeps the original body thrust + direct body torque model.
    #: ``betaflight_sitl`` allocates the requested wrench through the SITL rotor geometry.
    actuator_model: str = "direct"
    thrust_command_limit: float | None = None

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
    thrust_to_weight_randomization_frac: float | None = None
    aero_randomization_enable: bool = False
    aero_randomization_frac: float | tuple[float, float, float] | None = None
    linear_drag_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quadratic_drag_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_drag_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: Clamp each axis of the integrated rate-error state (anti-windup helper).
    rate_integral_limit: float | None = 2.0
    #: Per-axis torque magnitude limit [N·m]; ``None`` disables.
    torque_limit: float | None = None

    sitl_mass_kg: float = _SITL_MASS_KG
    sitl_max_total_thrust: float = _SITL_MAX_TOTAL_THRUST_N
    sitl_yaw_torque_coefficient: float = _SITL_YAW_TORQUE_COEFFICIENT
    sitl_yaw_damping_coefficient: float = _SITL_YAW_DAMPING_COEFFICIENT
    sitl_rotor_positions_body: tuple[tuple[float, float, float], ...] = _SITL_ROTOR_POSITIONS_BODY
    sitl_rotor_directions: tuple[float, float, float, float] = _SITL_ROTOR_DIRECTIONS
    sitl_calibration_path: str | None = None


@configclass
class BetaflightSitlSurrogateActionCfg(QuadcopterRatesActionCfg):
    """SITL-parity rates action using Betaflight/Gazebo rotor semantics."""

    actuator_model: str = "betaflight_sitl"
    thrust_to_weight: float = 4.0
    rate_command_scale: float = 9.0
    thrust_command_limit: float | None = 0.6
    pterm_scale: float = 1.0
    iterm_scale: float = 0.0
    dterm_scale: float = 0.0
    feedforward_scale: float = 0.0
    pid_scales: float = 1.0
    pid_p: tuple[float, float, float] = (0.281, 0.281, 0.281)
    pid_i: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pid_d: tuple[float, float, float] = (0.04, 0.04, 0.04)
    pid_f: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_limit: float | None = 2.0

class QuadcopterSecondOrderRatesAction(QuadcopterRatesAction):
    """Apply thrust as a body force and impose second-order body-rate tracking directly.

    This action matches simulators that accept collective thrust plus body-rate commands
    but hide their inner controller. The rotational dynamics are intentionally kinematic:
    the second-order model produces the actual body-rate trajectory, which is written to
    the root angular velocity each physics substep. No PID torque is applied.
    """

    cfg: QuadcopterSecondOrderRatesActionCfg

    def __init__(self, cfg: QuadcopterSecondOrderRatesActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._omega_target = torch.zeros(env.num_envs, 3, device=self.device)
        self._omega_cmd_filter = torch.zeros(env.num_envs, 3, device=self.device)
        self._omega_cmd_filter_dot = torch.zeros(env.num_envs, 3, device=self.device)
        self._env_ids_all = torch.arange(env.num_envs, device=self.device, dtype=torch.long)
        self._rate_filter_gain = torch.ones(env.num_envs, 3, device=self.device, dtype=torch.float32)
        self._rate_filter_wn = torch.ones(env.num_envs, 3, device=self.device, dtype=torch.float32)
        self._rate_filter_zeta = torch.ones(env.num_envs, 3, device=self.device, dtype=torch.float32)
        self._resample_rate_filter_randomization(None)
        self._probe_env_ids = torch.tensor(
            [eid for eid in cfg.probe_env_ids if 0 <= int(eid) < env.num_envs],
            device=self.device,
            dtype=torch.long,
        )
        self._probe_enabled = bool(cfg.probe_enable) and self._probe_env_ids.numel() > 0
        self._probe_action = torch.tensor(cfg.probe_action, device=self.device, dtype=torch.float32)
        self._probe_root_velocity_w = torch.tensor(
            (*cfg.probe_root_lin_vel_w, *cfg.probe_root_ang_vel_w), device=self.device, dtype=torch.float32
        )
        if cfg.probe_enable and not self._probe_enabled:
            print(
                "[SECOND_ORDER_RATES_PROBE] enabled but no valid env ids were found "
                f"(num_envs={env.num_envs}, requested={list(cfg.probe_env_ids)})."
            )

    def _resample_rate_filter_randomization(self, env_ids: Sequence[int] | None) -> None:
        eids = self._env_ids_tensor(env_ids)
        k = len(eids)
        if not self.cfg.rate_filter_enable:
            self._rate_filter_gain[eids] = 1.0
            self._rate_filter_wn[eids] = 1.0
            self._rate_filter_zeta[eids] = 1.0
            return

        freq_hz = self._sample_axis_values(
            self.cfg.rate_filter_natural_frequency_hz,
            self.cfg.rate_filter_natural_frequency_randomization_frac,
            k,
            self.device,
            min_value=0.1,
        )
        self._rate_filter_wn[eids] = 2.0 * torch.pi * freq_hz
        self._rate_filter_zeta[eids] = self._sample_axis_values(
            self.cfg.rate_filter_damping_ratio,
            self.cfg.rate_filter_damping_randomization_frac,
            k,
            self.device,
            min_value=0.05,
        )
        self._rate_filter_gain[eids] = self._sample_axis_values(
            self.cfg.rate_filter_command_gain,
            self.cfg.rate_filter_command_gain_randomization_frac,
            k,
            self.device,
            min_value=0.05,
        )

    def _filter_rate_command(self, omega_des: torch.Tensor, substeps: int | None = None) -> torch.Tensor:
        if not self.cfg.rate_filter_enable:
            return omega_des
        dt = float(getattr(self._env, "physics_dt", self._env.cfg.sim.dt))
        substeps = max(1, int(getattr(self._env.cfg, "decimation", 1)) if substeps is None else int(substeps))
        target = self._rate_filter_gain * omega_des
        max_rate = float(self.cfg.rate_filter_max_command_rad_s)
        for _ in range(substeps):
            acc = (
                self._rate_filter_wn.square() * (target - self._omega_cmd_filter)
                - 2.0 * self._rate_filter_zeta * self._rate_filter_wn * self._omega_cmd_filter_dot
            )
            self._omega_cmd_filter_dot += acc * dt
            self._omega_cmd_filter += self._omega_cmd_filter_dot * dt
        if max_rate > 0.0:
            self._omega_cmd_filter.clamp_(-max_rate, max_rate)
        return self._omega_cmd_filter

    def _probe_window_active(self, sim_step: int) -> bool:
        return self.cfg.probe_start_step <= sim_step < (self.cfg.probe_start_step + self.cfg.probe_num_steps)

    def _apply_probe_action_override(self, delayed_actions: torch.Tensor, sim_step: int) -> torch.Tensor:
        if not self._probe_enabled or not self._probe_window_active(sim_step):
            return delayed_actions
        delayed_actions = delayed_actions.clone()
        delayed_actions[self._probe_env_ids] = self._probe_action
        self._delayed_actions[:] = delayed_actions
        if hasattr(self._env, "_actions_buf"):
            self._env._actions_buf[:] = delayed_actions
        return delayed_actions

    def _apply_probe_state_override(self, sim_step: int) -> None:
        if not self._probe_enabled:
            return
        is_start = sim_step == self.cfg.probe_start_step
        is_forced = self.cfg.probe_force_state_each_step and self._probe_window_active(sim_step)
        if not (is_start or is_forced):
            return
        vels = self._probe_root_velocity_w.unsqueeze(0).expand(self._probe_env_ids.numel(), -1)
        self._asset.write_root_velocity_to_sim(vels, env_ids=self._probe_env_ids)

    def _log_probe_status(self, sim_step: int, omega_b_set: torch.Tensor, force: torch.Tensor, moment: torch.Tensor) -> None:
        if not self._probe_enabled or not self._probe_window_active(sim_step):
            return
        if self.cfg.probe_log_interval <= 0 or sim_step % self.cfg.probe_log_interval != 0:
            return
        eids = self._probe_env_ids
        lin_w = self._asset.data.root_lin_vel_w[eids]
        ang_b = self._asset.data.root_ang_vel_b[eids]
        omega_target = self._omega_target[eids]
        omega_set = omega_b_set[eids]
        probe_force = force[eids, 0, :]
        probe_moment = moment[eids, 0, :]
        print(
            "[SECOND_ORDER_RATES_PROBE] "
            f"step={sim_step} envs={eids.tolist()} "
            f"lin_w_mean={lin_w.mean(dim=0).tolist()} "
            f"ang_b_mean={ang_b.mean(dim=0).tolist()} "
            f"omega_target_mean={omega_target.mean(dim=0).tolist()} "
            f"omega_set_mean={omega_set.mean(dim=0).tolist()} "
            f"force_mean={probe_force.mean(dim=0).tolist()} "
            f"moment_mean={probe_moment.mean(dim=0).tolist()}"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            self._omega_cmd_filter.zero_()
            self._omega_cmd_filter_dot.zero_()
        else:
            eids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._omega_cmd_filter[eids] = 0.0
            self._omega_cmd_filter_dot[eids] = 0.0
        self._resample_thrust_randomization(env_ids)
        self._resample_rate_filter_randomization(env_ids)
        self._resample_aero_randomization(env_ids)

    def process_actions(self, actions: torch.Tensor):
        sim_step = int(getattr(self._env, "_sim_step_counter", 0))
        delayed_actions = self._compute_delayed_actions(actions)
        delayed_actions = self._apply_probe_action_override(delayed_actions, sim_step)
        thrust_cmd = (delayed_actions[:, 0] + 1.0) * 0.5
        self._thrust.zero_()
        self._moment.zero_()
        self._thrust[:, 0, 2] = self._thrust_to_weight * self._robot_weight * thrust_cmd
        self._omega_target[:] = delayed_actions[:, 1:] * self.cfg.rate_command_scale
        self._processed_actions[:] = delayed_actions

        if hasattr(self._env, "_motor_speeds"):
            omega_max = float(getattr(self._env.cfg, "motor_omega_max_rad_s", 1.0))
            collective_speed = torch.sqrt(torch.clamp(thrust_cmd, min=0.0)).unsqueeze(-1) * omega_max
            self._env._motor_speeds[:] = collective_speed.expand(-1, 4)

    def apply_actions(self):
        sim_step = int(getattr(self._env, "_sim_step_counter", 0))
        self._apply_probe_state_override(sim_step)
        omega_b = self._filter_rate_command(self._omega_target, substeps=1)
        lin_vel_w = self._asset.data.root_lin_vel_w
        omega_w = quat_apply(self._asset.data.root_quat_w, omega_b)
        root_vel = torch.cat([lin_vel_w, omega_w], dim=-1)
        self._asset.write_root_velocity_to_sim(root_vel, self._env_ids_all)

        force = self._thrust.clone()
        moment = torch.zeros_like(self._moment)
        if self.cfg.aero_randomization_enable:
            lin_vel_b = self._asset.data.root_lin_vel_b
            force[:, 0, :] += (
                -self._linear_drag * lin_vel_b
                - self._quadratic_drag * lin_vel_b.abs() * lin_vel_b
            )
            moment[:, 0, :] += -self._angular_drag * self._asset.data.root_ang_vel_b

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force, torques=moment, body_ids=self._body_id
        )
        self._log_probe_status(sim_step, omega_b_set=omega_b, force=force, moment=moment)

        if self.cfg.velocity_force_diagnostic:
            interval = max(1, int(self.cfg.velocity_force_diagnostic_interval_steps))
            if int(getattr(self._env, "_sim_step_counter", 0)) % interval == 0:
                print(
                    "[SECOND_ORDER_RATES] "
                    f"thrust_z_mean={force[:, 0, 2].mean().item():.3f}N "
                    f"omega_cmd_mean={self._omega_target.norm(dim=1).mean().item():.3f}rad/s "
                    f"omega_set_mean={omega_b.norm(dim=1).mean().item():.3f}rad/s "
                    f"lin_vel_z_mean={lin_vel_w[:, 2].mean().item():.3f}m/s"
                )


@configclass
class QuadcopterSecondOrderRatesActionCfg(QuadcopterRatesActionCfg):
    """Configuration for :class:`QuadcopterSecondOrderRatesAction`."""

    class_type: type[ActionTerm] = QuadcopterSecondOrderRatesAction
    thrust_to_weight_randomization_frac: float | None = None
    rate_filter_enable: bool = False
    rate_filter_natural_frequency_hz: tuple[float, float, float] = (12.0, 12.0, 8.0)
    rate_filter_natural_frequency_randomization_frac: float | tuple[float, float, float] | None = None
    rate_filter_damping_ratio: tuple[float, float, float] = (0.8, 0.8, 0.9)
    rate_filter_damping_randomization_frac: float | tuple[float, float, float] | None = None
    rate_filter_command_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rate_filter_command_gain_randomization_frac: float | tuple[float, float, float] | None = None
    rate_filter_max_command_rad_s: float = 0.0
    aero_randomization_enable: bool = False
    aero_randomization_frac: float | tuple[float, float, float] | None = None
    linear_drag_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quadratic_drag_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_drag_coeff: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Temporary probe mode: override selected envs for in-training behavior checks.
    probe_enable: bool = False
    probe_env_ids: tuple[int, ...] = (0, 1, 2, 3)
    probe_start_step: int = 500
    probe_num_steps: int = 400
    probe_action: tuple[float, float, float, float] = (0.2, 0.8, 0.0, 0.0)
    probe_root_lin_vel_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    probe_root_ang_vel_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    probe_force_state_each_step: bool = False
    probe_log_interval: int = 10
    velocity_force_diagnostic: bool = False
    velocity_force_diagnostic_interval_steps: int = 240

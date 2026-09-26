# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drone-profile container shared by every quadcopter task.

Only the *schema* lives here. Each task (racing, competition, hovering) defines its own
profile values in its own cfg module, so retuning one task never changes another.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.assets import ArticulationCfg


@dataclass
class DroneProfile:
    """Bundle a robot config with the physics parameters tuned for it."""

    robot_cfg: ArticulationCfg
    body_name: str
    thrust_to_weight: float
    moment_scale: tuple[float, float, float]  #: Per-axis (roll, pitch, yaw) moment scales [N·m] mapping action ±1 → full physical authority.
    motor_k_t: float #: Thrust coefficient :math:`k_T` [N/(rad/s)²] for :math:`T_i = k_T \Omega_i^2`.
    motor_k_d: float #: Drag torque coefficient :math:`k_D` [N·m/(rad/s)²] for yaw mixing row in :math:`B`.
    motor_arm_length_m: float #: Diagonal arm length :math:`l` [m] in the X-frame mixer :math:`B` (center to rotor).
    #: Nominal max rotor speed [rad/s] for scaling motor-speed observations (see ``motor_speeds_obs``).
    motor_omega_max_rad_s: float
    rate_command_scale: float
    pid_scales: float
    pid_p: tuple[float, float, float]
    pid_i: tuple[float, float, float]
    pid_d: tuple[float, float, float]
    pid_f: tuple[float, float, float]


def make_drone_profile(
    cfg: ArticulationCfg,
    body_name: str,
    thrust_to_weight: float,
    moment_scale: tuple[float, float, float],
    *,
    motor_k_t: float,
    motor_k_d: float,
    motor_arm_length_m: float,
    motor_omega_max_rad_s: float,
    rate_command_scale: float,
    pid_scales: float,
    pid_p: tuple[float, float, float],
    pid_i: tuple[float, float, float],
    pid_d: tuple[float, float, float],
    pid_f: tuple[float, float, float],
) -> DroneProfile:
    """Build a profile; the robot is placed under ``{ENV_REGEX_NS}/Robot`` with contact sensors on."""
    return DroneProfile(
        robot_cfg=cfg.replace(prim_path="{ENV_REGEX_NS}/Robot").replace(
            spawn=cfg.spawn.replace(activate_contact_sensors=True)
        ),
        body_name=body_name,
        thrust_to_weight=thrust_to_weight,
        moment_scale=moment_scale,
        motor_k_t=motor_k_t,
        motor_k_d=motor_k_d,
        motor_arm_length_m=motor_arm_length_m,
        motor_omega_max_rad_s=motor_omega_max_rad_s,
        rate_command_scale=rate_command_scale,
        pid_scales=pid_scales,
        pid_p=pid_p,
        pid_i=pid_i,
        pid_d=pid_d,
        pid_f=pid_f,
    )

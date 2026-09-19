"""Hovering curriculum policy; every value is local to hovering."""
from __future__ import annotations
from dataclasses import dataclass
from Racing_Quadcopter.tasks.manager_based.adaptive_curriculums import (
    CurriculumTarget,
    register_randomization_profile,
)

@dataclass(frozen=True)
class HoveringRandomizationProfile:
    action_gaussian_std: float = 0.02
    action_bias_range: tuple[float, float] = (-0.05, 0.05)
    action_lpf_start_range: tuple[float, float] = (1.0, 1.0)
    action_lpf_range: tuple[float, float] = (0.8, 1.0)
    base_ang_vel_noise: tuple[float, float] = (0.05, 0.01)
    gravity_noise: tuple[float, float] = (0.05, 0.01)
    max_delay: int = 2
    pid_gain_randomization_frac: float = 0.10
    thrust_to_weight_randomization_frac: float = 0.05
    aero_randomization_frac: float = 0.10
    linear_drag_coeff: tuple[float, float, float] = (0.08, 0.08, 0.12)
    quadratic_drag_coeff: tuple[float, float, float] = (0.01, 0.01, 0.018)
    angular_drag_coeff: tuple[float, float, float] = (0.002, 0.002, 0.0012)
    # Hover acceptance gate: begin permissive so route progression is learned,
    # then tighten jointly with domain randomization.
    collection_max_linear_speed_start_m_s: float = 1.50
    collection_max_linear_speed_m_s: float = 0.30
    collection_max_angular_speed_start_rad_s: float = 4.00
    collection_max_angular_speed_rad_s: float = 1.00
    collection_max_tilt_start_sin: float = 0.80
    collection_max_tilt_sin: float = 0.30

HOVERING_RANDOMIZATION = HoveringRandomizationProfile()

def hovering_randomization_targets() -> tuple[CurriculumTarget, ...]:
    """Explicit target list for hovering's standard-rate action."""
    p = HOVERING_RANDOMIZATION
    def env(name, path, start, target): return CurriculumTarget(name, "env_cfg", path, "lerp", target, start)
    def action(name, path, start, target, *, dtype=None):
        return CurriculumTarget(name, "action_term_cfg", path, "lerp", target, start, term_name="rates", dtype=dtype, clamp_min=0 if name == "rates_max_delay" else None)
    def obs(name, term, path, start, target): return CurriculumTarget(name, "observation_term_cfg", path, "lerp", target, start, term_name=term, group_name="policy")
    zero3 = (0.0, 0.0, 0.0)
    return (
        env("action_noise_enable", "action_noise.enable", False, True),
        env("action_noise_gaussian_std", "action_noise.gaussian_std", 0.0, p.action_gaussian_std),
        env("action_noise_bias_range", "action_noise.bias_range", (0.0, 0.0), p.action_bias_range),
        env("action_noise_lpf_alpha_range", "action_noise.lpf_alpha_range", p.action_lpf_start_range, p.action_lpf_range),
        action("rates_max_delay", "max_delay", 0, p.max_delay, dtype="int"),
        action("rates_pid_gain_rand_frac", "pid_gain_randomization_frac", 0.0, p.pid_gain_randomization_frac),
        action("rates_thrust_to_weight_rand_frac", "thrust_to_weight_randomization_frac", 0.0, p.thrust_to_weight_randomization_frac),
        action("rates_aero_randomization_enable", "aero_randomization_enable", False, True),
        action("rates_aero_randomization_frac", "aero_randomization_frac", 0.0, p.aero_randomization_frac),
        action("rates_linear_drag_coeff", "linear_drag_coeff", zero3, p.linear_drag_coeff),
        action("rates_quadratic_drag_coeff", "quadratic_drag_coeff", zero3, p.quadratic_drag_coeff),
        action("rates_angular_drag_coeff", "angular_drag_coeff", zero3, p.angular_drag_coeff),
        obs("obs_base_ang_vel_enable", "base_ang_vel", "noise.enable", False, True),
        obs("obs_base_ang_vel_std", "base_ang_vel", "noise.noise_cfg.std", 0.0, p.base_ang_vel_noise[0]),
        obs("obs_base_ang_vel_bias_std", "base_ang_vel", "noise.bias_noise_cfg.std", 0.0, p.base_ang_vel_noise[1]),
        obs("obs_gravity_enable", "projected_gravity", "noise.enable", False, True),
        obs("obs_gravity_std", "projected_gravity", "noise.noise_cfg.std", 0.0, p.gravity_noise[0]),
        obs("obs_gravity_bias_std", "projected_gravity", "noise.bias_noise_cfg.std", 0.0, p.gravity_noise[1]),
        env(
            "collection_max_linear_speed_m_s",
            "collection_max_linear_speed_m_s",
            p.collection_max_linear_speed_start_m_s,
            p.collection_max_linear_speed_m_s,
        ),
        env(
            "collection_max_angular_speed_rad_s",
            "collection_max_angular_speed_rad_s",
            p.collection_max_angular_speed_start_rad_s,
            p.collection_max_angular_speed_rad_s,
        ),
        env(
            "collection_max_tilt_sin",
            "collection_max_tilt_sin",
            p.collection_max_tilt_start_sin,
            p.collection_max_tilt_sin,
        ),
    )


def register_hovering_randomization_profile() -> None:
    """Register hovering's local values and target declarations."""
    register_randomization_profile(
        "hovering", HOVERING_RANDOMIZATION, hovering_randomization_targets
    )

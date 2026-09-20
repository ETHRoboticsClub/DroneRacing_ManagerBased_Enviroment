"""Competition randomization profile and shared-curriculum bindings."""

from __future__ import annotations

from dataclasses import dataclass

from Racing_Quadcopter.tasks.manager_based.adaptive_curriculums import (
    CurriculumTarget,
    register_randomization_profile,
)


@dataclass(frozen=True)
class CompetitionRandomizationProfile:
    """Single source of truth for competition curriculum maxima."""

    action_gaussian_std: float = 0.02
    action_bias_range: tuple[float, float] = (-0.05, 0.05)
    action_lpf_start_range: tuple[float, float] = (0.9, 1.0)
    action_lpf_range: tuple[float, float] = (0.8, 1.0)
    max_delay: int = 2
    filter_frequency_fraction: tuple[float, float, float] = (0.3, 0.3, 0.3)
    filter_damping_fraction: tuple[float, float, float] = (0.3, 0.3, 0.3)
    filter_gain_fraction: tuple[float, float, float] = (0.2, 0.2, 0.2)
    thrust_fraction: float = 0.05
    aero_fraction: float = 0.10
    linear_drag: tuple[float, float, float] = (0.08, 0.08, 0.12)
    quadratic_drag: tuple[float, float, float] = (0.010, 0.010, 0.018)
    angular_drag: tuple[float, float, float] = (0.0020, 0.0020, 0.0012)
    linear_velocity_noise: tuple[float, float] = (0.2, 1.0)
    angular_velocity_noise: tuple[float, float] = (0.05, 0.2)
    gravity_noise: tuple[float, float] = (0.02, 0.04)


COMPETITION_RANDOMIZATION = CompetitionRandomizationProfile()


def competition_randomization_targets() -> tuple[CurriculumTarget, ...]:
    """Build typed bindings from :data:`COMPETITION_RANDOMIZATION`."""

    def env(name: str, path: str, start: object, target: object) -> CurriculumTarget:
        return CurriculumTarget(name, "env_cfg", path, "lerp", target, start)

    def action(
        name: str,
        path: str,
        start: object,
        target: object,
        *,
        dtype: str | None = None,
        clamp_min: float | None = None,
    ) -> CurriculumTarget:
        return CurriculumTarget(
            name, "action_term_cfg", path, "lerp", target, start,
            term_name="rates", dtype=dtype, clamp_min=clamp_min,
        )

    def obs(name: str, term: str, path: str, start: object, target: object) -> CurriculumTarget:
        return CurriculumTarget(
            name, "observation_term_cfg", path, "lerp", target, start,
            term_name=term, group_name="policy",
        )

    profile = COMPETITION_RANDOMIZATION
    zero3 = (0.0, 0.0, 0.0)
    return (
        env("action_noise_enable", "action_noise.enable", False, True),
        env("action_noise_gaussian_std", "action_noise.gaussian_std", 0.0, profile.action_gaussian_std),
        env("action_noise_bias_range", "action_noise.bias_range", (0.0, 0.0), profile.action_bias_range),
        env("action_noise_lpf_alpha_range", "action_noise.lpf_alpha_range", profile.action_lpf_start_range, profile.action_lpf_range),
        action("rates_max_delay", "max_delay", 0, profile.max_delay, dtype="int", clamp_min=0),
        action("rates_filter_wn_rand_frac", "rate_filter_natural_frequency_randomization_frac", zero3, profile.filter_frequency_fraction),
        action("rates_filter_zeta_rand_frac", "rate_filter_damping_randomization_frac", zero3, profile.filter_damping_fraction),
        action("rates_filter_gain_rand_frac", "rate_filter_command_gain_randomization_frac", zero3, profile.filter_gain_fraction),
        action("rates_thrust_to_weight_rand_frac", "thrust_to_weight_randomization_frac", 0.0, profile.thrust_fraction),
        action("rates_aero_randomization_frac", "aero_randomization_frac", 0.0, profile.aero_fraction),
        action("rates_linear_drag_coeff", "linear_drag_coeff", zero3, profile.linear_drag),
        action("rates_quadratic_drag_coeff", "quadratic_drag_coeff", zero3, profile.quadratic_drag),
        action("rates_angular_drag_coeff", "angular_drag_coeff", zero3, profile.angular_drag),
        obs("obs_base_lin_vel_enable", "base_lin_vel", "noise.enable", False, True),
        obs("obs_base_ang_vel_enable", "base_ang_vel", "noise.enable", False, True),
        obs("obs_gravity_enable", "projected_gravity", "noise.enable", False, True),
        obs("obs_base_lin_vel_std", "base_lin_vel", "noise.noise_cfg.std", 0.0, profile.linear_velocity_noise[0]),
        obs("obs_base_lin_vel_bias_std", "base_lin_vel", "noise.bias_noise_cfg.std", 0.0, profile.linear_velocity_noise[1]),
        obs("obs_base_ang_vel_std", "base_ang_vel", "noise.noise_cfg.std", 0.0, profile.angular_velocity_noise[0]),
        obs("obs_base_ang_vel_bias_std", "base_ang_vel", "noise.bias_noise_cfg.std", 0.0, profile.angular_velocity_noise[1]),
        obs("obs_gravity_std", "projected_gravity", "noise.noise_cfg.std", 0.0, profile.gravity_noise[0]),
        obs("obs_gravity_bias_std", "projected_gravity", "noise.bias_noise_cfg.std", 0.0, profile.gravity_noise[1]),
    )


def register_competition_randomization_profile() -> None:
    """Register this module's values and targets without exposing either globally."""
    register_randomization_profile(
        "competition", COMPETITION_RANDOMIZATION, competition_randomization_targets
    )

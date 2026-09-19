"""Base-racing curriculum policy; every value is local to base racing."""
from __future__ import annotations
from dataclasses import dataclass
from Racing_Quadcopter.tasks.manager_based.adaptive_curriculums import (
    CurriculumTarget,
    register_randomization_profile,
)

@dataclass(frozen=True)
class RacingRandomizationProfile:
    action_gaussian_std: float = 0.02
    action_bias_range: tuple[float, float] = (-0.05, 0.05)
    action_lpf_start_range: tuple[float, float] = (0.9, 1.0)
    action_lpf_range: tuple[float, float] = (0.8, 1.0)
    # Match competition observation corruption. Dynamics remain deliberately
    # milder for the base PID-rate action than competition's second-order one.
    base_lin_vel_noise: tuple[float, float] = (0.2, 1.0)
    base_ang_vel_noise: tuple[float, float] = (0.05, 0.2)
    gravity_noise: tuple[float, float] = (0.02, 0.04)
    max_delay: int = 1
    pid_gain_randomization_frac: float = 0.05
    thrust_to_weight_randomization_frac: float = 0.03
    aero_randomization_frac: float = 0.05
    linear_drag_coeff: tuple[float, float, float] = (0.04, 0.04, 0.06)
    quadratic_drag_coeff: tuple[float, float, float] = (0.005, 0.005, 0.009)
    angular_drag_coeff: tuple[float, float, float] = (0.001, 0.001, 0.0006)

RACING_RANDOMIZATION = RacingRandomizationProfile()

def racing_randomization_targets(*, observation_terms: tuple[str, ...], action_type: str) -> tuple[CurriculumTarget, ...]:
    """Explicit target list for fields which exist in the selected racing action."""
    p = RACING_RANDOMIZATION
    def env(name, path, start, target): return CurriculumTarget(name, "env_cfg", path, "lerp", target, start)
    def action(name, path, start, target, *, dtype=None):
        return CurriculumTarget(name, "action_term_cfg", path, "lerp", target, start, term_name="rates", dtype=dtype, clamp_min=0 if name == "rates_max_delay" else None)
    obs_values = {"base_lin_vel": p.base_lin_vel_noise, "base_ang_vel": p.base_ang_vel_noise, "projected_gravity": p.gravity_noise}
    targets = [
        env("action_noise_enable", "action_noise.enable", False, True),
        env("action_noise_gaussian_std", "action_noise.gaussian_std", 0.0, p.action_gaussian_std),
        env("action_noise_bias_range", "action_noise.bias_range", (0.0, 0.0), p.action_bias_range),
        env("action_noise_lpf_alpha_range", "action_noise.lpf_alpha_range", p.action_lpf_start_range, p.action_lpf_range),
    ]
    for term in observation_terms:
        std, bias = obs_values[term]
        targets += [
            CurriculumTarget(f"obs_{term}_enable", "observation_term_cfg", "noise.enable", "lerp", True, False, term, "policy"),
            CurriculumTarget(f"obs_{term}_std", "observation_term_cfg", "noise.noise_cfg.std", "lerp", std, 0.0, term, "policy"),
            CurriculumTarget(f"obs_{term}_bias_std", "observation_term_cfg", "noise.bias_noise_cfg.std", "lerp", bias, 0.0, term, "policy"),
        ]
    if action_type == "moments":
        targets.append(CurriculumTarget("moments_max_delay", "action_term_cfg", "max_delay", "lerp", p.max_delay, 0, term_name="moments", dtype="int", clamp_min=0))
        return tuple(targets)
    if action_type != "rates": raise ValueError(f"Unsupported base-racing action type {action_type!r}.")
    zero3 = (0.0, 0.0, 0.0)
    targets += [
        action("rates_max_delay", "max_delay", 0, p.max_delay, dtype="int"),
        action("rates_pid_gain_rand_frac", "pid_gain_randomization_frac", 0.0, p.pid_gain_randomization_frac),
        action("rates_thrust_to_weight_rand_frac", "thrust_to_weight_randomization_frac", 0.0, p.thrust_to_weight_randomization_frac),
        action("rates_aero_randomization_enable", "aero_randomization_enable", False, True),
        action("rates_aero_randomization_frac", "aero_randomization_frac", 0.0, p.aero_randomization_frac),
        action("rates_linear_drag_coeff", "linear_drag_coeff", zero3, p.linear_drag_coeff),
        action("rates_quadratic_drag_coeff", "quadratic_drag_coeff", zero3, p.quadratic_drag_coeff),
        action("rates_angular_drag_coeff", "angular_drag_coeff", zero3, p.angular_drag_coeff),
    ]
    return tuple(targets)


def register_racing_randomization_profile(
    *, observation_terms: tuple[str, ...], action_type: str
) -> None:
    """Register base-racing's local profile and its selected action/obs bindings."""
    register_randomization_profile(
        "racing",
        RACING_RANDOMIZATION,
        lambda: racing_randomization_targets(
            observation_terms=observation_terms, action_type=action_type
        ),
    )

"""Environment-independent action-noise and episode speed-limit features.

Environment classes only allocate these features and expose manager terms.  All
validation, reset sampling, observations, penalties, and curriculum activation
live here so racing and waypoint tasks share identical semantics.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


@configclass
class ActionNoiseCfg:
    """Action corruption shared by every quadcopter task."""

    enable: bool = False
    per_episode_bias: bool = True
    gaussian_std: float = 0.0
    bias_range: tuple[float, float] = (0.0, 0.0)
    lpf_alpha: float = 1.0
    per_episode_lpf_alpha: bool = True
    lpf_alpha_range: tuple[float, float] = (1.0, 1.0)
    clip_actions: bool = True


@configclass
class SpeedLimitsCfg:
    """Optional episode-conditioned linear and angular speed limits.

    Disabled defaults deliberately use zero ranges.  Enabling the feature
    requires strictly positive ``(minimum, maximum)`` ranges.
    """

    enabled: bool = False
    linear_range: tuple[float, float] = (0.0, 0.0)
    angular_range: tuple[float, float] = (0.0, 0.0)


def penalty_action_magnitude(
    env: ManagerBasedRLEnv,
    clip_threshold: float = 1.0,
    clip_excess_multiplier: float = 100.0,
) -> torch.Tensor:
    """Penalize raw policy commands, strongly discouraging clamp saturation.

    The action manager clips policy actions to ``[-1, 1]`` before they reach
    the controller. Penalizing only that clipped command cannot distinguish a
    valid boundary command from an arbitrarily large raw request. This shared
    term makes the latter costly for every task using the common action path.
    """
    raw_actions = env._policy_raw_actions
    penalty = raw_actions.square().sum(dim=1)
    exceeds_clip = raw_actions.abs().max(dim=1).values > float(clip_threshold)
    multiplier = torch.where(
        exceeds_clip,
        torch.full_like(penalty, float(clip_excess_multiplier)),
        torch.ones_like(penalty),
    )
    return penalty * multiplier


def validate_shared_feature_events(cfg: object) -> None:
    """Reject active shared features whose required reset sampler is absent."""
    events = getattr(cfg, "events", None)
    required: list[str] = []
    action_noise = getattr(cfg, "action_noise", None)
    if action_noise is not None and (
        bool(getattr(action_noise, "enable", False))
        or bool(getattr(cfg, "adaptive_curriculum_enabled", False))
    ):
        required.append("reset_action_noise")
    speed_limits = getattr(cfg, "speed_limits", None)
    if speed_limits is not None and bool(speed_limits.enabled):
        required.append("randomize_speed_limits")

    missing = [name for name in required if events is None or getattr(events, name, None) is None]
    if missing:
        raise ValueError(
            "Active shared features require reset events: missing "
            + ", ".join(f"events.{name}" for name in missing)
            + "."
        )


def validate_speed_limits_cfg(cfg: SpeedLimitsCfg) -> None:
    """Validate enabled ranges while accepting zero-valued disabled defaults."""
    for name, value_range in (
        ("linear_range", cfg.linear_range),
        ("angular_range", cfg.angular_range),
    ):
        if len(value_range) != 2:
            raise ValueError(f"speed_limits.{name} must contain exactly (min, max).")
        low, high = (float(value_range[0]), float(value_range[1]))
        if cfg.enabled and (low <= 0.0 or high < low):
            raise ValueError(
                f"Enabled speed_limits.{name} must satisfy 0 < min <= max; got {value_range}."
            )
        if not cfg.enabled and (low < 0.0 or high < low):
            raise ValueError(
                f"Disabled speed_limits.{name} must satisfy 0 <= min <= max; got {value_range}."
            )


def initialize_action_noise(env: ManagerBasedRLEnv, action_dim: int) -> None:
    """Allocate action-corruption state before the action manager is built."""
    env._action_noise_bias = torch.zeros(env.num_envs, action_dim, device=env.device)
    env._action_noise_lpf_alpha = torch.ones(env.num_envs, 1, device=env.device)


def reset_action_noise(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Sample per-episode action bias and LPF state for selected environments."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
    count = env_ids.numel()
    if count == 0:
        return

    cfg = getattr(env.cfg, "action_noise", None)
    if cfg is not None and cfg.enable and cfg.per_episode_bias:
        low, high = (float(v) for v in cfg.bias_range)
        env._action_noise_bias[env_ids] = (
            torch.rand(count, env._action_noise_bias.shape[1], device=env.device)
            * (high - low)
            + low
        )
    else:
        env._action_noise_bias[env_ids] = 0.0

    if cfg is not None and cfg.enable and cfg.per_episode_lpf_alpha:
        low, high = sorted(float(v) for v in cfg.lpf_alpha_range)
        low = max(0.0, min(1.0, low))
        high = max(0.0, min(1.0, high))
        env._action_noise_lpf_alpha[env_ids] = (
            torch.rand(count, 1, device=env.device) * (high - low) + low
        )
    else:
        alpha = float(cfg.lpf_alpha) if cfg is not None else 1.0
        env._action_noise_lpf_alpha[env_ids] = max(0.0, min(1.0, alpha))


def initialize_speed_limits(env: ManagerBasedRLEnv) -> None:
    """Allocate optional speed-limit state with disabled-as-zero semantics."""
    cfg = env.cfg.speed_limits
    validate_speed_limits_cfg(cfg)
    linear = float(cfg.linear_range[1]) if cfg.enabled else 0.0
    angular = float(cfg.angular_range[1]) if cfg.enabled else 0.0
    env._velocity_limit = torch.full((env.num_envs,), linear, device=env.device)
    env._angular_velocity_limit = torch.full((env.num_envs,), angular, device=env.device)
    # Evaluation samples immediately; training waits for an explicit curriculum latch.
    env._speed_limit_randomization_active = bool(
        cfg.enabled and getattr(env.cfg, "eval_mode", False)
    )


def activate_speed_limit_randomization(env: ManagerBasedRLEnv) -> None:
    """Permanently latch reset-time speed-limit sampling on for an environment."""
    if not env.cfg.speed_limits.enabled or env._speed_limit_randomization_active:
        return
    env._speed_limit_randomization_active = True
    print(
        "[Curriculum] Speed-limit randomization activated permanently "
        f"at step {env.common_step_counter}"
    )


def randomize_speed_limits(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Sample paired episode limits, or keep fixed maxima before activation."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
    if env_ids.numel() == 0:
        return
    cfg = env.cfg.speed_limits
    if not cfg.enabled:
        env._velocity_limit[env_ids] = 0.0
        env._angular_velocity_limit[env_ids] = 0.0
        return

    linear_low, linear_high = (float(v) for v in cfg.linear_range)
    angular_low, angular_high = (float(v) for v in cfg.angular_range)
    if env._speed_limit_randomization_active:
        count = env_ids.numel()
        env._velocity_limit[env_ids] = (
            torch.rand(count, device=env.device) * (linear_high - linear_low) + linear_low
        )
        env._angular_velocity_limit[env_ids] = (
            torch.rand(count, device=env.device) * (angular_high - angular_low) + angular_low
        )
    else:
        env._velocity_limit[env_ids] = linear_high
        env._angular_velocity_limit[env_ids] = angular_high


def speed_limits_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return ``[linear, angular]`` limits, or zero columns when disabled."""
    if not env.cfg.speed_limits.enabled:
        return torch.empty((env.num_envs, 0), device=env.device)
    return torch.stack((env._velocity_limit, env._angular_velocity_limit), dim=-1)


def linear_speed_limit_gap(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return signed linear-speed headroom: configured limit minus actual speed.

    Positive values mean the drone is below its assigned episode limit; negative
    values mean it exceeds that limit.  Disabled tasks return zeros so callers
    can use this diagnostic without task-specific branching.
    """
    if not env.cfg.speed_limits.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    speed = torch.linalg.vector_norm(env._robot.data.root_lin_vel_w, dim=1)
    return env._velocity_limit - speed


def penalty_velocity_limit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Quadratic world-linear-speed excess, or zeros when limits are disabled."""
    if not env.cfg.speed_limits.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    speed = torch.linalg.vector_norm(env._robot.data.root_lin_vel_w, dim=1)
    return (speed - env._velocity_limit).clamp(min=0.0).square()


def penalty_angular_velocity_limit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Quadratic body-angular-speed excess, or zeros when limits are disabled."""
    if not env.cfg.speed_limits.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    speed = torch.linalg.vector_norm(env._robot.data.root_ang_vel_b, dim=1)
    return (speed - env._angular_velocity_limit).clamp(min=0.0).square()

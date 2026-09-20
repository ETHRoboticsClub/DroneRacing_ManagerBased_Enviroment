"""Task-independent adaptive curricula driven by normalized route progress."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase

from Racing_Quadcopter.tasks.manager_based.shared_features import linear_speed_limit_gap


@dataclass(frozen=True)
class CurriculumTarget:
    """Typed, immutable declaration of one progressive curriculum binding."""

    name: str
    target_type: str
    attr_path: str
    mode: str
    target_value: object
    start_value: object | None = None
    term_name: str | None = None
    group_name: str | None = None
    bool_threshold: float = 1.0e-6
    dtype: str | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None



@dataclass
class _ResolvedTarget:
    """Runtime binding resolved from a :class:`CurriculumTarget`."""

    spec: CurriculumTarget
    root: object
    hook: Callable[[object], None] | None
    start: object
    last: object


@dataclass(frozen=True)
class _RegisteredRandomizationProfile:
    """One task-owned value object and its task-owned target factory."""

    values: object
    target_factory: Callable[[], tuple[CurriculumTarget, ...]]


_RANDOMIZATION_PROFILES: dict[str, _RegisteredRandomizationProfile] = {}


def _finite_param(params: dict, name: str, default: float) -> float:
    """Return a finite numeric curriculum parameter with a useful error."""
    try:
        value = float(params.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return value


def _validate_progress_window_params(params: dict) -> None:
    """Validate the shared progress-window timing and decision thresholds."""
    interval = _finite_param(params, "eval_interval_s", 120.0)
    low = _finite_param(params, "low_ratio_threshold", 0.50)
    high = _finite_param(params, "high_ratio_threshold", 0.75)
    if interval <= 0.0:
        raise ValueError(f"eval_interval_s must be positive, got {interval}.")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            "Progress thresholds must satisfy 0 <= low_ratio_threshold "
            f"< high_ratio_threshold <= 1; got ({low}, {high})."
        )


def _validate_reward_curriculum_params(params: dict) -> None:
    """Validate reward-scale bounds before binding manager terms."""
    _validate_progress_window_params(params)
    step = _finite_param(params, "scale_step_fraction", 0.10)
    minimum = _finite_param(params, "min_scale", 1.0)
    maximum = _finite_param(params, "max_scale_default", 5.0)
    if step < 0.0:
        raise ValueError(f"scale_step_fraction must be non-negative, got {step}.")
    if minimum < 0.0:
        raise ValueError(f"min_scale must be non-negative, got {minimum}.")
    if maximum < minimum:
        raise ValueError(
            "max_scale_default must be at least min_scale; "
            f"got ({maximum}, {minimum})."
        )
    maxima = params.get("max_scale_by_term", {})
    if not isinstance(maxima, dict):
        raise ValueError("max_scale_by_term must be a dictionary when provided.")
    for term_name, maximum_for_term in maxima.items():
        value = _finite_param({"value": maximum_for_term}, "value", 0.0)
        if value < minimum:
            raise ValueError(
                f"max_scale_by_term[{term_name!r}] must be at least min_scale "
                f"({minimum}), got {value}."
            )


def _validate_randomization_curriculum_params(params: dict) -> None:
    """Validate randomization-scale bounds before resolving profile bindings."""
    _validate_progress_window_params(params)
    step = _finite_param(params, "scale_step_fraction", 0.10)
    minimum = _finite_param(params, "min_noise_scale", 0.05)
    maximum = _finite_param(params, "max_noise_scale", 1.0)
    initial = _finite_param(params, "initial_noise_scale", 0.20)
    if step < 0.0:
        raise ValueError(f"scale_step_fraction must be non-negative, got {step}.")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError(
            "Noise-scale bounds must satisfy 0 <= min_noise_scale <= "
            f"max_noise_scale <= 1; got ({minimum}, {maximum})."
        )
    if not 0.0 <= initial <= maximum:
        raise ValueError(
            "initial_noise_scale must satisfy 0 <= initial_noise_scale <= "
            f"max_noise_scale; got ({initial}, {maximum})."
        )


def register_randomization_profile(
    name: str, values: object, target_factory: Callable[[], tuple[CurriculumTarget, ...]]
) -> None:
    """Register one task-local value profile and its target declarations."""
    if not name:
        raise ValueError("Randomization profile name cannot be empty.")
    _RANDOMIZATION_PROFILES[name] = _RegisteredRandomizationProfile(values, target_factory)


def get_randomization_profile(name: str) -> object:
    """Return the task-local values object registered under ``name``."""
    try:
        return _RANDOMIZATION_PROFILES[name].values
    except KeyError as exc:
        raise ValueError(
            f"Unknown randomization profile {name!r}; available: {sorted(_RANDOMIZATION_PROFILES)}."
        ) from exc


def get_randomization_targets(name: str) -> tuple[CurriculumTarget, ...]:
    """Build the registered task-local target declarations for runtime use."""
    try:
        return tuple(_RANDOMIZATION_PROFILES[name].target_factory())
    except KeyError as exc:
        raise ValueError(
            f"Unknown randomization profile {name!r}; available: {sorted(_RANDOMIZATION_PROFILES)}."
        ) from exc


class _RouteProgressWindow:
    """Shared append-only reset accounting for all route-progress controllers."""

    def __init__(self) -> None:
        self.progress_sum = 0.0
        self.completion_sum = 0.0
        self.total_resets = 0.0
        self._last_recorded_step = -1

    def add(self, env: ManagerBasedRLEnv, env_ids, completion_term: str | None) -> None:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
        if ids.numel() == 0 or self._last_recorded_step == int(env.common_step_counter):
            return
        # ManagerBasedRLEnv invokes curricula once during its initial reset.  That
        # reset did not finish an episode and must not bias the first window.
        episode_lengths = getattr(env, "episode_length_buf", None)
        if episode_lengths is not None:
            ids = ids[episode_lengths[ids] > 0]
        if ids.numel() == 0:
            return
        self.progress_sum += float(env._last_episode_route_progress_score[ids].sum().item())
        if completion_term is not None:
            completed = env.termination_manager.get_term(completion_term)[ids]
            self.completion_sum += float(completed.float().sum().item())
        self.total_resets += float(ids.numel())
        self._last_recorded_step = int(env.common_step_counter)

    def snapshot(self) -> tuple[float, float, float]:
        """Return cumulative progress, completions, and completed episodes."""
        return self.progress_sum, self.completion_sum, self.total_resets

    def values_since(self, start: tuple[float, float, float]) -> tuple[float, float, float, float]:
        """Return averages for one controller's independent window."""
        progress_sum = self.progress_sum - start[0]
        completion_sum = self.completion_sum - start[1]
        total_resets = self.total_resets - start[2]
        if total_resets == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            progress_sum / total_resets,
            completion_sum / total_resets,
            total_resets,
            completion_sum,
        )


class _RouteProgressController(ManagerTermBase):
    """Common timing and window lifecycle for adaptive curricula."""

    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Multiple curriculum terms run for the same reset.  They must consume
        # one shared measurement rather than double-counting completed episodes.
        if not hasattr(env, "_route_progress_curriculum_window"):
            env._route_progress_curriculum_window = _RouteProgressWindow()
        self._window = env._route_progress_curriculum_window
        self._window_start = self._window.snapshot()
        self._last_eval_step = int(env.common_step_counter)
        # Curricula adapt difficulty during training; evaluation runs must see fixed
        # conditions. Mirrors apply_staged_curriculum, which already returns early here.
        self._eval_mode = bool(getattr(env.cfg, "eval_mode", False))

    def _measure(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        completion_term: str | None,
        eval_interval_s: float,
    ) -> tuple[float, float, float, float, bool]:
        self._window.add(env, env_ids, completion_term)
        progress, completion, total, completed = self._window.values_since(self._window_start)
        interval_steps = max(1, int(round(float(eval_interval_s) / float(env.step_dt))))
        due = int(env.common_step_counter) - self._last_eval_step >= interval_steps
        return progress, completion, total, completed, due

    def _finish_update(self, env: ManagerBasedRLEnv) -> None:
        self._window_start = self._window.snapshot()
        self._last_eval_step = int(env.common_step_counter)


class AdaptiveRewardScaleFromRouteProgress(_RouteProgressController):
    """Scale selected reward weights from the shared normalized route score."""

    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        _validate_reward_curriculum_params(cfg.params)
        self._reward_term_names = tuple(cfg.params["reward_term_names"])
        self._increase_only_term_names = frozenset(cfg.params.get("increase_only_term_names", ()))
        unknown = self._increase_only_term_names.difference(self._reward_term_names)
        if unknown:
            raise ValueError(
                "increase_only_term_names must be included in reward_term_names; "
                f"unknown terms: {tuple(sorted(unknown))}."
            )
        self._reward_term_cfgs = {
            name: env.reward_manager.get_term_cfg(name) for name in self._reward_term_names
        }
        self._base_weights = {
            name: float(term_cfg.weight) for name, term_cfg in self._reward_term_cfgs.items()
        }
        self._term_scales = dict.fromkeys(self._reward_term_names, 1.0)
        self._enabled_attr = cfg.params.get("enabled_attr")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        completion_term: str | None = None,
        reward_term_names: tuple[str, ...] = ("action_smoothness", "gate_centering"),
        eval_interval_s: float = 120.0,
        low_ratio_threshold: float = 0.50,
        high_ratio_threshold: float = 0.75,
        scale_step_fraction: float = 0.10,
        min_scale: float = 1.0,
        max_scale_default: float = 5.0,
        max_scale_by_term: dict[str, float] | None = None,
        increase_only_term_names: tuple[str, ...] = (),
        enabled_attr: str | None = None,
    ) -> dict[str, float]:
        if tuple(reward_term_names) != self._reward_term_names:
            raise ValueError("reward_term_names cannot change after controller initialization.")
        if frozenset(increase_only_term_names) != self._increase_only_term_names:
            raise ValueError("increase_only_term_names cannot change after controller initialization.")
        if enabled_attr != self._enabled_attr:
            raise ValueError("enabled_attr cannot change after controller initialization.")
        if self._eval_mode or (enabled_attr is not None and not bool(getattr(env.cfg, enabled_attr))):
            out = {"enabled": 0.0, "updated": 0.0}
            for name in self._reward_term_names:
                out[f"reward_scale/{name}"] = self._term_scales[name]
                out[f"reward_weight/{name}"] = float(self._reward_term_cfgs[name].weight)
            return out

        progress, completion, total, completed, due = self._measure(
            env, env_ids, completion_term, eval_interval_s
        )
        updated = 0.0
        decision = "hold"
        if due:
            step_fraction = max(0.0, float(scale_step_fraction))
            if progress < float(low_ratio_threshold):
                factor = 1.0 / (1.0 + step_fraction)
                decision = "decrease"
            elif progress > float(high_ratio_threshold):
                factor = 1.0 + step_fraction
                decision = "increase"
            else:
                factor = 1.0

            lower = float(min_scale)
            maxima = max_scale_by_term or {}
            for name in self._reward_term_names:
                term_factor = 1.0 if name in self._increase_only_term_names and factor < 1.0 else factor
                upper = max(lower, float(maxima.get(name, max_scale_default)))
                scale = max(lower, min(upper, self._term_scales[name] * term_factor))
                self._term_scales[name] = scale
                term_cfg = self._reward_term_cfgs[name]
                term_cfg.weight = self._base_weights[name] * scale
                env.reward_manager.set_term_cfg(name, term_cfg)
            updated = 1.0
            print(
                "[RouteProgressRewardCurriculum] "
                f"progress={progress:.3f}, completion={completion:.3f}, decision={decision}, "
                f"scales={self._term_scales}"
            )
            self._finish_update(env)

        out = {
            "normalized_route_progress_window": progress,
            "completion_ratio_window": completion,
            "window_total_resets": total,
            "window_completed_resets": completed,
            "updated": updated,
        }
        for name in self._reward_term_names:
            out[f"reward_scale/{name}"] = self._term_scales[name]
            out[f"reward_weight/{name}"] = float(self._reward_term_cfgs[name].weight)
        return out


class AdaptiveRandomizationFromRouteProgress(_RouteProgressController):
    """Apply generic config-defined randomization from normalized route progress."""

    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        _validate_randomization_curriculum_params(cfg.params)
        self._bindings: list[_ResolvedTarget] = []
        self._profile_name = str(cfg.params["randomization_profile"])
        self._target_specs = get_randomization_targets(self._profile_name)
        self._validate_target_specs(self._target_specs)
        self._activate_speed_limits_on_increase = bool(
            cfg.params.get("activate_speed_limits_on_increase", False)
        )
        self._enabled_attr = cfg.params.get("enabled_attr")
        self._noise_scale = float(cfg.params["initial_noise_scale"])
        self._bindings_applied = False
        if not self._eval_mode and (self._enabled_attr is None or bool(getattr(env.cfg, self._enabled_attr))):
            self._initialize_bindings(env, self._target_specs)
            self._apply_scale_to_bindings(self._noise_scale)
            self._bindings_applied = True

    @staticmethod
    def _get_attr_path(root: object, path: str) -> object:
        for name in path.split("."):
            root = getattr(root, name)
        return root

    @staticmethod
    def _set_attr_path(root: object, path: str, value: object) -> None:
        parts = path.split(".")
        for name in parts[:-1]:
            root = getattr(root, name)
        setattr(root, parts[-1], value)

    @staticmethod
    def _validate_target_specs(specs: tuple[CurriculumTarget, ...]) -> None:
        """Fail at construction with a useful error for malformed bindings."""
        valid_types = {
            "env_cfg",
            "reward_term_cfg",
            "action_term_cfg",
            "observation_group_cfg",
            "observation_term_cfg",
        }
        names: set[str] = set()
        for index, spec in enumerate(specs):
            if not isinstance(spec, CurriculumTarget):
                raise TypeError(
                    f"Curriculum target {index} must be CurriculumTarget, got {type(spec).__name__}."
                )
            target_type = spec.target_type
            if target_type not in valid_types:
                raise ValueError(f"Curriculum target {index} has unknown target_type {target_type!r}.")
            if not spec.attr_path:
                raise ValueError(f"Curriculum target {index} has an empty attr_path.")
            if target_type in {"reward_term_cfg", "action_term_cfg", "observation_term_cfg"} and not spec.term_name:
                raise ValueError(f"Curriculum target {index} requires term_name.")
            if target_type in {"observation_group_cfg", "observation_term_cfg"} and not spec.group_name:
                raise ValueError(f"Curriculum target {index} requires group_name.")
            if spec.name in names:
                raise ValueError(f"Duplicate curriculum target name {spec.name!r}.")
            names.add(spec.name)

    @staticmethod
    def _resolve_target_root(env: ManagerBasedRLEnv, spec: CurriculumTarget):
        target_type = spec.target_type
        if target_type == "env_cfg":
            return env.cfg, None
        if target_type == "reward_term_cfg":
            name = spec.term_name
            return env.reward_manager.get_term_cfg(name), lambda cfg: env.reward_manager.set_term_cfg(name, cfg)
        if target_type == "action_term_cfg":
            return env.action_manager.get_term(spec.term_name).cfg, None
        if target_type == "observation_group_cfg":
            group = spec.group_name
            obs_cfg = env.observation_manager.cfg
            return (obs_cfg[group] if isinstance(obs_cfg, dict) else getattr(obs_cfg, group)), None
        if target_type == "observation_term_cfg":
            group = spec.group_name
            name = spec.term_name
            names = env.observation_manager._group_obs_term_names.get(group, [])
            if name not in names:
                raise ValueError(f"Observation term {name!r} not found in group {group!r}.")
            return env.observation_manager._group_obs_term_cfgs[group][names.index(name)], None
        raise ValueError(f"Unknown randomization target_type {target_type!r}.")

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return isinstance(value, (tuple, list))

    def _scale_value(self, spec: CurriculumTarget, scale: float, start: object) -> object:
        mode = spec.mode
        target = spec.target_value
        if isinstance(target, bool):
            value: object = target if scale >= spec.bool_threshold else bool(start)
        elif self._is_sequence(target):
            starts = tuple(float(v) for v in (start if self._is_sequence(start) else target))
            targets = tuple(float(v) for v in target)
            if mode != "lerp":
                raise ValueError(f"Unsupported sequence mode {mode!r}.")
            value = tuple(a + scale * (b - a) for a, b in zip(starts, targets, strict=True))
        elif isinstance(target, (int, float)):
            if mode == "lerp":
                value = float(start) + scale * (float(target) - float(start))
            elif mode == "mul":
                value = float(target) * scale
            else:
                raise ValueError(f"Unsupported numeric mode {mode!r}.")
        else:
            raise TypeError(f"Unsupported target value type: {type(target)}")
        if spec.clamp_min is not None and isinstance(value, (int, float)):
            value = max(spec.clamp_min, float(value))
        if spec.clamp_max is not None and isinstance(value, (int, float)):
            value = min(spec.clamp_max, float(value))
        if spec.dtype == "int":
            value = int(round(float(value)))
        return value

    def _initialize_bindings(self, env: ManagerBasedRLEnv, specs: tuple[CurriculumTarget, ...]) -> None:
        for spec in specs:
            root, hook = self._resolve_target_root(env, spec)
            target = spec.target_value
            start = spec.start_value
            if start is None:
                if isinstance(target, bool):
                    start = False
                elif self._is_sequence(target):
                    start = tuple(0.0 for _ in target)
                else:
                    start = 0.0
            self._bindings.append(
                _ResolvedTarget(spec, root, hook, start, self._get_attr_path(root, spec.attr_path))
            )

    def _apply_scale_to_bindings(self, scale: float) -> None:
        for binding in self._bindings:
            value = self._scale_value(binding.spec, scale, binding.start)
            self._set_attr_path(binding.root, binding.spec.attr_path, value)
            if binding.hook is not None:
                binding.hook(binding.root)
            binding.last = value

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        completion_term: str | None = None,
        eval_interval_s: float = 120.0,
        low_ratio_threshold: float = 0.50,
        high_ratio_threshold: float = 0.75,
        scale_step_fraction: float = 0.10,
        min_noise_scale: float = 0.05,
        max_noise_scale: float = 1.0,
        initial_noise_scale: float = 0.20,
        randomization_profile: str = "",
        activate_speed_limits_on_increase: bool = False,
        enabled_attr: str | None = None,
    ) -> dict[str, float]:
        del initial_noise_scale
        if randomization_profile != self._profile_name:
            raise ValueError("randomization_profile cannot change after controller initialization.")
        if activate_speed_limits_on_increase != self._activate_speed_limits_on_increase:
            raise ValueError("activate_speed_limits_on_increase cannot change after initialization.")
        if enabled_attr != self._enabled_attr:
            raise ValueError("enabled_attr cannot change after controller initialization.")
        if self._eval_mode or (enabled_attr is not None and not bool(getattr(env.cfg, enabled_attr))):
            return {
                "enabled": 0.0,
                "updated": 0.0,
                "randomization_noise_scale": self._noise_scale,
            }
        if not self._bindings_applied:
            self._initialize_bindings(env, self._target_specs)
            self._apply_scale_to_bindings(self._noise_scale)
            self._bindings_applied = True
        progress, completion, total, completed, due = self._measure(
            env, env_ids, completion_term, eval_interval_s
        )
        updated = 0.0
        if due:
            previous = self._noise_scale
            change = max(0.0, float(scale_step_fraction)) * (1.0 - self._noise_scale)
            decision = "hold"
            if progress < float(low_ratio_threshold):
                self._noise_scale -= change
                decision = "decrease"
            elif progress > float(high_ratio_threshold):
                self._noise_scale += change
                decision = "increase"
                if self._activate_speed_limits_on_increase:
                    from Racing_Quadcopter.tasks.manager_based.shared_features import (
                        activate_speed_limit_randomization,
                    )

                    activate_speed_limit_randomization(env)
            lower = float(min_noise_scale)
            upper = max(lower, float(max_noise_scale))
            self._noise_scale = max(lower, min(upper, self._noise_scale))
            self._apply_scale_to_bindings(self._noise_scale)
            updated = 1.0
            print(
                "[RouteProgressRandomizationCurriculum] "
                f"progress={progress:.3f}, completion={completion:.3f}, decision={decision}, "
                f"noise_scale={previous:.3f}->{self._noise_scale:.3f}"
            )
            self._finish_update(env)

        out = {
            "normalized_route_progress_window": progress,
            "completion_ratio_window": completion,
            "window_total_resets": total,
            "window_completed_resets": completed,
            "updated": updated,
            "randomization_noise_scale": self._noise_scale,
        }
        speed_limits_cfg = getattr(env.cfg, "speed_limits", None)
        if speed_limits_cfg is not None and speed_limits_cfg.enabled:
            speed_gap = linear_speed_limit_gap(env)
            out["speed_limit_randomization_active"] = float(env._speed_limit_randomization_active)
            out["linear_speed_limit_mean"] = float(env._velocity_limit.mean().item())
            out["angular_speed_limit_mean"] = float(env._angular_velocity_limit.mean().item())
            # Keep both signed headroom and absolute error: the former shows whether the
            # policy is generally below/above its limit, while the latter cannot cancel
            # under-limit and over-limit environments in a batched mean.
            out["linear_speed_limit_gap_mean"] = float(speed_gap.mean().item())
            out["linear_speed_limit_gap_abs_mean"] = float(speed_gap.abs().mean().item())
            out["linear_speed_mean"] = float((env._velocity_limit - speed_gap).mean().item())
        for binding in self._bindings:
            value = binding.last
            if isinstance(value, (bool, int, float)):
                out[f"target/{binding.spec.name}"] = float(value)
        return out


def action_noise_targets(
    *,
    gaussian_std: float = 0.02,
    bias_range: tuple[float, float] = (-0.05, 0.05),
    lpf_alpha_range: tuple[float, float] = (0.8, 1.0),
) -> tuple[CurriculumTarget, ...]:
    """Reusable progressive targets for the shared action-corruption config."""
    return (
        CurriculumTarget("action_noise_enable", "env_cfg", "action_noise.enable", "lerp", True, False),
        CurriculumTarget("action_noise_gaussian_std", "env_cfg", "action_noise.gaussian_std", "lerp", gaussian_std, 0.0),
        CurriculumTarget("action_noise_bias_range", "env_cfg", "action_noise.bias_range", "lerp", bias_range, (0.0, 0.0)),
        CurriculumTarget("action_noise_lpf_alpha_range", "env_cfg", "action_noise.lpf_alpha_range", "lerp", lpf_alpha_range, (1.0, 1.0)),
    )


def observation_noise_targets(
    term_names: tuple[str, ...],
    *,
    group_name: str = "policy",
    std: float = 0.05,
    bias_std: float = 0.01,
) -> tuple[CurriculumTarget, ...]:
    """Build enable/std/bias bindings for identically configured observation terms."""
    # The observation group is kept corruption-enabled at construction so Isaac
    # Lab preserves per-term noise configs. Individual models stay disabled
    # until the curriculum enables them.
    targets: list[CurriculumTarget] = []
    for term_name in term_names:
        targets.extend(
            (
                CurriculumTarget(f"obs_{term_name}_enable", "observation_term_cfg", "noise.enable", "lerp", True, False, term_name, group_name),
                CurriculumTarget(f"obs_{term_name}_std", "observation_term_cfg", "noise.noise_cfg.std", "lerp", std, 0.0, term_name, group_name),
                CurriculumTarget(f"obs_{term_name}_bias_std", "observation_term_cfg", "noise.bias_noise_cfg.std", "lerp", bias_std, 0.0, term_name, group_name),
            )
        )
    return tuple(targets)


def quadcopter_dynamics_targets(
    *,
    action_term_name: str = "rates",
    max_delay: int = 2,
    pid_gain_randomization_frac: float = 0.10,
    thrust_to_weight_randomization_frac: float = 0.05,
    aero_randomization_frac: float = 0.10,
    linear_drag_coeff: tuple[float, float, float] = (0.08, 0.08, 0.12),
    quadratic_drag_coeff: tuple[float, float, float] = (0.01, 0.01, 0.018),
    angular_drag_coeff: tuple[float, float, float] = (0.002, 0.002, 0.0012),
) -> tuple[CurriculumTarget, ...]:
    """Reusable delay/controller/thrust/aerodynamic randomization bindings."""
    def target(name: str, path: str, value: object) -> CurriculumTarget:
        return CurriculumTarget(
            name=name,
            target_type="action_term_cfg",
            term_name=action_term_name,
            attr_path=path,
            mode="lerp",
            start_value=False if isinstance(value, bool) else None,
            target_value=value,
        )

    return action_delay_targets(action_term_name=action_term_name, max_delay=max_delay) + (
        target("rates_pid_gain_rand_frac", "pid_gain_randomization_frac", pid_gain_randomization_frac),
        target(
            "rates_thrust_to_weight_rand_frac",
            "thrust_to_weight_randomization_frac",
            thrust_to_weight_randomization_frac,
        ),
        target(
            "rates_aero_randomization_enable",
            "aero_randomization_enable",
            True,
        ),
        target("rates_aero_randomization_frac", "aero_randomization_frac", aero_randomization_frac),
        target("rates_linear_drag_coeff", "linear_drag_coeff", linear_drag_coeff),
        target("rates_quadratic_drag_coeff", "quadratic_drag_coeff", quadratic_drag_coeff),
        target("rates_angular_drag_coeff", "angular_drag_coeff", angular_drag_coeff),
    )


def action_delay_targets(
    *, action_term_name: str, max_delay: int = 2
) -> tuple[CurriculumTarget, ...]:
    """Build a progressive delay binding for any compatible action term."""
    return (
        CurriculumTarget(
            name=f"{action_term_name}_max_delay",
            target_type="action_term_cfg",
            term_name=action_term_name,
            attr_path="max_delay",
            mode="lerp",
            start_value=0,
            target_value=max_delay,
            dtype="int",
            clamp_min=0,
        ),
    )

"""Optional action diagnostics shared by manager-based quadcopter tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class ActionDiagnostics:
    """Own rate-limited action statistics and their per-environment history."""

    _DIM_NAMES = ("thrust", "roll", "pitch", "yaw")

    def __init__(self, env) -> None:
        self._env = env
        self._physical = torch.zeros_like(env._actions_buf)
        self._previous_physical = torch.zeros_like(env._actions_buf)
        self._last_step = -1

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long).view(-1)
        self._physical[ids] = 0.0
        self._previous_physical[ids] = 0.0

    def update_if_due(self) -> None:
        env = self._env
        if not env.cfg.action_diagnostics_enabled:
            return
        interval = max(1, int(round(env.cfg.action_diagnostics_interval_s / env.step_dt)))
        if env.common_step_counter - self._last_step < interval:
            return
        self._last_step = env.common_step_counter

        self._previous_physical[:] = self._physical
        self._physical[:] = self._physical_commands()
        series = {
            "raw_cmd_abs": env._policy_raw_actions.abs(),
            "raw_rate_abs": (env._policy_raw_actions - env._prev_policy_raw_actions).abs(),
            "normalized_cmd_abs": env._actions_buf.abs(),
            "normalized_rate_abs": (env._actions_buf - env._prev_actions).abs(),
            "physical_scaled_cmd_abs": self._physical.abs(),
            "physical_scaled_rate_abs": (self._physical - self._previous_physical).abs(),
        }
        for prefix, values in series.items():
            self._log_vector(values, prefix)

    def _log_vector(self, values: torch.Tensor, prefix: str) -> None:
        means = values.mean(dim=0)
        stds = values.std(dim=0, unbiased=False)
        if values.shape[0] == 0:
            p95 = torch.zeros(values.shape[1], device=values.device, dtype=values.dtype)
        else:
            rank = max(1, int(math.ceil(0.95 * values.shape[0])))
            p95 = values.kthvalue(rank, dim=0).values
        log = self._env.extras.setdefault("log", {})
        for index, name in enumerate(self._DIM_NAMES):
            log[f"ActionStats/{prefix}/mean/{name}"] = means[index]
            log[f"ActionStats/{prefix}/std/{name}"] = stds[index]
            log[f"ActionStats/{prefix}/p95/{name}"] = p95[index]

    def _physical_commands(self) -> torch.Tensor:
        env = self._env
        commands = torch.zeros_like(env._actions_buf)
        if not env.action_manager.active_terms:
            return commands
        term = env.action_manager.get_term(env.action_manager.active_terms[0])
        if hasattr(term, "_thrust"):
            commands[:, 0] = term._thrust[:, 0, 2]
        if hasattr(term.cfg, "rate_command_scale") and hasattr(term, "_delayed_actions"):
            scale = torch.as_tensor(term.cfg.rate_command_scale, device=env.device, dtype=commands.dtype)
            if scale.ndim == 0:
                scale = scale.repeat(3)
            commands[:, 1:] = term._delayed_actions[:, 1:] * scale.view(1, 3)
        elif hasattr(term, "_moment"):
            commands[:, 1:] = term._moment[:, 0, :]
        else:
            commands[:] = env._actions_buf
        return commands

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Periodic console statistics for the racing quadcopter environment.

Sliding-window design
---------------------
``_speed_window``  — one float per *env-episode* (mean speed in m/s for that env
                     over its whole episode).  Each env-episode counts once,
                     regardless of episode length or how many envs are flying.

``_run_window``    — one float per *run* (0-1 track completion fraction).
                     A run starts at spawn or right after a lap finishes.
                     Completed laps append 1.0.
                     Terminated (crash/out-of-bounds) runs append the fraction
                     of the circuit completed before the terminal step.
                     Timed-out unfinished laps are silently ignored.

``_gates_window``  — one int per *episode* (total gates passed that episode).

``Speed``  = mean(_speed_window) × 3.6       →  average mean speed per episode (km/h).
``Lap%``   = mean(_run_window) × 100         →  average track completion per run.
``Gates``  = mean(_gates_window)             →  average gates passed per episode.

Warmup period
-------------
``cfg.stats_warmup_s`` (default 0, set to 30 s in Play configs) suppresses stats
accumulation for the first N simulation seconds.  Early episodes are discarded because
drones have not yet settled into steady-state racing behaviour — lap completions are
rare and speed/centering numbers are unrepresentative.  The warmup duration and
completion are printed to console.  ``step_stats`` prints a progress indicator while
warming up so the user can see elapsed time.
"""

from __future__ import annotations

import os
import time
from collections import deque

import torch

try:
    from tqdm import tqdm as _tqdm
    _tqdm_write = _tqdm.write
except ImportError:
    _tqdm_write = print

from isaaclab.envs import ManagerBasedRLEnv

_WINDOW_SIZE = 2000


def _stats_mode(env: ManagerBasedRLEnv) -> str:
    """Return the current run mode label used in stats lines (TRAIN / EVAL / RUN)."""
    mode = getattr(env, "_stats_mode", None)
    if mode is None:
        mode = os.environ.get("RACING_RUN_MODE", "RUN")
    mode = str(mode).strip().upper()
    return mode if mode else "RUN"


def init_stats(env: ManagerBasedRLEnv) -> None:
    """Initialise all running statistics used for the periodic console print."""
    env._stats_mode = _stats_mode(env)
    env._last_speed_log_time = time.monotonic()
    env._total_lap_time_sum = 0.0
    env._total_lap_count = 0
    env._speed_window = deque(maxlen=_WINDOW_SIZE)   # float m/s, one entry per env-episode
    env._run_window   = deque(maxlen=_WINDOW_SIZE)   # float 0-1, one entry per run
    env._gates_window = deque(maxlen=_WINDOW_SIZE)   # int, one entry per episode

    warmup_s = float(getattr(env.cfg, "stats_warmup_s", 0.0))
    env._stats_warmup_steps = int(warmup_s / env.step_dt) if warmup_s > 0.0 else 0
    env._stats_warmup_done = env._stats_warmup_steps == 0

    if env._stats_warmup_steps > 0:
        _tqdm_write(
            f"[{_stats_mode(env)} Stats] Warmup: {warmup_s:.0f} s ({env._stats_warmup_steps} steps) — "
            "stats collection paused until drones settle."
        )


def _in_warmup(env: ManagerBasedRLEnv) -> bool:
    """Return True while the warmup period is still active."""
    if env._stats_warmup_done:
        return False
    if env.common_step_counter < env._stats_warmup_steps:
        return True
    # Warmup just ended — print once and latch the flag.
    env._stats_warmup_done = True
    elapsed_s = env.common_step_counter * env.step_dt
    _tqdm_write(f"[{_stats_mode(env)} Stats] Warmup complete at {elapsed_s:.1f} s — stats collection active.")
    return False


def step_stats(env: ManagerBasedRLEnv) -> None:
    """Emit the periodic stats line when the log interval elapses."""
    now = time.monotonic()
    if now - env._last_speed_log_time < env.cfg.speed_log_interval_s:
        return
    env._last_speed_log_time = now

    if _in_warmup(env):
        elapsed_s = env.common_step_counter * env.step_dt
        total_s = env._stats_warmup_steps * env.step_dt
        _tqdm_write(
            f"[{_stats_mode(env)} Stats | warming up] {elapsed_s:.0f} / {total_s:.0f} s elapsed — "
            "waiting for drones to settle..."
        )
        return

    sw, rw, gw = env._speed_window, env._run_window, env._gates_window
    speed_str  = f"{sum(sw) / len(sw) * 3.6:.1f} km/h" if sw else "n/a"
    lap_pct    = f"{sum(rw) / len(rw) * 100:.1f}%"      if rw else "n/a"
    mean_gates = f"{sum(gw) / len(gw):.1f}"              if gw else "n/a"
    lap_str    = (
        f"{env._total_lap_time_sum / env._total_lap_count:.1f} s"
        if env._total_lap_count > 0 else "n/a"
    )

    _tqdm_write(
        f"[{_stats_mode(env)} Stats | n={len(rw)}] "
        f"Speed: {speed_str}  |  "
        f"Gates/ep: {mean_gates}  |  "
        f"Lap%: {lap_pct}  |  "
        f"Lap time: {lap_str}"
    )


def accumulate_episode_stats(
    env: ManagerBasedRLEnv,
    env_ids_tensor: torch.Tensor,
    gates_passed: torch.Tensor,
    lap_counts: torch.Tensor,
    gate_counts: torch.Tensor,
    episode_mean_speed: torch.Tensor,
) -> None:
    """Update the sliding windows at episode end.

    Skipped entirely during the warmup period so early garbage episodes do not
    pollute the windows.

    For each env being reset:
      - Its episode mean speed (m/s) is appended to ``_speed_window``.
      - Each completed lap this episode contributes 1.0 to ``_run_window``.
      - The unfinished final run contributes ``gates_in_run / gate_count``
        **only if the episode terminated** (crash/out-of-bounds).
        Timed-out unfinished laps are discarded.
      - Total gates passed this episode is appended to ``_gates_window``.
    """
    if _in_warmup(env):
        return

    reset_terminated = getattr(env, "reset_terminated", None)

    speed_list = episode_mean_speed.tolist()
    laps_list  = lap_counts.long().tolist()
    gates_list = gates_passed.long().tolist()
    gc_list    = gate_counts.long().tolist()
    term_list  = (
        reset_terminated[env_ids_tensor].tolist()
        if reset_terminated is not None
        else [False] * len(laps_list)
    )

    for speed, laps, gp, gc, is_term in zip(speed_list, laps_list, gates_list, gc_list, term_list):
        env._speed_window.append(speed)

        for _ in range(laps):
            env._run_window.append(1.0)

        # Unfinished run at episode end — only count if terminated, not timed out.
        if is_term and gc > 0:
            env._run_window.append((gp % gc) / gc)

        env._gates_window.append(gp)

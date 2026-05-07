#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hyperparameter grid-search sweep for the drone racing environment.

Usage
-----
Edit PARAM_GRID, TRAIN_ITERATIONS, EVAL_STEPS, NUM_ENVS at the top, then run::

    python scripts/sweep.py

Or provide options from the command line::

    python scripts/sweep.py --config path/to/sweep.yaml

Results are written to ``logs/sweep/results_<timestamp>.csv`` and printed as a
summary table when the sweep finishes.

How it works
------------
For every combination of values in PARAM_GRID the script:
  1. Trains a policy via ``scripts/skrl/train.py`` with ``--override`` flags
     and a unique ``--exp-name`` so checkpoints land in
     ``logs/skrl/sweep/<exp_name>/``.
  2. Finds the best checkpoint produced by that run.
  3. Evaluates the checkpoint via ``scripts/skrl/play.py`` with
     ``--eval-steps`` to limit the evaluation length.  The play script
     prints a ``[SWEEP_RESULT]`` line that this script parses.
  4. Appends the result row to the CSV and prints a live summary table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import os
import re
import subprocess
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION — edit this section to define your sweep
# ---------------------------------------------------------------------------

TASK_TRAIN = "Isaac-Racing-Quadcopter-Manager-Fixed-v0"
TASK_PLAY  = "Isaac-Racing-Quadcopter-Manager-Fixed-Play-v0"

# Each key is an env_cfg attribute (dotted paths like "rewards.crash.weight" are OK).
# Each value is a list of values to try.  Every combination will be evaluated.
#
# Useful parameters to sweep:
#
#   recovery_spawn_prob       float [0-1]   Fraction of envs that spawn tumbling/inverted
#                                           in stage 2.  Higher → more robustness training,
#                                           harder early learning.
#
#   curriculum_stage1_steps   int           Step at which the hover phase ends and full racing
#                                           begins (upside-down termination turns off, crash
#                                           weights restored).  Too low → unstable start;
#                                           too high → wasted time in hover phase.
#
#   curriculum_stage2_steps   int           Step at which recovery spawns activate.
#                                           Set to 0 to disable recovery spawns entirely.
#
#   episode_length_s          float         Episode duration in seconds.  Longer = more gates
#                                           per episode but slower reset throughput.
#
#   rewards.crash.weight      float         Crash penalty (negative).  Default: -5000.0
#                                           (MANAGER_SCALE 100 × direct weight -50).
#                                           More negative → agent avoids crashes harder
#                                           at the risk of being overly conservative.
#
# NOTE:
# This is an intentionally richer example spanning multiple categories.
# Grid size grows multiplicatively; trim value lists if runs become too expensive.
PARAM_GRID: dict[str, list] = {
    # ------------------------------------------------------------------
    # Curriculum / general task settings
    # ------------------------------------------------------------------
    "recovery_spawn_prob": [0.2],
    "curriculum_stage1_steps": [10000],
    "curriculum_stage2_steps": [10000],
    "episode_length_s": [30.0],

    # ------------------------------------------------------------------
    # Dynamics / task-level randomization
    # ------------------------------------------------------------------
    "randomize_velocity_limit": [True],
    "velocity_limit_range": ["(8.0, 15.0)"],

    # ------------------------------------------------------------------
    # Observation randomization
    # ------------------------------------------------------------------
    "observation_noise.enable": [True],
    "observation_noise.gaussian_std_base_lin_vel": [0.015],
    "observation_noise.gaussian_std_base_ang_vel": [0.015],
    "observation_noise.gaussian_std_obs_gates": [0.02],
    "observation_noise.bias_range_obs_gates": ["(-0.08, 0.08)"],
    "observation_noise.dropout_prob": [0.0],

    # ------------------------------------------------------------------
    # Action randomization
    # ------------------------------------------------------------------
    "action_noise.enable": [True],
    "action_noise.gaussian_std": [0.02],
    "action_noise.bias_range": ["(-0.03, 0.03)"],
    "action_noise.lpf_alpha": [0.85],

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    "rewards.gate_progress.weight": [20.0],
    # Convenience shorthand: direct (unscaled) coefficient → converted to
    # rewards.<term>.weight by multiplying with MANAGER_SCALE.
    "DIRECT_VELOCITY_TOWARD_GATE": [0.1],
}

TRAIN_ITERATIONS: int = 417   # --max_iterations passed to train.py (iterations, not timesteps)
                               # total timesteps = TRAIN_ITERATIONS * rollouts (usually 24)
                               # e.g. 1000 → ~24k steps (quick sanity check)
                               #      5000 → ~120k steps (decent comparison)
EVAL_STEPS: int       = 3000    # --eval-steps passed to play.py (env steps, not timesteps)
NUM_ENVS: int         = 4096    # number of parallel envs for both train and eval

# Pass extra fixed args to train / play (e.g. ["--headless"])
EXTRA_TRAIN_ARGS: list[str] = ["--headless"]
EXTRA_PLAY_ARGS:  list[str] = ["--headless"]

# Combination strategy:
# - "grid": full Cartesian product (all permutations)
# - "sequential": index-aligned selection (run i uses PARAM_GRID[key][i] for all keys)
#                 Requires all PARAM_GRID value-lists to have identical length.
SWEEP_MODE: str = "grid"

# ---------------------------------------------------------------------------
# Internals — no edits needed below
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON     = sys.executable


def _read_manager_scale() -> float:
    """Read MANAGER_SCALE from the racing rewards source file without importing the package.

    This avoids importing the Racing_Quadcopter package (which pulls in Omniverse
    extensions like `omni` that may not be available in this environment).
    """
    rewards_path = (
        REPO_ROOT
        / "source"
        / "Racing_Quadcopter"
        / "Racing_Quadcopter"
        / "tasks"
        / "manager_based"
        / "racing_quadcopter"
        / "mdp"
        / "rewards.py"
    )
    try:
        text = rewards_path.read_text()
        m = re.search(r"^MANAGER_SCALE\s*=\s*([0-9.eE+\-]+)", text, re.MULTILINE)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 100.0


MANAGER_SCALE = _read_manager_scale()

RESULTS_DIR = REPO_ROOT / "logs" / "sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_CSV = RESULTS_DIR / f"results_{SWEEP_ID}.csv"

CSV_FIELDS = ["exp_name"] + list(PARAM_GRID.keys()) + [
    "speed_kmh", "lap_pct", "mean_gates", "lap_time_s", "status",
]

_OVERRIDE_ERROR_MARKERS = (
    "Unknown override path",
    "Unknown override key",
    "Malformed override",
    "expects tuple-like value",
    "expects list-like value",
    "expects dict value",
    "Failed to cast override",
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sweep mode and YAML configuration."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps for drone racing.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file. Supported keys: "
            "param_grid, sweep_mode, task_train, task_play, train_iterations, "
            "eval_steps, num_envs, extra_train_args, extra_play_args, "
            "env_vars (dict of VAR: value, inherited by train/play subprocesses), "
            "gate_track."
        ),
    )
    return parser.parse_args()


def _load_yaml_config(path: str) -> dict:
    """Load and validate a YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Sweep config file not found: {cfg_path}")

    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from e

    loaded = yaml.safe_load(cfg_path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Sweep config YAML must be a mapping at the top level.")
    return loaded


def _normalize_param_grid(raw_param_grid: object) -> dict[str, list]:
    """Validate and normalize PARAM_GRID loaded from YAML."""
    if not isinstance(raw_param_grid, dict):
        raise ValueError("'param_grid' must be a mapping of key -> list.")

    normalized: dict[str, list] = {}
    for k, v in raw_param_grid.items():
        if not isinstance(k, str):
            raise ValueError(f"param_grid keys must be strings. Got key: {k!r}")
        if not isinstance(v, list):
            raise ValueError(
                f"param_grid['{k}'] must be a list. Got {type(v).__name__}."
            )
        normalized[k] = v

    if not normalized:
        raise ValueError("'param_grid' is empty. Provide at least one parameter.")

    return normalized


def _apply_env_vars(env_vars: object) -> None:
    """Set arbitrary process env vars from a ``env_vars`` YAML mapping.

    Keys become environment variable names; values are coerced to str.
    These are inherited by every subprocess spawned via _run_stream because
    that function copies os.environ at launch time.
    """
    if not isinstance(env_vars, dict):
        raise ValueError("'env_vars' must be a mapping of VAR_NAME -> value.")
    for k, v in env_vars.items():
        os.environ[str(k)] = str(v)
    print(f"[Sweep] env_vars set: {', '.join(f'{k}={v}' for k, v in env_vars.items())}")


def _apply_gate_track(gate_track: object) -> None:
    """Dump an inline ``gate_track`` list to a temp YAML and expose it via env var.

    The env_cfg reads ``RACING_GATE_TRACK_YAML`` in __post_init__ before terrain
    generation, which runs *before* ``--override`` is applied — so passing the
    track via env var (inherited by subprocesses) is the only way to inject it.
    """
    import tempfile

    if not isinstance(gate_track, list) or not gate_track:
        raise ValueError("'gate_track' must be a non-empty list of gate entries.")

    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError("PyYAML is required to dump gate_track (pip install pyyaml).") from e

    track_dir = RESULTS_DIR / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    fd, track_path = tempfile.mkstemp(prefix=f"track_{SWEEP_ID}_", suffix=".yaml", dir=str(track_dir))
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump({"gate_track": gate_track}, f, sort_keys=False)

    os.environ["RACING_GATE_TRACK_YAML"] = track_path
    print(f"[Sweep] Inline gate_track dumped to {track_path} (RACING_GATE_TRACK_YAML set).")


def _ensure_list(name: str, value: object) -> list:
    """Validate list-typed config fields."""
    if not isinstance(value, list):
        raise ValueError(f"'{name}' must be a list. Got {type(value).__name__}.")
    return list(value)


def _apply_runtime_config(args: argparse.Namespace) -> None:
    """Apply YAML and CLI configuration overrides onto module-level sweep settings.

    Precedence:
            defaults < YAML file (--config)
    """
    global TASK_TRAIN, TASK_PLAY
    global PARAM_GRID
    global TRAIN_ITERATIONS, EVAL_STEPS, NUM_ENVS
    global EXTRA_TRAIN_ARGS, EXTRA_PLAY_ARGS
    global SWEEP_MODE, CSV_FIELDS

    if args.config:
        cfg = _load_yaml_config(args.config)

        if "task_train" in cfg:
            TASK_TRAIN = str(cfg["task_train"])
        if "task_play" in cfg:
            TASK_PLAY = str(cfg["task_play"])
        if "train_iterations" in cfg:
            TRAIN_ITERATIONS = int(cfg["train_iterations"])
        if "eval_steps" in cfg:
            EVAL_STEPS = int(cfg["eval_steps"])
        if "num_envs" in cfg:
            NUM_ENVS = int(cfg["num_envs"])
        if "extra_train_args" in cfg:
            EXTRA_TRAIN_ARGS = [str(x) for x in _ensure_list("extra_train_args", cfg["extra_train_args"])]
        if "extra_play_args" in cfg:
            EXTRA_PLAY_ARGS = [str(x) for x in _ensure_list("extra_play_args", cfg["extra_play_args"])]
        if "param_grid" in cfg:
            PARAM_GRID = _normalize_param_grid(cfg["param_grid"])
        if "sweep_mode" in cfg:
            SWEEP_MODE = str(cfg["sweep_mode"])
        if "env_vars" in cfg:
            _apply_env_vars(cfg["env_vars"])
        if "gate_track" in cfg:
            _apply_gate_track(cfg["gate_track"])

    SWEEP_MODE = SWEEP_MODE.strip().lower()
    if SWEEP_MODE not in {"grid", "sequential"}:
        raise ValueError(f"Unknown SWEEP_MODE={SWEEP_MODE!r}. Supported values: 'grid', 'sequential'.")

    # Recompute CSV columns after potential PARAM_GRID override from YAML.
    CSV_FIELDS = ["exp_name"] + list(PARAM_GRID.keys()) + [
        "speed_kmh", "lap_pct", "mean_gates", "lap_time_s", "status",
    ]


def _is_tqdm_progress_line(line: str) -> bool:
    """Heuristic: detect tqdm progress updates.

    Some environments render tqdm updates as full newline-separated lines when
    stdout is piped, which floods logs. We detect such lines and render them
    in-place from this parent process.
    """
    s = line.strip()
    if not s:
        return False
    if "it/s" in s and "%|" in s:
        return True
    # Fallback for partially wrapped tqdm lines.
    return bool(re.search(r"\b\d{1,3}%\|", s))


def _run_stream(cmd: list[str], run_mode: str | None = None) -> tuple[int, str]:
    """Run *cmd*, stream its stdout to our stdout, and return (returncode, captured_output).

    TQDM_DISABLE=1 prevents skrl's tqdm bars from flooding the output with
    carriage-return update lines when stdout is piped.
    PYTHONUNBUFFERED=1 ensures print() in subprocesses flushes immediately.
    """
    buf: list[str] = []
    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if run_mode is not None:
        env["RACING_RUN_MODE"] = run_mode
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.stdout is not None
    in_progress_line = False
    progress_text = ""
    progress_width = 0

    def _clear_progress_line() -> None:
        nonlocal progress_width
        if progress_width > 0:
            print("\r" + (" " * progress_width) + "\r", end="", flush=True)
            progress_width = 0

    for line in proc.stdout:
        buf.append(line)

        if _is_tqdm_progress_line(line):
            # Re-render tqdm updates on a single terminal line.
            progress_text = line.rstrip("\r\n")
            progress_width = max(progress_width, len(progress_text))
            print("\r" + progress_text, end="", flush=True)
            in_progress_line = True
            continue

        if in_progress_line:
            # tqdm often emits whitespace-only helper lines; suppress them.
            if not line.strip():
                continue

            # Print regular lines above the bar, then redraw the bar.
            _clear_progress_line()
            print(line, end="", flush=True)
            if progress_text:
                progress_width = len(progress_text)
                print("\r" + progress_text, end="", flush=True)
            continue

        print(line, end="", flush=True)

    if in_progress_line:
        print(flush=True)

    proc.wait()
    return proc.returncode, "".join(buf)


def _sanitize_name_fragment(value) -> str:
    """Convert arbitrary parameter values into safe experiment-name fragments."""
    s = str(value).replace(" ", "")
    s = s.replace("/", "-").replace("\\", "-")
    s = s.replace("(", "").replace(")", "")
    s = s.replace(",", "_")
    s = s.replace("[", "").replace("]", "")
    s = s.replace("{", "").replace("}", "")
    return s


def _param_hash(params: dict) -> str:
    """Stable short hash for a parameter combination."""
    canonical = "|".join(f"{k}={params[k]!r}" for k in sorted(params.keys()))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _build_exp_name(idx: int, params: dict) -> str:
    """Build a bounded experiment name safe for filesystem path limits."""
    max_len = 120
    p_hash = _param_hash(params)

    # Keep a tiny human-readable hint (first two params), then append a hash.
    hints: list[str] = []
    for k, v in list(params.items())[:2]:
        key_hint = re.sub(r"[^A-Za-z0-9]+", "", k.split(".")[-1])[:10]
        val_hint = re.sub(r"[^A-Za-z0-9._-]+", "", _sanitize_name_fragment(v))[:12]
        if key_hint and val_hint:
            hints.append(f"{key_hint}-{val_hint}")
    hint_block = "_".join(hints)

    if hint_block:
        exp_name = f"{SWEEP_ID}_run_{idx:04d}_{hint_block}_{p_hash}"
    else:
        exp_name = f"{SWEEP_ID}_run_{idx:04d}_{p_hash}"

    # Final guardrail against path component length issues.
    if len(exp_name) > max_len:
        exp_name = f"{SWEEP_ID}_run_{idx:04d}_{p_hash}"

    return exp_name


def _raise_on_invalid_override(output: str, stage: str, exp_name: str) -> None:
    """Fail fast if train/play output indicates an invalid override key/value."""
    for marker in _OVERRIDE_ERROR_MARKERS:
        if marker in output:
            tail = "\n".join(output.splitlines()[-80:])
            raise RuntimeError(
                f"[Sweep:{stage}] Invalid override detected for exp '{exp_name}': '{marker}'.\n"
                "Aborting sweep to avoid silently running wrong configurations.\n"
                f"---- {stage} output tail ----\n{tail}"
            )


def _find_best_checkpoint(exp_name: str) -> str | None:
    """Return checkpoint for the most recent run under logs/skrl/sweep/<exp_name>.

    Why this is needed:
    - train.py writes each invocation into a timestamped sub-directory under the
      same exp_name root.
    - if exp_name is reused across script invocations, stale runs accumulate.
    - we must prefer the latest sub-run, not the lexicographically first match.
    """
    base = REPO_ROOT / "logs" / "skrl" / "sweep" / exp_name

    if not base.exists():
        return None

    # Consider only timestamped run directories with checkpoints.
    run_dirs = [p for p in base.iterdir() if p.is_dir() and (p / "checkpoints").is_dir()]
    if not run_dirs:
        return None

    # Newest run first (mtime is robust even if naming changes).
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints"

        # Prefer best_agent.pt for that run.
        best = ckpt_dir / "best_agent.pt"
        if best.exists():
            return str(best)

        # Fall back to highest training step agent_*.pt in that run.
        pts = list(ckpt_dir.glob("agent_*.pt"))
        if not pts:
            continue

        def _step(path: Path) -> int:
            m = re.search(r"agent_(\d+)\.pt$", path.name)
            return int(m.group(1)) if m else -1

        return str(max(pts, key=_step))

    return None


def _parse_sweep_result(output: str) -> dict | None:
    m = re.search(
        r"\[SWEEP_RESULT\]\s+Speed=(\S+)\s+Lap=(\S+)\s+Gates=(\S+)\s+LapTime=(\S+)",
        output,
    )
    if m is None:
        return None
    return {
        "speed_kmh":  float(m.group(1)),
        "lap_pct":    float(m.group(2)),
        "mean_gates": float(m.group(3)),
        "lap_time_s": float(m.group(4)),
    }


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    param_keys = list(PARAM_GRID.keys())
    metric_keys = ["speed_kmh", "lap_pct", "mean_gates", "lap_time_s", "status"]
    header = param_keys + metric_keys
    col_w = {k: max(len(k), 8) for k in header}
    for row in rows:
        for k in header:
            col_w[k] = max(col_w[k], len(str(row.get(k, ""))))

    sep = "  ".join("-" * col_w[k] for k in header)
    hdr = "  ".join(k.ljust(col_w[k]) for k in header)
    print("\n" + "=" * len(sep))
    print("SWEEP SUMMARY")
    print("=" * len(sep))
    print(hdr)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(col_w[k]) for k in header))
    print("=" * len(sep))
    print(f"Results saved to: {RESULTS_CSV}")


def _build_param_sets() -> list[dict]:
    """Build parameter sets according to SWEEP_MODE.

    Returns:
        A list of per-run parameter dictionaries.

    Modes:
        - grid: full Cartesian product over all PARAM_GRID lists.
        - sequential: index-aligned zip over PARAM_GRID lists.
                      Example:
                        p1=[0,1], p2=[2,3] -> (0,2), (1,3)
    """
    mode = SWEEP_MODE.strip().lower()
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    if mode == "grid":
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if mode == "sequential":
        if not keys:
            return []

        expected_len = len(values[0])
        mismatched = [(k, len(v)) for k, v in PARAM_GRID.items() if len(v) != expected_len]
        if mismatched:
            details = ", ".join(f"{k}:{n}" for k, n in mismatched)
            raise ValueError(
                "SWEEP_MODE='sequential' requires all PARAM_GRID value-lists to have identical length. "
                f"Expected {expected_len}; mismatched lengths -> {details}."
            )

        return [{k: PARAM_GRID[k][i] for k in keys} for i in range(expected_len)]

    raise ValueError(
        f"Unknown SWEEP_MODE={SWEEP_MODE!r}. Supported values: 'grid', 'sequential'."
    )


def run_combo(idx: int, params: dict) -> dict:
    """Train + evaluate one parameter combination.  Returns a result row dict."""
    # Use a bounded, deterministic name to avoid OSError: filename too long.
    exp_name = _build_exp_name(idx, params)

    overrides: list[str] = []
    for k, v in params.items():
        # Support shorthand DIRECT_<TERM> keys that specify the "direct" (unscaled)
        # reward coefficient as used in mdp/rewards.py (e.g. DIRECT_VELOCITY_TOWARD_GATE).
        # These are converted to the manager-style weight field
        # `rewards.<term>.weight` by multiplying with MANAGER_SCALE so users can
        # sweep the direct magnitudes conveniently.
        if k.startswith("DIRECT_"):
            term = k[len("DIRECT_"):].lower()
            cfg_key = f"rewards.{term}.weight"
            override_val = MANAGER_SCALE * float(v)
            overrides += ["--override", f"{cfg_key}={override_val}"]
        else:
            overrides += ["--override", f"{k}={v}"]

    print(f"\n{'='*70}")
    print(f"[Sweep {idx}] {exp_name}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. Train
    # ------------------------------------------------------------------
    train_cmd = [
        PYTHON,
        str(SCRIPT_DIR / "skrl" / "train.py"),
        "--task", TASK_TRAIN,
        "--num_envs", str(NUM_ENVS),
        "--max_iterations", str(TRAIN_ITERATIONS),
        "--exp-name", exp_name,
        *overrides,
        *EXTRA_TRAIN_ARGS,
    ]
    print(f"\n[Sweep {idx}] TRAIN: {' '.join(train_cmd)}\n")
    rc, train_output = _run_stream(train_cmd, run_mode="TRAIN")
    _raise_on_invalid_override(train_output, stage="train", exp_name=exp_name)
    if rc != 0:
        print(f"[Sweep {idx}] Training failed (exit {rc})")
        return {"exp_name": exp_name, **params, "status": f"train_failed({rc})"}

    # ------------------------------------------------------------------
    # 2. Find checkpoint
    # ------------------------------------------------------------------
    ckpt = _find_best_checkpoint(exp_name)
    if ckpt is None:
        print(f"[Sweep {idx}] No checkpoint found under logs/skrl/sweep/{exp_name}/")
        return {"exp_name": exp_name, **params, "status": "no_checkpoint"}

    print(f"[Sweep {idx}] Checkpoint: {ckpt}")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    play_cmd = [
        PYTHON,
        str(SCRIPT_DIR / "skrl" / "play.py"),
        "--task", TASK_PLAY,
        "--num_envs", str(NUM_ENVS),
        "--checkpoint", ckpt,
        "--eval-steps", str(EVAL_STEPS),
        *overrides,
        *EXTRA_PLAY_ARGS,
    ]
    print(f"\n[Sweep {idx}] EVAL:  {' '.join(play_cmd)}\n")
    rc, output = _run_stream(play_cmd, run_mode="EVAL")
    _raise_on_invalid_override(output, stage="play", exp_name=exp_name)

    metrics = _parse_sweep_result(output)
    if metrics is None:
        print(f"[Sweep {idx}] No [SWEEP_RESULT] line found in play.py output (exit {rc})")
        return {"exp_name": exp_name, **params, "status": f"no_result(exit={rc})"}

    return {"exp_name": exp_name, **params, **metrics, "status": "ok"}


def main() -> None:
    args = _parse_args()
    _apply_runtime_config(args)

    param_sets = _build_param_sets()

    if args.config:
        print(f"Sweep config: {args.config}")
    print(f"Sweep mode: {SWEEP_MODE}")
    print(f"Sweep: {len(param_sets)} combinations × (train {TRAIN_ITERATIONS} iters + eval {EVAL_STEPS} steps)")
    print(f"Results CSV: {RESULTS_CSV}\n")

    results: list[dict] = []

    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for idx, params in enumerate(param_sets):
            row = run_combo(idx, params)
            results.append(row)

            # ensure all CSV fields present
            for f in CSV_FIELDS:
                row.setdefault(f, "")
            writer.writerow(row)
            fh.flush()

            _print_summary(results)

    print(f"\nSweep complete. {len(results)} runs.  Results → {RESULTS_CSV}")


if __name__ == "__main__":
    main()

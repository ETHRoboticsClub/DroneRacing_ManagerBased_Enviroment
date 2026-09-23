# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained skrl checkpoint on a hand-designed custom track.

Forks scripts/skrl/play.py.  The only behavioural difference is that **before**
the env is built we replace ``env_cfg.scene.terrain.terrain_generator`` with a
single-cell grid whose gates are exactly the ones stored in a track JSON
produced by ``track_designer.py``.

Usage::

    python scripts/custom_track/play_custom_track.py --track my_track \\
        --task Isaac-Racing-Quadcopter-Manager-v0 --checkpoint <ckpt.pt>

``--track`` accepts:

  - a bare name like ``oval``  → resolves to ``scripts/custom_track/tracks/oval.json``
  - a name with extension ``oval.json``
  - a relative or absolute path containing ``/``

The grid is forced to 1x1 with the cell sized to enclose the track + margin,
and ``num_envs`` is forced to 1.  The simulator's racing-task code (rewards,
spawn, vis, gate-passage detection) reads the same ``gate_registry``
mechanism the procedural terrain uses, so no env code changes are needed.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a skrl checkpoint on a custom hand-designed track."
)
parser.add_argument(
    "--track",
    type=str,
    required=True,
    help=(
        "Track to load. Bare name (resolved against scripts/custom_track/tracks/), "
        "filename, or path to a track JSON saved by track_designer.py."
    ),
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--cell_margin",
    type=float,
    default=2.0,
    help="Padding (m) added around the track bbox when sizing the terrain cell.",
)
parser.add_argument(
    "--min_cell_size",
    type=float,
    default=10.0,
    help="Minimum side length (m) of the terrain cell, regardless of track bbox.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Resolve the track path BEFORE Isaac launches so we fail fast on a bad input
# without paying the cost of starting the simulator.
_HERE = Path(__file__).resolve().parent
_TRACKS_DIR = _HERE / "tracks"


def _resolve_track_path(arg: str) -> Path:
    """Map a user-supplied track argument to an absolute JSON path."""
    if "/" in arg or "\\" in arg or arg.startswith(("./", "../")):
        p = Path(arg).expanduser().resolve()
    else:
        # Bare name → look in tracks/ with .json extension if missing.
        name = arg if arg.endswith(".json") else f"{arg}.json"
        p = (_TRACKS_DIR / name).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Track not found: {p}")
    return p


_TRACK_PATH = _resolve_track_path(args_cli.track)
print(f"[INFO] Custom track: {_TRACK_PATH}")

# clear out sys.argv for Hydra (must happen after our own parser is done)
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import random
import time

import numpy as np
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    def get_published_pretrained_checkpoint(*args, **kwargs):
        return None

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import Racing_Quadcopter.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from Racing_Quadcopter.terrains import racing_gates as _racing_gates
from Racing_Quadcopter.terrains import GateData, clear_gate_registry
from Racing_Quadcopter.terrains.racing_gates_cfg import MeshRacingGatesTerrainCfg

# Local import (track_io lives next to this script).
sys.path.insert(0, str(_HERE))
import track_io  # noqa: E402

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# ---------------------------------------------------------------------------
# Custom-track terrain construction
# ---------------------------------------------------------------------------
# The simulator pulls gate data from the module-level ``gate_registry`` in
# Racing_Quadcopter.terrains.racing_gates (see racing_quadcopter_env.py:131).
# To inject a custom track we replace the sub-terrain function with one that
# writes our gates into that registry.  The env's centre-offset / origin
# mapping is unchanged — see racing_quadcopter_env.py:144 — so we have to
# undo that offset when placing the trimesh frames so the runtime world-frame
# positions match the user's design coordinates exactly.

def _build_custom_terrain_function(track: dict, gate_color=(1.0, 0.3, 0.0)):
    """Return a closure with signature ``(difficulty, cfg) -> (meshes, origin)``.

    The closure captures the loaded track and emits gate trimesh frames at
    positions derived from it.  Each call also writes a ``GateData`` entry
    into the module-level registry so the env can pick it up at startup.

    This factory runs once (in ``_custom_terrain_generator_cfg``) to parse the
    JSON and interpolate the spline, since that only depends on the track, not
    on ``difficulty``/``cfg`` — the closure reuses that result on every call
    instead of recomputing it.
    """
    positions, quats_wxyz, meta = track_io.track_to_arrays(track)
    if positions.shape[0] == 0:
        raise ValueError(f"Custom track contains no gates.")

    gate_size = float(meta["gate_size"])
    bar_thickness = float(meta["bar_thickness"])
    closed = bool(meta["loop_closed"])

    # Pre-compute waypoints for the visualisers; same data structure the
    # procedural generator produces.
    if positions.shape[0] >= 2:
        waypoints, _ = track_io.interpolate_loop(positions, num_samples=300, closed=closed)
    else:
        waypoints = positions.copy()

    def custom_terrain_fn(difficulty: float, cfg) -> tuple[list, np.ndarray]:
        size = cfg.size
        meshes_list: list = []

        ground_mesh = make_plane(size, height=0.0, center_zero=False)
        meshes_list.append(ground_mesh)

        # World-frame → corner-origin frame: the env later subtracts size/2
        # (centre_offset) and adds terrain_origin, which is (0,0,0) for a 1×1
        # grid centred at world origin. So local_corner = world + size/2.
        offset = np.array([size[0] / 2.0, size[1] / 2.0, 0.0])
        local_positions = positions + offset

        for pos, quat in zip(local_positions, quats_wxyz):
            meshes_list.extend(_racing_gates.create_gate_frame(
                position=pos,
                orientation=quat,
                gate_size=gate_size,
                bar_thickness=bar_thickness,
                color=gate_color,
            ))

        # Waypoints in the same local frame so the visualiser draws the
        # spline trajectory through the gates.
        local_waypoints = waypoints + offset

        _racing_gates.gate_registry[(round(difficulty, 6), cfg.seed)] = GateData(
            positions=local_positions.copy(),
            orientations=quats_wxyz.copy(),
            gate_size=gate_size,
            bar_thickness=bar_thickness,
            waypoints=local_waypoints.copy(),
        )

        # Cell origin used by TerrainGenerator to place the patch — centre of
        # the ground mesh at z=0, matching the procedural function.
        origin = np.array([size[0] / 2.0, size[1] / 2.0, 0.0])
        return meshes_list, origin

    return custom_terrain_fn


def _custom_terrain_generator_cfg(
    track: dict,
    margin: float,
    min_size: float,
) -> TerrainGeneratorCfg:
    """Build a 1x1 ``TerrainGeneratorCfg`` whose single cell renders the track."""
    positions, _, meta = track_io.track_to_arrays(track)
    if positions.shape[0] == 0:
        raise ValueError("Custom track contains no gates.")

    # Cell is centred at world origin and must contain every gate. The half
    # side length on each axis is the larger of |min| and |max|, padded with
    # ``margin``. We use a square cell so the spawn box is symmetric.
    half_x = max(abs(positions[:, 0].min()), abs(positions[:, 0].max())) + margin
    half_y = max(abs(positions[:, 1].min()), abs(positions[:, 1].max())) + margin
    side = max(2.0 * half_x, 2.0 * half_y, float(min_size))
    size = (float(side), float(side))

    sub_cfg = MeshRacingGatesTerrainCfg(
        proportion=1.0,
        seed=0,
        # gate_size_range is required by the dataclass but unused — our
        # closure ignores it and reads the size from the loaded track.
        gate_size_range=(meta["gate_size"], meta["gate_size"] + 1e-4),
        bar_thickness=float(meta["bar_thickness"]),
    )
    sub_cfg.function = _build_custom_terrain_function(track)

    return TerrainGeneratorCfg(
        seed=0,
        size=size,
        difficulty_range=(0.5, 0.5),  # single fixed value — terrain fn ignores it
        border_width=0.5,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={"custom": sub_cfg},
    )


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent on a custom track.

    Same flow as ``scripts/skrl/play.py`` (resolve checkpoint, build env,
    run inference in a loop), with the terrain swapped for the custom track
    before the env is constructed and ``num_envs`` pinned to 1.
    """
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Load + apply the custom track. Done as early as possible so any errors
    # surface before checkpoint resolution.
    track = track_io.load_track(_TRACK_PATH)
    n_gates = len(track["gates"])
    if n_gates == 0:
        raise RuntimeError(f"Custom track {_TRACK_PATH} has no gates.")
    print(f"[INFO] Loaded {n_gates} gates from {_TRACK_PATH.name}; loop_closed={track['loop_closed']}")

    # Single env, single cell. Multi-env doesn't help here because all envs
    # would share the same terrain origin and the spawn box would put them
    # on top of each other. If the user explicitly overrides via Hydra we
    # respect it, but the default is 1.
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Replace the procedural generator with our single-cell custom one.
    # ``__post_init__`` already populated env_cfg.scene.terrain — we only swap
    # the ``terrain_generator`` field on it.
    clear_gate_registry()  # drop any leftover entries from prior runs in the same process
    env_cfg.scene.terrain.terrain_generator = _custom_terrain_generator_cfg(
        track,
        margin=float(args_cli.cell_margin),
        min_size=float(args_cli.min_cell_size),
    )

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # Push the resolved seed into both the skrl experiment config and the env
    # config so RNG-dependent behaviour (e.g. spawn jitter) is reproducible.
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # Resolve checkpoint path (same logic as scripts/skrl/play.py).
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # Build the env with our patched env_cfg (custom terrain, num_envs=1).
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_custom_track"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    elif hasattr(runner.agent, "set_mode"):
        runner.agent.set_mode("eval")
    else:
        runner.agent.enable_training_mode(enabled=False, apply_to_models=True)

    obs, _ = env.reset()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # skrl agent APIs differ across versions/algorithms in whether
            # ``act`` takes an extra positional slot (e.g. RNN hidden state)
            # before ``timestep``/``timesteps`` — some versions instead just
            # renamed the observations param to ``states`` without adding
            # one. Param *count* (not the name ``states``) tells them apart:
            # 4 params means the extra slot exists.
            _act_sig = inspect.signature(runner.agent.act).parameters
            if len(_act_sig) >= 4:
                outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # Prefer the deterministic mean action over a sampled one for
            # evaluation/playback (no exploration noise).
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

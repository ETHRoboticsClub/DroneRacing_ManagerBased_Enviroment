# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot a recorded (real-world) trajectory against a replayed (sim) trajectory.

Both inputs follow the same JSON schema described in
``replay_actions.py`` — they expose a ``trajectory`` list of states, each with
at least ``position``, ``linear_velocity`` and ``orientation`` (quaternion
``[x, y, z, w]`` — scalar LAST). Either file may also embed ``actions``;
this script only reads the trajectory part.

The produced figure shows:
  * 3D path overlay
  * per-axis position vs time
  * per-axis linear velocity vs time
  * orientation as roll / pitch / yaw vs time (degrees)
  * body rates vs time (if angular_velocity is present)

Usage::

    python scripts/deployment/visualize_trajectories.py \
        --real path/to/recording.json \
        --sim  logs/deployment/<ts>/sim_run.json \
        [--save out.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Number of body-frame orientation quivers drawn along each 3D path.
N_ORIENTATION_QUIVERS = 18
# Length of the body-z arrow as a fraction of the path's bounding-box diagonal.
QUIVER_LEN_FRAC = 0.06


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _load_trajectory(path: Path, key: str = "trajectory") -> dict[str, np.ndarray]:
    with path.open("r") as f:
        data = json.load(f)
    if key not in data or not data[key]:
        raise ValueError(f"{path} has no non-empty '{key}' field.")

    states = data[key]
    n = len(states)

    def stack(key: str, dim: int, default: float = float("nan")) -> np.ndarray:
        out = np.full((n, dim), default, dtype=np.float64)
        for i, s in enumerate(states):
            if key in s:
                vals = np.asarray(s[key], dtype=np.float64).reshape(-1)
                out[i, : vals.size] = vals
        return out

    # Time axis: prefer explicit "t"; fall back to metadata.dt or index.
    if "t" in states[0]:
        t = np.array([float(s.get("t", i)) for i, s in enumerate(states)])
    else:
        meta = data.get("metadata", {})
        dt = float(meta.get("dt", meta.get("step_dt", 1.0)))
        t = np.arange(n) * dt

    return {
        "t": t,
        "position": stack("position", 3),
        "linear_velocity": stack("linear_velocity", 3),
        "orientation": stack("orientation", 4, default=0.0),
        "angular_velocity": stack("angular_velocity", 3),
        "label": data.get("metadata", {}).get("source_input", path.name),
    }


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
def _quat_to_euler_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, w]`` quaternions (scalar last) to roll/pitch/yaw in degrees (ZYX, intrinsic)."""
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    # Roll (x)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw (z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(np.stack([roll, pitch, yaw], axis=-1))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_axis_triplet(ax_triplet, t, vals, *, label: str, color: str, ls: str = "-") -> None:
    for ax, v in zip(ax_triplet, vals.T):
        ax.plot(t, v, color=color, linestyle=ls, linewidth=1.5, label=label)


def visualize(real: dict, sim: dict | None, save: Path | None) -> None:
    """Plot ``real`` (and optionally ``sim``). If ``sim`` is None, only the real trajectory is drawn."""
    has_sim = sim is not None
    pos_sets = [real["position"]] + ([sim["position"]] if has_sim else [])

    # ----- 3D path (own figure, large) -----
    fig3d = plt.figure(figsize=(12, 10))
    ax3d = fig3d.add_subplot(111, projection="3d")

    # Shared color scale for angular-velocity magnitude.
    real_w_mag = _vec_mag(real["angular_velocity"])
    sim_w_mag = _vec_mag(sim["angular_velocity"]) if has_sim else np.zeros((0,))
    w_all = np.concatenate([w for w in (real_w_mag, sim_w_mag) if w.size and np.isfinite(w).any()])
    if w_all.size and np.isfinite(w_all).any():
        norm = Normalize(vmin=float(np.nanmin(w_all)), vmax=float(np.nanmax(w_all)) + 1e-9)
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = "viridis"

    # Path lines colored by |ω|.
    _add_colored_path(ax3d, real["position"], real_w_mag, cmap=cmap, norm=norm, linewidth=1.6, linestyle="-")
    if has_sim:
        _add_colored_path(ax3d, sim["position"], sim_w_mag, cmap=cmap, norm=norm, linewidth=2.6, linestyle="--")

    # Orientation quivers (body-z axis = thrust direction) at sampled points.
    box_diag = _bbox_diag(*pos_sets)
    arrow_len = box_diag * QUIVER_LEN_FRAC + 1e-6
    _draw_orientation_quivers(ax3d, real["position"], real["orientation"], color="tab:blue", length=arrow_len)
    if has_sim:
        _draw_orientation_quivers(ax3d, sim["position"], sim["orientation"], color="tab:orange", length=arrow_len)

    # Start markers + dummy legend handles.
    ax3d.scatter(*real["position"][0], color="tab:blue", marker="o", s=60, label="real start")
    ax3d.plot([], [], [], color="tab:blue", linewidth=1.6, label="real path (colored by |ω|)")
    if has_sim:
        ax3d.scatter(*sim["position"][0], color="tab:orange", marker="o", s=60, label="sim start")
        ax3d.plot([], [], [], color="tab:orange", linewidth=2.6, linestyle="--", label="sim path (colored by |ω|)")

    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    title = "3D trajectory  •  arrows = body-z  •  color = |ω|"
    if not has_sim:
        title += "  •  real only"
    ax3d.set_title(title)
    ax3d.legend(loc="upper right", fontsize=8)
    _equalize_3d_axes(ax3d, *pos_sets)

    # Colorbar for |ω|.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig3d.colorbar(sm, ax=ax3d, shrink=0.6, pad=0.08)
    cbar.set_label("|ω| [rad/s]")

    # ----- Time-series figure: position / velocity / orientation -----
    fig_ts, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    pos_axes, vel_axes, rpy_axes = axes[0], axes[1], axes[2]

    # Position
    _plot_axis_triplet(pos_axes, real["t"], real["position"], label="real", color="tab:blue")
    if has_sim:
        _plot_axis_triplet(pos_axes, sim["t"], sim["position"], label="sim", color="tab:orange", ls="--")
    for ax, name in zip(pos_axes, ("x", "y", "z")):
        ax.set_ylabel(f"pos {name} [m]")
        ax.grid(True, alpha=0.3)
    pos_axes[0].legend(loc="best", fontsize=8)

    # Linear velocity
    _plot_axis_triplet(vel_axes, real["t"], real["linear_velocity"], label="real", color="tab:blue")
    if has_sim:
        _plot_axis_triplet(vel_axes, sim["t"], sim["linear_velocity"], label="sim", color="tab:orange", ls="--")
    for ax, name in zip(vel_axes, ("x", "y", "z")):
        ax.set_ylabel(f"v{name} [m/s]")
        ax.grid(True, alpha=0.3)

    # Orientation (Euler)
    real_rpy = _quat_to_euler_deg(real["orientation"])
    _plot_axis_triplet(rpy_axes, real["t"], real_rpy, label="real", color="tab:blue")
    if has_sim:
        sim_rpy = _quat_to_euler_deg(sim["orientation"])
        _plot_axis_triplet(rpy_axes, sim["t"], sim_rpy, label="sim", color="tab:orange", ls="--")
    for ax, name in zip(rpy_axes, ("roll", "pitch", "yaw")):
        ax.set_ylabel(f"{name} [deg]")
        ax.set_xlabel("t [s]")
        ax.grid(True, alpha=0.3)

    fig_ts.suptitle(("Real vs replayed trajectory" if has_sim else "Real trajectory") + " — time series", fontsize=14)
    fig_ts.tight_layout(rect=(0, 0, 1, 0.97))

    # ----- Optional third figure: body rates -----
    real_w_finite = not np.isnan(real["angular_velocity"]).all()
    sim_w_finite = has_sim and not np.isnan(sim["angular_velocity"]).all()
    if real_w_finite or sim_w_finite:
        fig_w, axs_w = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        _plot_axis_triplet(axs_w, real["t"], real["angular_velocity"], label="real", color="tab:blue")
        if has_sim:
            _plot_axis_triplet(axs_w, sim["t"], sim["angular_velocity"], label="sim", color="tab:orange", ls="--")
        for ax, name in zip(axs_w, ("ωx (roll)", "ωy (pitch)", "ωz (yaw)")):
            ax.set_ylabel(f"{name} [rad/s]")
            ax.grid(True, alpha=0.3)
        axs_w[-1].set_xlabel("t [s]")
        axs_w[0].legend(loc="best", fontsize=8)
        fig_w.suptitle("Body angular rates")
        fig_w.tight_layout(rect=(0, 0, 1, 0.97))

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig3d.savefig(save.with_name(save.stem + "_3d" + save.suffix), dpi=150)
        fig_ts.savefig(save.with_name(save.stem + "_timeseries" + save.suffix), dpi=150)
        print(f"[INFO] Saved 3D figure to {save.with_name(save.stem + '_3d' + save.suffix)}")
        print(f"[INFO] Saved time-series figure to {save.with_name(save.stem + '_timeseries' + save.suffix)}")
    plt.show()


def _vec_mag(v: np.ndarray) -> np.ndarray:
    """Per-row L2 norm; rows containing any NaN return NaN."""
    if v.size == 0:
        return np.zeros((0,))
    return np.linalg.norm(v, axis=-1)


def _bbox_diag(*point_sets: np.ndarray) -> float:
    pts = np.concatenate([p for p in point_sets if p.size], axis=0)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.size == 0:
        return 1.0
    return float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)))


def _body_z_world(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotate body +Z = (0, 0, 1) into world frame using ``[x, y, z, w]`` quaternions."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    )


def _add_colored_path(ax, positions: np.ndarray, scalar: np.ndarray, *, cmap, norm, linewidth: float, linestyle: str):
    """Draw a 3D path whose color encodes a per-vertex scalar (e.g. |ω|)."""
    if positions.shape[0] < 2:
        return None
    segs = np.stack([positions[:-1], positions[1:]], axis=1)  # (N-1, 2, 3)
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth, linestyle=linestyle)
    if scalar.size == positions.shape[0]:
        seg_scalar = 0.5 * (scalar[:-1] + scalar[1:])
    else:
        seg_scalar = np.zeros(segs.shape[0])
    lc.set_array(seg_scalar)
    ax.add_collection3d(lc)
    return lc


def _draw_orientation_quivers(ax, positions: np.ndarray, quat_xyzw: np.ndarray, *, color: str, length: float) -> None:
    """Sample N_ORIENTATION_QUIVERS points along the path and draw body-z arrows."""
    n = positions.shape[0]
    if n == 0 or quat_xyzw.shape[0] != n:
        return
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(quat_xyzw).all(axis=1)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return
    take = np.linspace(0, idx.size - 1, num=min(N_ORIENTATION_QUIVERS, idx.size), dtype=int)
    sel = idx[take]
    p = positions[sel]
    bz = _body_z_world(quat_xyzw[sel])
    ax.quiver(
        p[:, 0], p[:, 1], p[:, 2],
        bz[:, 0], bz[:, 1], bz[:, 2],
        length=length, normalize=True,
        color=color, linewidth=1.0, arrow_length_ratio=0.3, alpha=0.85,
    )


def _equalize_3d_axes(ax, *point_sets: np.ndarray) -> None:
    """Make the 3D axes share an equal-scale cubic bounding box."""
    pts = np.concatenate([p for p in point_sets if p.size], axis=0)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if finite.size == 0:
        return
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    centre = 0.5 * (mins + maxs)
    half = float(max(maxs - mins).item()) * 0.55 + 1e-3
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize real vs replayed drone trajectories.")
    p.add_argument(
        "--real",
        type=str,
        default=None,
        help=(
            "Optional JSON with the recorded (real-world) trajectory. "
            "If omitted, the visualizer reads the 'reference_trajectory' embedded in --sim "
            "(written by replay_actions.py, already sliced to the replay window)."
        ),
    )
    p.add_argument(
        "--sim",
        type=str,
        default=None,
        help=(
            "Optional JSON with the replayed sim trajectory. "
            "If omitted, only the --real trajectory is plotted (no comparison)."
        ),
    )
    p.add_argument("--save", type=str, default=None, help="Optional path to save the main figure (PNG).")
    p.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="Plot states starting from this index (applied to both --real and --sim).",
    )
    p.add_argument(
        "--end_step",
        type=int,
        default=None,
        help="Stop plotting at this index (exclusive). Default: end of trajectory.",
    )
    return p.parse_args()


def _slice(traj: dict, start: int, end: int | None) -> dict:
    """Slice every per-step array in a loaded trajectory dict to ``[start, end)``."""
    out = dict(traj)
    for k in ("t", "position", "linear_velocity", "orientation", "angular_velocity"):
        if k in out and hasattr(out[k], "shape") and out[k].shape[0] > 0:
            n = out[k].shape[0]
            s = max(0, min(start, n))
            e = n if end is None else max(s, min(int(end), n))
            out[k] = out[k][s:e]
    return out


def main() -> None:
    args = _parse_args()
    if args.real is None and args.sim is None:
        raise SystemExit("Need at least one of --real or --sim.")

    sim = None
    if args.sim is not None:
        sim_path = Path(args.sim).resolve()
        sim = _load_trajectory(sim_path)
        print(f"[INFO] Sim trajectory loaded from --sim ({sim['t'].shape[0]} states).")

    if args.real is not None:
        real = _load_trajectory(Path(args.real).resolve())
        print(f"[INFO] Real trajectory loaded from --real ({real['t'].shape[0]} states).")
    else:
        # Fall back to the reference embedded in the sim file.
        real = _load_trajectory(Path(args.sim).resolve(), key="reference_trajectory")
        print(f"[INFO] Real trajectory loaded from --sim's embedded 'reference_trajectory' ({real['t'].shape[0]} states).")
    if args.start_step != 0 or args.end_step is not None:
        n_real_before = real["t"].shape[0]
        real = _slice(real, args.start_step, args.end_step)
        end_str = str(args.end_step) if args.end_step is not None else "end"
        msg = f"[INFO] Window [{args.start_step}:{end_str}]: real={real['t'].shape[0]}/{n_real_before}"
        if sim is not None:
            n_sim_before = sim["t"].shape[0]
            sim = _slice(sim, args.start_step, args.end_step)
            msg += f", sim={sim['t'].shape[0]}/{n_sim_before}"
        print(msg + " states.")
    save = Path(args.save).resolve() if args.save else None
    visualize(real, sim, save)


if __name__ == "__main__":
    main()

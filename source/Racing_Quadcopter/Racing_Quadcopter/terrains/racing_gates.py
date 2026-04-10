# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate racing gate terrains using the ``trimesh`` library.

Gate data is stored in a module-level registry so it can be retrieved by the environment
after terrain generation. This is necessary because the TerrainGenerator copies the cfg
before calling the terrain function, so writing to ``cfg`` does not propagate back.
"""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import trimesh
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from .racing_gates_cfg import MeshRacingGatesTerrainCfg


@dataclass
class GateData:
    """Gate data produced by a single sub-terrain generation call.

    Positions and orientations are in the sub-terrain's **local** frame
    (before the TerrainGenerator centres and offsets the mesh).
    """

    positions: np.ndarray  # [num_gates, 3]
    orientations: np.ndarray  # [num_gates, 4] (w, x, y, z)
    gate_size: float
    start_position: np.ndarray  # [3]
    start_tangent: np.ndarray  # [3]


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------
# Keyed by ``(difficulty, seed)`` so every unique sub-terrain cell that was
# generated can be looked up afterwards.  The environment only needs the most
# recent generation batch, so it is safe to clear this between runs.
_gate_registry: dict[tuple[float, int | None], GateData] = {}


def get_gate_registry() -> dict[tuple[float, int | None], GateData]:
    """Return a reference to the module-level gate data registry."""
    return _gate_registry


def clear_gate_registry() -> None:
    """Clear all stored gate data (call between training runs if needed)."""
    _gate_registry.clear()


def racing_gates_terrain(
    difficulty: float, cfg: "MeshRacingGatesTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat terrain with racing gates placed along a generated trajectory.

    The gates are rectangular frames made of 4 bars (top, bottom, left, right).
    The trajectory is generated procedurally based on the configuration.

    Gate coordinate system:
    - x-axis: right (horizontal)
    - y-axis: forward (passage direction, through the gate)
    - z-axis: up (vertical)

    Gate data is written to :func:`get_gate_registry` so the environment can
    retrieve it after terrain generation.

    Args:
        difficulty: The difficulty of the terrain (0 to 1). Higher difficulty means
            smaller gate openings and more pronounced height variation.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing:
            - List of trimesh objects (ground plane + gate meshes)
            - Origin of the terrain (centre of ground plane at ground level)
    """
    meshes_list = []

    # Create ground plane
    ground_mesh = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_mesh)

    # Generate trajectory waypoints
    waypoints, tangents = _generate_racing_trajectory(
        size=cfg.size,
        num_waypoints=cfg.num_waypoints,
        trajectory_type=cfg.trajectory_type,
        difficulty=difficulty,
        height_range=cfg.gate_height_range,
        rng=np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng(),
    )

    # Compute gate size based on difficulty
    gate_size = cfg.gate_size_range[1] - difficulty * (cfg.gate_size_range[1] - cfg.gate_size_range[0])

    # Create RNG for gate placement (reuse seed for consistency)
    placement_rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()

    # Generate gates along trajectory
    gate_positions, gate_orientations = _place_gates_along_trajectory(
        waypoints=waypoints,
        tangents=tangents,
        gate_spacing=cfg.gate_spacing,
        size=cfg.size,
        gate_spacing_range=cfg.gate_spacing_range,
        min_gates=cfg.min_gates,
        rng=placement_rng,
    )

    # Store gate data in module-level registry
    _gate_registry[(round(difficulty, 6), cfg.seed)] = GateData(
        positions=gate_positions.copy(),
        orientations=gate_orientations.copy(),
        gate_size=gate_size,
        start_position=waypoints[0].copy(),
        start_tangent=tangents[0].copy(),
    )

    # Create gate meshes
    for pos, quat in zip(gate_positions, gate_orientations):
        gate_meshes = _create_gate_frame(
            position=pos,
            orientation=quat,
            gate_size=gate_size,
            bar_thickness=cfg.bar_thickness,
            color=cfg.gate_color,
        )
        meshes_list.extend(gate_meshes)

    # Origin is at the centre of the terrain at ground level
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])

    return meshes_list, origin


def _generate_racing_trajectory(
    size: tuple[float, float],
    num_waypoints: int,
    trajectory_type: str,
    difficulty: float,
    height_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate waypoints and tangents for a racing trajectory.
    
    Args:
        size: Size of the terrain (width, length)
        num_waypoints: Number of waypoints to generate
        trajectory_type: Type of trajectory ("figure8", "oval", "random")
        difficulty: Difficulty level (0 to 1)
        height_range: Range of heights for gates (min, max)
        rng: Random number generator
        
    Returns:
        waypoints: [N, 3] array of waypoint positions
        tangents: [N, 3] array of normalized tangent vectors
    """
    center_x = size[0] / 2.0
    center_y = size[1] / 2.0
    
    # Scale radius to fit within terrain, leaving margin
    margin = 1.0  # meters from edge
    max_radius_x = (size[0] / 2.0 - margin) * 0.8
    max_radius_y = (size[1] / 2.0 - margin) * 0.8
    
    if trajectory_type == "figure8":
        waypoints, tangents = _generate_figure8(
            center_x, center_y, max_radius_x, max_radius_y, 
            num_waypoints, height_range, difficulty, rng
        )
    elif trajectory_type == "oval":
        waypoints, tangents = _generate_oval(
            center_x, center_y, max_radius_x, max_radius_y,
            num_waypoints, height_range, difficulty, rng
        )
    else:  # random
        waypoints, tangents = _generate_random_track(
            center_x, center_y, max_radius_x, max_radius_y,
            num_waypoints, height_range, difficulty, rng
        )
    
    return waypoints, tangents


def _generate_figure8(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a figure-8 trajectory."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Figure-8 parametric equations (lemniscate-like)
    scale = 0.7
    x = cx + rx * scale * np.sin(t)
    y = cy + ry * scale * np.sin(t) * np.cos(t)
    
    # Height variation based on difficulty
    height_var = (height_range[1] - height_range[0]) * difficulty * 0.5
    base_height = (height_range[0] + height_range[1]) / 2.0
    z = base_height + height_var * np.sin(2 * t) + rng.uniform(-0.1, 0.1, n) * difficulty
    z = np.clip(z, height_range[0], height_range[1])
    
    waypoints = np.stack([x, y, z], axis=1)
    
    # Compute tangents
    tangents = np.zeros_like(waypoints)
    for i in range(n):
        next_idx = (i + 1) % n
        tangents[i] = waypoints[next_idx] - waypoints[i]
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
    
    return waypoints, tangents


def _generate_oval(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an oval/ellipse trajectory."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    x = cx + rx * 0.8 * np.cos(t)
    y = cy + ry * 0.8 * np.sin(t)
    
    # Height variation
    height_var = (height_range[1] - height_range[0]) * difficulty * 0.3
    base_height = (height_range[0] + height_range[1]) / 2.0
    z = base_height + height_var * np.sin(2 * t)
    z = np.clip(z, height_range[0], height_range[1])
    
    waypoints = np.stack([x, y, z], axis=1)
    
    # Compute tangents
    tangents = np.zeros_like(waypoints)
    for i in range(n):
        next_idx = (i + 1) % n
        tangents[i] = waypoints[next_idx] - waypoints[i]
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
    
    return waypoints, tangents


def _generate_random_track(
    cx: float, cy: float, rx: float, ry: float,
    n: int, height_range: tuple[float, float], difficulty: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random smooth track using control points and spline interpolation.
    
    Enhanced version that creates more complex trajectories with:
    - More control points for more turns
    - Greater radius variations for complex shapes
    - More dramatic height changes
    - Additional noise for unpredictability
    """
    # Generate MORE control points for complex trajectories (8-16 instead of 6-10)
    num_control = max(8, int(12 + difficulty * 8))
    
    t_ctrl = np.linspace(0, 2 * np.pi, num_control, endpoint=False)
    
    # Much more dramatic radius perturbations for complex shapes
    r_perturb_base = 1.0 + (rng.uniform(-0.6, 0.6, num_control) * (0.5 + difficulty * 0.5))
    
    # Add secondary oscillations to create more complex shapes
    secondary_freq = rng.uniform(1.5, 3.5)  # Random secondary frequency
    secondary_amp = rng.uniform(0.2, 0.4)   # Secondary amplitude
    r_perturb_secondary = 1.0 + secondary_amp * np.sin(secondary_freq * t_ctrl)
    r_perturb = r_perturb_base * r_perturb_secondary
    
    # Create base positions with larger utilization of available space
    ctrl_x = cx + rx * 0.85 * np.cos(t_ctrl) * r_perturb  # Use 85% instead of 70%
    ctrl_y = cy + ry * 0.85 * np.sin(t_ctrl) * r_perturb
    
    # Much more dramatic height variations with oscillations
    base_heights = rng.uniform(height_range[0], height_range[1], num_control)
    
    # Add sinusoidal height variations for roller-coaster effect  
    height_freq1 = rng.uniform(1.0, 3.0)  # Primary frequency
    height_freq2 = rng.uniform(4.0, 6.0)  # Secondary frequency
    height_amp1 = (height_range[1] - height_range[0]) * 0.3  # 30% of height range
    height_amp2 = (height_range[1] - height_range[0]) * 0.15  # 15% of height range
    
    height_variation = (
        height_amp1 * np.sin(height_freq1 * t_ctrl + rng.uniform(0, 2*np.pi)) +
        height_amp2 * np.sin(height_freq2 * t_ctrl + rng.uniform(0, 2*np.pi))
    )
    
    ctrl_z = np.clip(base_heights + height_variation, height_range[0], height_range[1])
    
    # Add random spiral component for 3D complexity
    spiral_strength = rng.uniform(0.1, 0.3)
    spiral_offset_x = spiral_strength * rx * np.sin(3 * t_ctrl)
    spiral_offset_y = spiral_strength * ry * np.cos(3 * t_ctrl)
    
    ctrl_x += spiral_offset_x
    ctrl_y += spiral_offset_y
    
    # Interpolate to get smooth trajectory
    from scipy.interpolate import CubicSpline
    
    # Close the loop by appending first point
    t_ctrl_ext = np.append(t_ctrl, 2 * np.pi)
    ctrl_x_ext = np.append(ctrl_x, ctrl_x[0])
    ctrl_y_ext = np.append(ctrl_y, ctrl_y[0])
    ctrl_z_ext = np.append(ctrl_z, ctrl_z[0])
    
    cs_x = CubicSpline(t_ctrl_ext, ctrl_x_ext, bc_type='periodic')
    cs_y = CubicSpline(t_ctrl_ext, ctrl_y_ext, bc_type='periodic')
    cs_z = CubicSpline(t_ctrl_ext, ctrl_z_ext, bc_type='periodic')
    
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cs_x(t)
    y = cs_y(t)
    z = np.clip(cs_z(t), height_range[0], height_range[1])
    
    waypoints = np.stack([x, y, z], axis=1)
    
    # Compute tangents from spline derivatives
    dx = cs_x(t, 1)
    dy = cs_y(t, 1)
    dz = cs_z(t, 1)
    tangents = np.stack([dx, dy, dz], axis=1)
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
    
    return waypoints, tangents


def _place_gates_along_trajectory(
    waypoints: np.ndarray,
    tangents: np.ndarray,
    gate_spacing: float,
    size: tuple[float, float],
    gate_spacing_range: tuple[float, float] | None = None,
    min_gates: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place gates at intervals along the trajectory.
    
    Args:
        waypoints: [N, 3] trajectory waypoints
        tangents: [N, 3] trajectory tangents
        gate_spacing: Base distance between gates (used if gate_spacing_range is None)
        size: Terrain size for bounds checking
        gate_spacing_range: If provided, randomize spacing between (min, max)
        min_gates: Minimum number of gates to place
        
    Returns:
        gate_positions: [M, 3] gate center positions
        gate_orientations: [M, 4] gate orientations as quaternions (w, x, y, z)
            All orientations are upright (perpendicular to ground)
    """
    # Compute cumulative arc length
    n = waypoints.shape[0]
    segment_lengths = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
    # Add last segment (closing the loop)
    segment_lengths = np.append(segment_lengths, np.linalg.norm(waypoints[0] - waypoints[-1]))
    cumulative_arc = np.zeros(n + 1)
    cumulative_arc[1:] = np.cumsum(segment_lengths)
    total_arc = cumulative_arc[-1]
    
    # Determine number of gates
    if gate_spacing_range is not None:
        avg_spacing = (gate_spacing_range[0] + gate_spacing_range[1]) / 2.0
        num_gates = max(min_gates, int(total_arc / avg_spacing))
    else:
        num_gates = max(min_gates, int(total_arc / gate_spacing))
    
    gate_positions = []
    gate_orientations = []
    
    # Use provided RNG or create default one
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate random spacings if range is provided
    if gate_spacing_range is not None:
        spacings = rng.uniform(gate_spacing_range[0], gate_spacing_range[1], num_gates)
    else:
        spacings = np.full(num_gates, gate_spacing)
    
    current_arc = 0.0
    for i in range(num_gates):
        target_arc = current_arc
        if target_arc >= total_arc:
            break
            
        # Find waypoint segment
        idx = np.searchsorted(cumulative_arc[:-1], target_arc)
        idx = np.clip(idx, 1, n) - 1
        
        # Interpolate position
        if idx < n - 1:
            prev_arc = cumulative_arc[idx]
            next_arc = cumulative_arc[idx + 1]
            t = (target_arc - prev_arc) / (next_arc - prev_arc + 1e-8)
            pos = (1 - t) * waypoints[idx] + t * waypoints[idx + 1]
            tangent = (1 - t) * tangents[idx] + t * tangents[idx + 1]
        else:
            pos = waypoints[idx]
            tangent = tangents[idx]
        
        # Check bounds
        margin = 0.5
        if pos[0] < margin or pos[0] > size[0] - margin:
            current_arc += spacings[i]
            continue
        if pos[1] < margin or pos[1] > size[1] - margin:
            current_arc += spacings[i]
            continue
        
        # Compute UPRIGHT gate orientation (perpendicular to ground)
        # Gate frame: x=right, y=forward (through), z=up (always vertical)
        # Project tangent onto horizontal plane
        tangent_horizontal = tangent.copy()
        tangent_horizontal[2] = 0.0  # Remove vertical component
        tangent_horizontal = tangent_horizontal / (np.linalg.norm(tangent_horizontal) + 1e-8)
        
        z_axis = np.array([0.0, 0.0, 1.0])  # Always up
        right = np.cross(tangent_horizontal, z_axis)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Build rotation matrix [right, tangent_horizontal, z_axis] as columns
        rot_mat = np.stack([right, tangent_horizontal, z_axis], axis=1)
        
        # Convert to quaternion
        rot = tf.Rotation.from_matrix(rot_mat)
        quat = rot.as_quat()  # Returns (x, y, z, w)
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w, x, y, z)
        
        gate_positions.append(pos)
        gate_orientations.append(quat)
        current_arc += spacings[i]
    
    return np.array(gate_positions), np.array(gate_orientations)


def _create_gate_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    gate_size: float,
    bar_thickness: float,
    color: tuple[float, float, float] | None = None,
) -> list[trimesh.Trimesh]:
    """Create a rectangular gate frame with 4 bars.
    
    Args:
        position: [3] gate center position
        orientation: [4] quaternion (w, x, y, z)
        gate_size: Size of the gate opening
        bar_thickness: Thickness of the bars
        color: RGB color tuple (0-1 range)
        
    Returns:
        List of 4 trimesh objects for the gate bars
    """
    half_size = gate_size / 2.0
    meshes = []
    
    # Convert quaternion to rotation matrix
    quat_scipy = [orientation[1], orientation[2], orientation[3], orientation[0]]  # (x, y, z, w)
    rot = tf.Rotation.from_quat(quat_scipy)
    rot_mat = rot.as_matrix()
    
    # Bar specifications in local gate frame
    # Gate frame: x=right, y=forward, z=up
    bar_specs = [
        # (name, local_offset, dimensions)
        ("top", np.array([0, 0, half_size]), np.array([gate_size + bar_thickness, bar_thickness, bar_thickness])),
        ("bottom", np.array([0, 0, -half_size]), np.array([gate_size + bar_thickness, bar_thickness, bar_thickness])),
        ("left", np.array([-half_size, 0, 0]), np.array([bar_thickness, bar_thickness, gate_size + bar_thickness])),
        ("right", np.array([half_size, 0, 0]), np.array([bar_thickness, bar_thickness, gate_size + bar_thickness])),
    ]
    
    for name, local_offset, dims in bar_specs:
        # Create box
        bar_mesh = trimesh.creation.box(dims)
        
        # Transform to world frame
        world_offset = rot_mat @ local_offset
        bar_pos = position + world_offset
        
        # Build transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = bar_pos
        
        bar_mesh.apply_transform(transform)
        
        # Set color if specified
        if color is not None:
            bar_mesh.visual.face_colors = [int(c * 255) for c in color] + [255]
        
        meshes.append(bar_mesh)
    
    return meshes

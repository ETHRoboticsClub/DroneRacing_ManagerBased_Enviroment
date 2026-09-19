# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination callables for the hovering / waypoint-following env."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.terminations import *  # noqa: F401, F403


def crashed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when the drone leaves the legal flight envelope. Shape [N]."""
    z = env._robot.data.root_pos_w[:, 2]
    out_of_bounds = (z < env.cfg.min_height) | (z > env.cfg.max_height)
    upside = env._robot.data.projected_gravity_b[:, 2] > env.cfg.upside_down_threshold
    return out_of_bounds | upside


def all_collected(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when every waypoint of the env's track has been collected. Shape [N].

    ``waypoint_utils.update_waypoint_state`` sets this on the step the final
    waypoint is collected.
    """
    return env._all_collected_buf
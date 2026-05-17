# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific MDP utilities for the hovering / waypoint-following env."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .events import *        # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *       # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .waypoint_utils import update_waypoint_state  # noqa: F401


def scale_dataclass_attributes(scaling_factor: float):
    """Decorator: multiply all numeric class attributes by ``scaling_factor``.

    Mirrors the racing package — keeps human-readable weight values in the
    cfg while the RewardManager sees the scaled values, so the SKRL
    ``rewards_shaper_scale = 0.01`` reproduces the cfg-level magnitudes.
    """

    def decorator(cls):
        for field_name in cls.__annotations__:
            value = getattr(cls, field_name, None)
            if isinstance(value, (int, float)):
                setattr(cls, field_name, scaling_factor * value)
        return cls

    return decorator
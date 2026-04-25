# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific MDP utilities (re-export Isaac Lab mdp + racing quadcopter)."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import *       # noqa: F401, F403
from .curriculums import *   # noqa: F401, F403
from .events import *        # noqa: F401, F403
from .gate_utils import *    # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *       # noqa: F401, F403
from .stats import *         # noqa: F401, F403
from .terminations import *  # noqa: F401, F403


def scale_dataclass_attributes(scaling_factor: float):
    def decorator(cls):
        for field_name in cls.__annotations__:
            value = getattr(cls, field_name, None)
            if isinstance(value, (int, float)):
                setattr(cls, field_name, scaling_factor * value)
        return cls

    return decorator

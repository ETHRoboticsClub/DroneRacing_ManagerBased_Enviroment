# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms shared by the racing and hovering tasks (the AI-GP snapshots carry their own copies).

Re-exports Isaac Lab's ``isaaclab.envs.mdp`` plus the quadcopter terms. Files are grouped by
manager (actions, observations, rewards, events, terminations, curriculums); within a file,
gate-racing terms come first and waypoint (hovering) terms follow in their own section.
Helpers: ``gate_utils`` / ``stats`` (gate-racing state), ``waypoint_utils`` / ``waypoint_io``
(waypoint state), ``route_progress``, ``shared_features`` (action noise + speed limits),
``noise`` (observation noise models), ``adaptive_curriculums``.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import *               # noqa: F401, F403
from .adaptive_curriculums import *  # noqa: F401, F403
from .curriculums import *           # noqa: F401, F403
from .events import *                # noqa: F401, F403
from .gate_utils import *            # noqa: F401, F403
from .noise import *                 # noqa: F401, F403
from .observations import *          # noqa: F401, F403
from .rewards import *               # noqa: F401, F403
from .route_progress import *        # noqa: F401, F403
from .shared_features import *       # noqa: F401, F403
from .stats import *                 # noqa: F401, F403
from .terminations import *          # noqa: F401, F403
from .waypoint_io import *           # noqa: F401, F403
from .waypoint_utils import *        # noqa: F401, F403


# Legacy helper from the old x100 reward-weight convention. Imported by the racing and hovering
# cfgs but not applied anywhere (weights are direct values now).
def scale_dataclass_attributes(scaling_factor: float):
    def decorator(cls):
        for field_name in cls.__annotations__:
            value = getattr(cls, field_name, None)
            if isinstance(value, (int, float)):
                setattr(cls, field_name, scaling_factor * value)
        return cls

    return decorator

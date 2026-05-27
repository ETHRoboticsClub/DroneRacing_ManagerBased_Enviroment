# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hovering / waypoint-following manager-based task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Hovering-Quadcopter-Manager-v0",
    entry_point=f"{__name__}.hovering_quadcopter_env:HoveringQuadcopterRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hovering_quadcopter_env_cfg:HoveringQuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HoveringQuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Hovering-Quadcopter-Manager-Play-v0",
    entry_point=f"{__name__}.hovering_quadcopter_env:HoveringQuadcopterRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hovering_quadcopter_env_cfg:HoveringQuadcopterEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HoveringQuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Cardinal-spawn evaluation variant — mirrors test_controller_offline.py:
# exactly 4 cells, drone spawned 1 m E/W/N/S of a fixed goal at [0, 0, 1].
gym.register(
    id="Isaac-Hovering-Eval-Quadcopter-Manager-v0",
    entry_point=f"{__name__}.hovering_quadcopter_env:HoveringQuadcopterRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hovering_quadcopter_env_cfg:HoveringEvalQuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HoveringQuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
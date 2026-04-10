# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Racing quadcopter manager-based task (standalone)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Racing-Quadcopter-Manager-v0",
    entry_point=f"{__name__}.racing_quadcopter_env:RacingQuadcopterRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.racing_quadcopter_env_cfg:RacingQuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RacingQuadcopterPPORunnerCfg",
    },
)

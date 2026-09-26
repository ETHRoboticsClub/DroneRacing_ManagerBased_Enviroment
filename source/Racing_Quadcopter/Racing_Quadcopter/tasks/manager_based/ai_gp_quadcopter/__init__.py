# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AI-GP (AI Grand Prix) task: two FROZEN, self-contained snapshots.

* ``june11/``       — the version we competed with (commit f6828d9). Default.
* ``experimental/`` — the post-June experiments (snapshot of 2026-09-26).

Neither snapshot imports anything from the rest of ``Racing_Quadcopter`` or from the other
snapshot, so changes to racing, hovering or shared code cannot affect them. Do not modify
either unless a change to that snapshot is explicitly intended. See ``README.md``.

``AI_GP_VARIANT`` (``june11`` | ``experimental``) selects which snapshot the task IDs run.
Entry points are strings, so only the selected snapshot is ever imported.
"""

import os

import gymnasium as gym

AI_GP_VARIANT_ENV_VAR = "AI_GP_VARIANT"
AI_GP_VARIANT = os.environ.get(AI_GP_VARIANT_ENV_VAR, "june11")
if AI_GP_VARIANT not in ("june11", "experimental"):
    # Raised (not asserted) so ``python -O`` cannot silently fall through to a default.
    raise ValueError(f"Invalid {AI_GP_VARIANT_ENV_VAR}={AI_GP_VARIANT!r}. Allowed: 'june11', 'experimental'.")

_PKG = f"{__name__}.{AI_GP_VARIANT}"
if AI_GP_VARIANT == "june11":
    _MLP_RUNNER = f"{_PKG}.agents.rsl_rl_ppo_cfg:RacingQuadcopterPPORunnerCfg"         # June 11
    _RNN_RUNNER = f"{_PKG}.agents.rsl_rl_ppo_rnn_cfg:CompetitionRecurrentPPORunnerCfg"  # post-June addition
else:
    _MLP_RUNNER = f"{_PKG}.agents.rsl_rl_ppo_competition_cfg:CompetitionPPORunnerCfg"
    _RNN_RUNNER = f"{_PKG}.agents.rsl_rl_ppo_competition_cfg:CompetitionRecurrentPPORunnerCfg"

for _id, _cfg_cls, _runner in (
    ("Isaac-AI-GP-Quadcopter-v0", "CompetitionEnvCfg", _MLP_RUNNER),
    ("Isaac-AI-GP-Quadcopter-Play-v0", "CompetitionEnvCfg_PLAY", _MLP_RUNNER),
    # GRU policy (rsl_rl only; skrl uses the MLP config for these IDs too).
    ("Isaac-AI-GP-Quadcopter-RNN-v0", "CompetitionEnvCfg", _RNN_RUNNER),
    ("Isaac-AI-GP-Quadcopter-RNN-Play-v0", "CompetitionEnvCfg_PLAY", _RNN_RUNNER),
):
    gym.register(
        id=_id,
        entry_point=f"{_PKG}.competition_env:RacingQuadcopterRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_PKG}.competition_env_cfg:{_cfg_cls}",
            "rsl_rl_cfg_entry_point": _runner,
            "skrl_cfg_entry_point": f"{_PKG}.agents:skrl_ppo_competition_cfg.yaml",
        },
    )

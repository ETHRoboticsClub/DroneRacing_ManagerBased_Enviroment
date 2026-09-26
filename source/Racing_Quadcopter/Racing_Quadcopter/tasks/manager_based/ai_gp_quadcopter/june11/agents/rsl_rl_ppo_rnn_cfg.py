# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRU runner for ``Isaac-Racing-Quadcopter-Competition-RNN[-Play]-v0`` on the June 11 env.

POST-JUNE ADDITION: the RNN experiment was introduced on Jun 29 (b1dd396) and did not exist on
June 11. It is kept so the experiment can run on the frozen June 11 environment; everything
else in this package is the June snapshot. Hyperparameters match the Jun 29+ recurrent runner.
``experiment_name`` is distinct so these runs never mix with the experimental variant's logs.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)


@configclass
class CompetitionRecurrentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 10000
    save_interval = 100
    experiment_name = "ai_gp_jun11_gru"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[200, 100],
        activation="tanh",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[200, 100],
        activation="tanh",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

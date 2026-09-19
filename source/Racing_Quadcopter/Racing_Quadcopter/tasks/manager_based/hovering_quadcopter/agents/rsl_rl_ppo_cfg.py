# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)


@configclass
class HoveringQuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000
    save_interval = 500
    experiment_name = "hovering_quadcopter_manager"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HoveringQuadcopterRecurrentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """GRU PPO variant of the hovering policy."""

    num_steps_per_env = 64
    max_iterations = 10000
    save_interval = 500
    experiment_name = "hovering_quadcopter_manager_gru"
    # The actor uses deployable mocap/IMU observations; the critic additionally
    # receives simulator linear velocity through the asymmetric ``critic`` group.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

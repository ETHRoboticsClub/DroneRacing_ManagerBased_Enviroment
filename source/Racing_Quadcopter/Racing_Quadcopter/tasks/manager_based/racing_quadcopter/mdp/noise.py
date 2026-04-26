# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation-noise models for racing quadcopter."""

from __future__ import annotations

import torch

import dataclasses

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBias, NoiseModelWithAdditiveBiasCfg


class NoiseModelWithAdditiveBiasAndDropout(NoiseModelWithAdditiveBias):
    """Gaussian noise model with additive per-episode bias and optional dropout."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        cfg = self._noise_model_cfg
        if not cfg.enable:
            return data

        out = super().__call__(data)

        if cfg.keep_prob < 1.0:
            mask = (torch.rand_like(out) < cfg.keep_prob).to(dtype=out.dtype)
            out = out * mask

        return out


@configclass
class NoiseModelWithAdditiveBiasAndDropoutCfg(NoiseModelWithAdditiveBiasCfg):
    """Config for :class:`NoiseModelWithAdditiveBiasAndDropout`."""

    class_type: type = NoiseModelWithAdditiveBiasAndDropout

    enable: bool = True
    dropout: float = 0.0
    
    keep_prob: float = dataclasses.field(init=False)
    
    def __post_init__(self):
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"Dropout must be in the range [0.0, 1.0], got {self.dropout}.")
        self.keep_prob = 1.0 - self.dropout


def drone_noise_model_factory(
    std_noise: float | torch.Tensor,
    std_bias: float | torch.Tensor,
    enable: bool = True,
    dropout: float = 0.0,
    **kwargs,
    ) -> NoiseModelWithAdditiveBiasAndDropoutCfg:
    """
    Factory function for eaasily creating an :class:`NoiseModelWithAdditiveBiasAndDropoutCfg`.
    - std_noise: standard deviation of the per-step Gaussian 0-mean noise to be added to the observation.
    - std_bias: standard deviation of the per-episode bias to be added to the observation.
    """
    return NoiseModelWithAdditiveBiasAndDropoutCfg(
        noise_cfg=GaussianNoiseCfg(
            mean=0.0,
            std=std_noise,
            operation="add",
        ),
        bias_noise_cfg=GaussianNoiseCfg(
            mean=0.0,
            std=std_bias,
            operation="abs",
        ),
        enable=enable,
        dropout=dropout,
        **kwargs,
    )
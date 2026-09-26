"""Shared observation-noise models for manager-based quadcopter tasks."""

from __future__ import annotations

import dataclasses

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBias, NoiseModelWithAdditiveBiasCfg


class NoiseModelWithAdditiveBiasAndDropout(NoiseModelWithAdditiveBias):
    """Gaussian noise with additive per-episode bias and optional dropout."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        cfg = self._noise_model_cfg
        if not cfg.enable:
            return data
        out = super().__call__(data)
        if cfg.keep_prob < 1.0:
            out = out * (torch.rand_like(out) < cfg.keep_prob).to(dtype=out.dtype)
        return out


@configclass
class NoiseModelWithAdditiveBiasAndDropoutCfg(NoiseModelWithAdditiveBiasCfg):
    """Configuration for additive Gaussian noise, bias, and dropout."""

    class_type: type = NoiseModelWithAdditiveBiasAndDropout
    enable: bool = False
    dropout: float = 0.0
    keep_prob: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"Dropout must be in the range [0.0, 1.0], got {self.dropout}.")
        self.keep_prob = 1.0 - self.dropout


def drone_noise_model_factory(
    std_noise: float | torch.Tensor,
    std_bias: float | torch.Tensor,
    enable: bool = False,
    dropout: float = 0.0,
    **kwargs,
) -> NoiseModelWithAdditiveBiasAndDropoutCfg:
    """Build the common Gaussian observation-noise model."""
    return NoiseModelWithAdditiveBiasAndDropoutCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=std_noise, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=std_bias, operation="abs"),
        enable=enable,
        dropout=dropout,
        **kwargs,
    )

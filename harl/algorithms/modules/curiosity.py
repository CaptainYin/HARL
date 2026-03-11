"""Intrinsic Curiosity Module (simple forward dynamics model).

Given current state s_t (observation) and action a_t, predicts next state s_{t+1}.
The per-transition intrinsic reward is the prediction error (MSE) between
predicted and true next observation. This module is intentionally lightweight
and generic so it can be plugged into on-policy algorithms such as IPPO.

Design choices / simplifications:
- Observations are flattened before being fed to the model (works for vector
  obs; for images a separate encoder should be used — extend as needed).
- Discrete actions are one-hot encoded; continuous actions are used directly.
- Forward model: simple MLP with ReLU activations (sizes configurable).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenForwardModel(nn.Module):
    """Forward dynamics model f(s_t, a_t) -> s_hat_{t+1} for flattened observations."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers = []
        last = obs_dim + act_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(last, hs))
            layers.append(nn.ReLU())
            last = hs
        layers.append(nn.Linear(last, obs_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.model(x)


class CuriosityModule(nn.Module):
    """Wrapper providing intrinsic reward computation.

    Usage:
        pred_next, per_sample_mse, loss = module(obs_t, actions, obs_tp1)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(obs_dim, int):
            raise TypeError(f"obs_dim must be int (flattened size), got {type(obs_dim)}: {obs_dim}")
        if not isinstance(action_dim, int):
            raise TypeError(f"action_dim must be int (flattened size), got {type(action_dim)}: {action_dim}")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Use name not colliding with nn.Module.forward to avoid accidental recursion / shadowing
        self.fwd_model = FlattenForwardModel(obs_dim, action_dim, hidden_sizes)
        if device is not None:
            self.to(device)

    @staticmethod
    def one_hot(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
        oh = F.one_hot(actions.long().view(-1), num_classes=num_actions).float()
        return oh

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor, discrete: bool = False, num_actions: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute predicted next obs and intrinsic errors.

        Parameters
        ----------
        obs : (B, obs_dim_flat)
        actions : (B, A) or (B,) if discrete indices
        next_obs : (B, obs_dim_flat)
        discrete : whether action is discrete (then we one-hot encode)
        num_actions : required if discrete
        Returns
        -------
        pred_next : predicted next observation
        per_sample_mse : (B,) mean squared error per sample
        loss : scalar MSE loss (mean over batch)
        """
        if discrete:
            assert num_actions is not None, "num_actions must be provided for discrete actions"
            act_in = self.one_hot(actions, num_actions)
        else:
            act_in = actions
        pred_next = self.fwd_model(obs, act_in)
        # per-sample MSE across obs dims
        diff = pred_next - next_obs
        per_sample_mse = (diff.pow(2).mean(dim=-1))
        loss = per_sample_mse.mean()
        return pred_next, per_sample_mse, loss

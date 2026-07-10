"""Policy networks: small CNN over the stacked binary masks + scalar MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _ortho(layer: nn.Module, std: float = 2 ** 0.5) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MaskEncoder(nn.Module):
    """(B, stack, H, W) binary in [0,1] -> (B, 256). Sized for 120x213 input."""

    def __init__(self, stack: int, height: int, width: int):
        super().__init__()
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(stack, 16, 8, stride=4)), nn.ReLU(),
            _ortho(nn.Conv2d(16, 32, 4, stride=2)), nn.ReLU(),
            _ortho(nn.Conv2d(32, 32, 3, stride=2)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, stack, height, width)).shape[1]
        self.fc = nn.Sequential(_ortho(nn.Linear(n_flat, 256)), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class AimPolicy(nn.Module):
    """Gaussian policy over (dyaw, dpitch) in [-1, 1] (env scales to deg/tick)."""

    def __init__(self, stack: int, height: int, width: int, n_scalars: int):
        super().__init__()
        self.encoder = MaskEncoder(stack, height, width)
        self.scalar_fc = nn.Sequential(_ortho(nn.Linear(n_scalars, 32)), nn.ReLU())
        self.actor_mean = _ortho(nn.Linear(256 + 32, 2), std=0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, 2), -0.3))
        self.critic = _ortho(nn.Linear(256 + 32, 1), std=1.0)

    def features(self, masks: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder(masks), self.scalar_fc(scalars)], dim=1)

    def value(self, masks: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(masks, scalars)).squeeze(-1)

    def act(self, masks: torch.Tensor, scalars: torch.Tensor,
            action: torch.Tensor | None = None, deterministic: bool = False):
        feat = self.features(masks, scalars)
        mean = self.actor_mean(feat)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(feat).squeeze(-1)
        return action, logprob, entropy, value


class SwingPolicy(nn.Module):
    """Categorical policy over {0: hold, 1: attack}."""

    def __init__(self, stack: int, height: int, width: int, n_scalars: int):
        super().__init__()
        self.encoder = MaskEncoder(stack, height, width)
        self.scalar_fc = nn.Sequential(_ortho(nn.Linear(n_scalars, 32)), nn.ReLU())
        self.actor = _ortho(nn.Linear(256 + 32, 2), std=0.01)
        self.critic = _ortho(nn.Linear(256 + 32, 1), std=1.0)

    def features(self, masks: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encoder(masks), self.scalar_fc(scalars)], dim=1)

    def value(self, masks: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(masks, scalars)).squeeze(-1)

    def act(self, masks: torch.Tensor, scalars: torch.Tensor,
            action: torch.Tensor | None = None, deterministic: bool = False):
        feat = self.features(masks, scalars)
        dist = Categorical(logits=self.actor(feat))
        if action is None:
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(feat).squeeze(-1)
        return action, logprob, entropy, value

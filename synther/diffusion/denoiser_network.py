# Denoiser networks for diffusion.

import math
from typing import Optional

import gin
import torch
import torch.nn as nn
import torch.optim
from einops import rearrange
from torch.nn import functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
            self,
            dim: int,
            is_random: bool = False,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))


class ResidualMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            width: int,
            depth: int,
            output_dim: int,
            activation: str = "relu",
            layer_norm: bool = False,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
            nn.LayerNorm(width) if layer_norm else torch.nn.Identity(),
        )

        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.activation(self.network(x)))


@gin.configurable
class ResidualMLPDenoiser(nn.Module):
    def __init__(
            self,
            d_in: int,
            dim_t: int = 128,
            mlp_width: int = 1024,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = True,
            learned_sinusoidal_dim: int = 16,
            activation: str = "relu",
            layer_norm: bool = True,
            cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.residual_mlp = ResidualMLP(
            input_dim=dim_t,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )
        if cond_dim is not None:
            self.proj = nn.Linear(d_in + cond_dim, dim_t)
            self.conditional = True
        else:
            self.proj = nn.Linear(d_in, dim_t)
            self.conditional = False

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            cond=None,
    ) -> torch.Tensor:
        if self.conditional:
            assert cond is not None
            x = torch.cat((x, cond), dim=-1)
        time_embed = self.time_mlp(timesteps)
        x = self.proj(x) + time_embed
        return self.residual_mlp(x)

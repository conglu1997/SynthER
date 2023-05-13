# Normalizers for diffusion.

from typing import List

import torch
from torch import nn


class BaseNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MinMaxNormalizer(BaseNormalizer):
    def __init__(self, dataset: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.register_buffer('min', dataset.min(dim=0).values)
        self.register_buffer('max', dataset.max(dim=0).values + eps)
        print('Mins:', self.min)
        print('Maxs:', self.max)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min) / (self.max - self.min) * 2 - 1

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1) / 2 * (self.max - self.min) + self.min

    def reset(self, dataset: torch.Tensor, eps: float = 1e-5):
        self.min = dataset.min(dim=0).values
        self.max = dataset.max(dim=0).values + eps
        print('Mins:', self.min)
        print('Maxs:', self.max)


class Normalizer(BaseNormalizer):
    def __init__(
            self,
            dataset: torch.Tensor,
            eps: float = 1e-5,
            skip_dims: List[int] = [],
            target_std: float = 1.0,
    ):
        super().__init__()
        self.register_buffer('mean', dataset.mean(dim=0))
        self.register_buffer('std', dataset.std(dim=0) + eps)
        self.skip_dims = skip_dims
        if skip_dims:
            self.mean[skip_dims] = 0.0
            self.std[skip_dims] = 1.0
        self.target_std = target_std
        print('Means:', self.mean)
        print('Stds:', self.std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std * self.target_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.target_std * self.std + self.mean

    def reset(self, dataset: torch.Tensor, eps: float = 1e-5):
        self.mean = dataset.mean(dim=0)
        self.std = dataset.std(dim=0) + eps
        if self.skip_dims:
            self.mean[self.skip_dims] = 0.0
            self.std[self.skip_dims] = 1.0
        print('Means:', self.mean)
        print('Stds:', self.std)


def normalizer_factory(
        normalizer_type: str,
        dataset: torch.Tensor,
        skip_dims: List[int] = [],
        **kwargs,
) -> BaseNormalizer:
    if normalizer_type == 'minmax':
        return MinMaxNormalizer(dataset, **kwargs)
    elif normalizer_type == 'standard':
        return Normalizer(dataset, skip_dims=skip_dims, **kwargs)
    else:
        raise ValueError(f'Unknown normalizer type: {normalizer_type}')

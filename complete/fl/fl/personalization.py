"""Personalization methods: FedProx loss term and FedBN utilities."""

from typing import Iterable, Tuple

import torch


def fedprox_loss(
    model: torch.nn.Module,
    global_params: Iterable[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """FedProx proximal term mu/2 * ||w - w_global||^2."""
    prox = torch.tensor(0.0, device=next(model.parameters()).device)
    for p, g in zip(model.parameters(), global_params):
        prox = prox + torch.sum((p - g.to(p.device)) ** 2)
    return (mu / 2.0) * prox


def split_bn_and_nonbn_params(model: torch.nn.Module) -> Tuple[Iterable[torch.nn.Parameter], Iterable[torch.nn.Parameter]]:
    bn_params = []
    other_params = []
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            for p in m.parameters(recurse=False):
                bn_params.append(p)
        else:
            for p in m.parameters(recurse=False):
                other_params.append(p)
    return bn_params, other_params



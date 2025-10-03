"""Differential Privacy utilities using Opacus if available, else manual stub."""

from typing import Optional

import torch

try:
    from opacus import PrivacyEngine  # type: ignore
except Exception:  # pragma: no cover
    PrivacyEngine = None  # type: ignore


def attach_dp_if_enabled(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device: torch.device,
    dp_cfg: dict,
):
    if not dp_cfg.get("enabled", False):
        return None

    if PrivacyEngine is None:
        raise RuntimeError("Opacus not installed. Install opacus to enable DP-SGD.")

    noise_multiplier = float(dp_cfg.get("noise_multiplier", 0.8))
    max_grad_norm = float(dp_cfg.get("max_grad_norm", 1.0))

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return privacy_engine, model, optimizer, dataloader



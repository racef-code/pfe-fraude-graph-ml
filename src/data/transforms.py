"""Transformations de features. Normalisation z-score basée sur le TRAIN
uniquement (les stats val/test ne doivent pas fuiter)."""
import torch


def standardize_features(x: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Centre-réduit x avec moyenne/écart-type calculés sur le train seulement."""
    mean = x[train_mask].mean(dim=0, keepdim=True)
    std = x[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std

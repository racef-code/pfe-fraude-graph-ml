"""Transformations de features.

Règle anti-fuite : les statistiques des nœuds cibles évalués (val/test) ne
rentrent jamais dans la normalisation. Pour YelpChi/homogène, on utilise le
train mask. Pour le graphe fiscal hétérogène, on normalise `company` avec les
entreprises train uniquement ; les nœuds support sans label (person/address/...)
peuvent être normalisés sur tous leurs nœuds car ils ne portent pas la cible.
"""
from __future__ import annotations

import torch


def _zscore(x: torch.Tensor, stats_mask: torch.Tensor | None = None) -> torch.Tensor:
    if stats_mask is not None:
        if not bool(stats_mask.any()):
            return x
        base = x[stats_mask]
    else:
        base = x
    mean = base.mean(dim=0, keepdim=True)
    std = base.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def standardize_features(x: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Centre-réduit x avec moyenne/écart-type calculés sur le train seulement."""
    return _zscore(x, train_mask)


def standardize_hetero_features(data, target_node_type: str = "company"):
    """Normalise les features d'un HeteroData sans fuite sur les nœuds cibles.

    Mutates and returns ``data`` for convenience.
    """
    for ntype in data.node_types:
        x = data[ntype].x
        mask = None
        if ntype == target_node_type and hasattr(data[ntype], "train_mask"):
            mask = data[ntype].train_mask
        data[ntype].x = _zscore(x, mask)
    return data

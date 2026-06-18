"""Entraînement pour HeteroData fiscal.

Séparé de ``train_gnn.py`` pour garder le chemin homogène YelpChi simple.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from src.config import DEVICE


def train_hetero_company_gnn(model, data, cfg, class_weight=None):
    """Entraîne un modèle hétérogène et optimise seulement les labels company."""
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    if class_weight is not None:
        class_weight = class_weight.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    train_mask = data["company"].train_mask
    val_mask = data["company"].val_mask
    if not bool(val_mask.any()):
        val_mask = train_mask

    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[train_mask], data["company"].y[train_mask], weight=class_weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            val_loss = F.cross_entropy(out[val_mask], data["company"].y[val_mask], weight=class_weight).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model


def predict_hetero_company_scores(model, data):
    """Probabilité de fraude pour chaque entreprise."""
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(logits, dim=1)[:, 1]
    return probs.cpu().numpy()

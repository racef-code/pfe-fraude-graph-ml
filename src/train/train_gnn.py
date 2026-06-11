"""Boucle d'entraînement GNN full-batch + early-stop. CrossEntropy avec
class_weight gère le déséquilibre (piège n°2)."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from src.config import DEVICE


def train_gnn(model, data, cfg, class_weight=None):
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    if class_weight is not None:
        class_weight = class_weight.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                               weight=class_weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask],
                                       weight=class_weight).item()
        if val_loss < best_val:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_scores(model, data):
    """Probabilité de la classe positive (index 1)."""
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index), dim=1)
    return probs[:, 1].cpu().numpy()


def class_weights_from_labels(y) -> torch.Tensor:
    """w_c = N / (n_classes * count_c). Pour le binaire -> pondère le positif."""
    y = np.asarray(y.cpu() if hasattr(y, "cpu") else y)
    classes, counts = np.unique(y, return_counts=True)
    w = len(y) / (len(classes) * counts)
    return torch.tensor(w, dtype=torch.float)

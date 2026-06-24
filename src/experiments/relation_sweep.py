"""Sweep d'architectures GNN x relations sur YelpChi.

Question : la faiblesse de GraphSAGE vient-elle du GNN, ou du GRAPHE qu'on lui
donne ? On entraîne un GNN sur chaque relation séparément (net_rur très
homophile vs homo/rsr quasi aléatoires) et on compare. Générique : marche pour
GraphSAGE comme pour GAT via un `model_factory`.
"""
from collections import defaultdict

import numpy as np

from src.config import TrainConfig, set_seed
from src.data.transforms import standardize_features
from src.data.yelpchi import load_yelpchi
from src.eval.metrics import compute_metrics
from src.train.train_gnn import (
    class_weights_from_labels,
    predict_scores,
    train_gnn,
)

METRIC_KEYS = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]


def run_gnn_sweep(model_factory, relations, seeds=(0, 1, 2), cfg: TrainConfig | None = None):
    """model_factory(in_dim)->nn.Module. Renvoie {relation: {metric: (mean, std)}}."""
    cfg = cfg or TrainConfig()
    out = {}
    for rel in relations:
        acc = defaultdict(list)
        for seed in seeds:
            set_seed(seed)
            data = load_yelpchi(seed=seed, relation=rel)
            data = data.clone()
            data.x = standardize_features(data.x, data.train_mask)
            cw = class_weights_from_labels(data.y[data.train_mask])
            model = model_factory(data.x.shape[1])
            model = train_gnn(model, data, cfg, class_weight=cw)
            scores = predict_scores(model, data)
            y = data.y.cpu().numpy()
            te = data.test_mask.cpu().numpy()
            for k, v in compute_metrics(y[te], scores[te]).items():
                acc[k].append(v)
        out[rel] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}
    return out


def format_sweep(results: dict, title: str, n_seeds: int) -> str:
    lines = [title]
    header = "Relation".ljust(12) + "".join(k.ljust(20) for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for rel, metrics in results.items():
        row = rel.ljust(12)
        for k in METRIC_KEYS:
            mean, std = metrics[k]
            row += f"{mean:.4f}+/-{std:.4f}".ljust(20)
        lines.append(row)
    lines.append(f"(moyenne +/- ecart-type sur {n_seeds} seeds)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from src.models.graphsage import GraphSAGE

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rels = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "net_rur", "net_rtr", "homo", "net_rsr"
    ]
    seeds = tuple(range(n))
    print(f"GraphSAGE sweep relations={rels} seeds={seeds} ...", flush=True)
    res = run_gnn_sweep(lambda d: GraphSAGE(d, 64, 2, dropout=0.5), rels, seeds)
    print(flush=True)
    print(format_sweep(res, "GraphSAGE par relation", n), flush=True)

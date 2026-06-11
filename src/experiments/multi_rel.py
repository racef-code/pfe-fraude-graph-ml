"""Benchmark du GNN multi-relationnel sur YelpChi (5 seeds).

Combine net_rur (propre, sparse) + net_rtr + net_rsr (denses). À comparer aux
GNN single-relation (relation_sweep) et à XGBoost (benchmark).
"""
from collections import defaultdict

import numpy as np

from src.config import TrainConfig, set_seed
from src.data.transforms import standardize_features
from src.data.yelpchi import load_yelpchi_multi
from src.eval.metrics import compute_metrics
from src.models.multi_rel_gnn import MultiRelGNN
from src.train.train_gnn import class_weights_from_labels, predict_scores, train_gnn

METRIC_KEYS = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]


def run_multi_rel(seeds=(0, 1, 2, 3, 4), relations=("net_rur", "net_rtr", "net_rsr"),
                  cfg: TrainConfig | None = None):
    cfg = cfg or TrainConfig()
    acc = defaultdict(list)
    for seed in seeds:
        set_seed(seed)
        data, eis = load_yelpchi_multi(seed=seed, relations=relations)
        data = data.clone()
        data.x = standardize_features(data.x, data.train_mask)
        cw = class_weights_from_labels(data.y[data.train_mask])
        model = MultiRelGNN(data.x.shape[1], cfg.hidden_dim, 2, edge_index_list=eis,
                            dropout=cfg.dropout)
        model = train_gnn(model, data, cfg, class_weight=cw)
        scores = predict_scores(model, data)
        y = data.y.cpu().numpy()
        te = data.test_mask.cpu().numpy()
        for k, v in compute_metrics(y[te], scores[te]).items():
            acc[k].append(v)
    return {k: (float(np.mean(acc[k])), float(np.std(acc[k]))) for k in METRIC_KEYS}


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds = tuple(range(n))
    print(f"MultiRelGNN (net_rur+net_rtr+net_rsr) seeds={seeds} ...", flush=True)
    res = run_multi_rel(seeds=seeds)
    print(flush=True)
    for k in METRIC_KEYS:
        m, s = res[k]
        print(f"{k:12s}: {m:.4f}+/-{s:.4f}", flush=True)

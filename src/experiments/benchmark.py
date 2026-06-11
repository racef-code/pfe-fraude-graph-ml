"""Banc d'essai multi-seed : GraphSAGE vs XGBoost vs XGBoost+features de graphe.

Pour chaque seed : nouveau split stratifié, entraînement des 3 modèles sur les
mêmes données, évaluation sur le test. On agrège en moyenne ± écart-type pour
des résultats crédibles (pas un coup de chance sur un seul split).

Modèles :
  - GraphSAGE        : features normalisées + message-passing (relations).
  - XGBoost          : mêmes features normalisées, AUCUN graphe.
  - XGBoost+graph    : features + 2 features de graphe leak-free (degré, ratio
                       de fraude des voisins-train).
"""
from collections import defaultdict

import numpy as np

from src.config import TrainConfig, set_seed
from src.data.graph_features import compute_graph_features
from src.data.transforms import standardize_features
from src.data.yelpchi import load_yelpchi
from src.eval.metrics import compute_metrics
from src.models.graphsage import GraphSAGE
from src.train.baseline_xgb import predict_xgb, train_xgb
from src.train.train_gnn import (
    class_weights_from_labels,
    predict_scores,
    train_gnn,
)

METRIC_KEYS = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]


def _run_one_seed(seed: int, relation: str, cfg: TrainConfig) -> dict:
    set_seed(seed)
    data = load_yelpchi(seed=seed, relation=relation)

    # normalisation z-score (stats train uniquement)
    x_norm = standardize_features(data.x, data.train_mask)
    gf = compute_graph_features(data.edge_index, data.y, data.train_mask, data.num_nodes)
    gf_norm = standardize_features(gf, data.train_mask)

    y = data.y.cpu().numpy()
    train = data.train_mask.cpu().numpy()
    test = data.test_mask.cpu().numpy()
    n_neg, n_pos = (y[train] == 0).sum(), (y[train] == 1).sum()
    spw = n_neg / n_pos

    # --- GraphSAGE (features normalisées) ---
    data_n = data.clone()
    data_n.x = x_norm
    cw = class_weights_from_labels(data.y[data.train_mask])
    gnn = GraphSAGE(x_norm.shape[1], cfg.hidden_dim, 2, dropout=cfg.dropout)
    gnn = train_gnn(gnn, data_n, cfg, class_weight=cw)
    gnn_scores = predict_scores(gnn, data_n)

    # --- XGBoost (sans graphe) ---
    Xn = x_norm.cpu().numpy()
    xgb = train_xgb(Xn[train], y[train], scale_pos_weight=spw)
    xgb_scores = predict_xgb(xgb, Xn)

    # --- XGBoost + features de graphe ---
    Xg = np.concatenate([Xn, gf_norm.cpu().numpy()], axis=1)
    xgbg = train_xgb(Xg[train], y[train], scale_pos_weight=spw)
    xgbg_scores = predict_xgb(xgbg, Xg)

    return {
        "GraphSAGE": compute_metrics(y[test], gnn_scores[test]),
        "XGBoost": compute_metrics(y[test], xgb_scores[test]),
        "XGBoost+graph": compute_metrics(y[test], xgbg_scores[test]),
    }


def run_benchmark(
    seeds=(0, 1, 2),
    relation: str = "homo",
    cfg: TrainConfig | None = None,
    return_per_seed: bool = False,
):
    """Renvoie {model: {metric: (mean, std)}} agrégé sur les seeds.

    Si return_per_seed=True, renvoie aussi la liste brute par seed
    [{model: {metric: value}}] pour les comparaisons APPARIÉES (même split).
    """
    cfg = cfg or TrainConfig()
    per_seed = []
    acc = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        res = _run_one_seed(seed, relation, cfg)
        per_seed.append(res)
        for model, metrics in res.items():
            for k, v in metrics.items():
                acc[model][k].append(v)
    out = {}
    for model, metrics in acc.items():
        out[model] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}
    if return_per_seed:
        return out, per_seed
    return out


def paired_delta(per_seed, model: str, baseline: str, metric: str) -> dict:
    """Delta apparié (model - baseline) du `metric`, seed par seed.

    Comparaison correcte quand les deux modèles partagent le même split par
    seed : on regarde la distribution des différences, pas le chevauchement
    des bandes marginales. Renvoie mean, std et nb de seeds où model > baseline.
    """
    diffs = [s[model][metric] - s[baseline][metric] for s in per_seed]
    diffs = np.asarray(diffs)
    return {
        "mean": float(diffs.mean()),
        "std": float(diffs.std()),
        "n_positive": int((diffs > 0).sum()),
        "n_total": len(diffs),
    }


def format_table(results: dict, n_seeds: int) -> str:
    lines = []
    header = "Model".ljust(16) + "".join(k.ljust(20) for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for model, metrics in results.items():
        row = model.ljust(16)
        for k in METRIC_KEYS:
            mean, std = metrics[k]
            row += f"{mean:.4f}+/-{std:.4f}".ljust(20)
        lines.append(row)
    lines.append(f"\n(moyenne +/- ecart-type sur {n_seeds} seeds)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    seeds = tuple(range(int(sys.argv[1]))) if len(sys.argv) > 1 else (0, 1, 2)
    print(f"Benchmark sur seeds={seeds} ...", flush=True)
    results = run_benchmark(seeds=seeds)
    print(flush=True)
    print(format_table(results, len(seeds)), flush=True)

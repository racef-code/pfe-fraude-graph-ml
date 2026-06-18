"""Synthetic IS benchmark for the future heterogeneous fiscal graph.

This uses synthetic data only. It is an engineering/protocol smoke test, not a
scientific claim about real tax fraud. Its value is to verify that the full
future path works before real company/IS data arrives:

    tables -> HeteroData -> XGBoost baseline -> FiscalHeteroGNN -> metrics

The script supports multi-seed runs so results are reported like the YelpChi
benchmark (mean +/- std + paired deltas).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.config import TrainConfig, set_seed
from src.data.fiscal_graph import build_fiscal_graph, make_synthetic_is_fiscal_tables
from src.data.transforms import standardize_hetero_features
from src.eval.metrics import compute_metrics
from src.models.hetero_fiscal_gnn import FiscalHeteroGNN
from src.models.relation_gated_fiscal_gnn import RelationGatedFiscalGNN
from src.train.baseline_xgb import predict_xgb, train_xgb
from src.train.train_gnn import class_weights_from_labels
from src.train.train_hetero import predict_hetero_company_scores, train_hetero_company_gnn

METRIC_KEYS = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]
MODEL_KEYS = ["XGBoost-company-features", "FiscalHeteroGNN", "RelationGatedFiscalGNN"]


def run_synthetic_is_experiment(seed: int = 42, n_companies: int = 240, epochs: int = 30):
    """Run one synthetic IS split and return per-model metrics.

    Normalization is anti-leak for the target node type: company statistics are
    computed from train companies only via ``standardize_hetero_features``.
    """
    set_seed(seed)
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=n_companies, seed=seed)
    data = build_fiscal_graph(nodes, edges, seed=seed)
    data = standardize_hetero_features(data, target_node_type="company")

    y = data["company"].y.cpu().numpy()
    train = data["company"].train_mask.cpu().numpy()
    test = data["company"].test_mask.cpu().numpy()
    x_company = data["company"].x.cpu().numpy()
    n_neg, n_pos = (y[train] == 0).sum(), (y[train] == 1).sum()
    spw = n_neg / max(n_pos, 1)

    # Baseline: company behavior/features only, no graph message-passing.
    xgb = train_xgb(x_company[train], y[train], scale_pos_weight=spw)
    xgb_scores = predict_xgb(xgb, x_company)

    # Heterogeneous GNN: company features + relation context.
    cfg = TrainConfig(hidden_dim=32, lr=0.01, epochs=epochs, patience=10, dropout=0.2)
    cw = class_weights_from_labels(data["company"].y[data["company"].train_mask])

    model = FiscalHeteroGNN(data.metadata(), hidden_dim=cfg.hidden_dim, out_dim=2, dropout=cfg.dropout)
    # Initialize lazy PyG modules before optimizer/class-weighted training.
    _ = model(data.x_dict, data.edge_index_dict)
    model = train_hetero_company_gnn(model, data, cfg, class_weight=cw)
    hetero_scores = predict_hetero_company_scores(model, data)

    gated = RelationGatedFiscalGNN(data.metadata(), hidden_dim=cfg.hidden_dim, out_dim=2, dropout=cfg.dropout)
    _ = gated(data.x_dict, data.edge_index_dict)
    gated = train_hetero_company_gnn(gated, data, cfg, class_weight=cw)
    gated_scores = predict_hetero_company_scores(gated, data)

    return {
        "XGBoost-company-features": compute_metrics(y[test], xgb_scores[test]),
        "FiscalHeteroGNN": compute_metrics(y[test], hetero_scores[test]),
        "RelationGatedFiscalGNN": compute_metrics(y[test], gated_scores[test]),
        "n_company": int(data["company"].num_nodes),
        "n_test_company": int(test.sum()),
        "edge_types": [str(t) for t in data.edge_types],
    }


def run_synthetic_is_benchmark(seeds=(0, 1, 2), n_companies: int = 240, epochs: int = 30):
    """Run multiple seeds and return aggregate metrics + raw per-seed metrics."""
    per_seed = []
    acc = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        res = run_synthetic_is_experiment(seed=seed, n_companies=n_companies, epochs=epochs)
        per_seed.append(res)
        for model in MODEL_KEYS:
            for metric, value in res[model].items():
                acc[model][metric].append(value)
    summary = {
        model: {
            metric: (float(np.mean(values)), float(np.std(values)))
            for metric, values in metrics.items()
        }
        for model, metrics in acc.items()
    }
    return summary, per_seed


def paired_delta(per_seed, model: str, baseline: str, metric: str) -> dict:
    """Paired seed-by-seed delta, same protocol as the YelpChi benchmark."""
    diffs = np.asarray([seed_res[model][metric] - seed_res[baseline][metric] for seed_res in per_seed])
    return {
        "mean": float(diffs.mean()),
        "std": float(diffs.std()),
        "n_positive": int((diffs > 0).sum()),
        "n_total": int(len(diffs)),
    }


def format_single_result(results: dict) -> str:
    lines = []
    for model in MODEL_KEYS:
        lines.append(model)
        for k in METRIC_KEYS:
            lines.append(f"  {k:12s}: {results[model][k]:.4f}")
    lines.append(f"n_company: {results['n_company']}")
    lines.append(f"n_test_company: {results['n_test_company']}")
    lines.append(f"edge_types: {len(results['edge_types'])}")
    return "\n".join(lines)


def format_benchmark(summary: dict, per_seed: list[dict], n_seeds: int) -> str:
    lines = ["Synthetic IS heterogeneous benchmark", ""]
    header = "Model".ljust(28) + "".join(k.ljust(20) for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for model in MODEL_KEYS:
        row = model.ljust(28)
        for metric in METRIC_KEYS:
            mean, std = summary[model][metric]
            row += f"{mean:.4f}+/-{std:.4f}".ljust(20)
        lines.append(row)
    lines.append(f"\n(moyenne +/- ecart-type sur {n_seeds} seeds; données synthétiques)")
    for model in [m for m in MODEL_KEYS if m != "XGBoost-company-features"]:
        for metric in ["auc_pr", "auc_roc", "recall_at_k"]:
            d = paired_delta(per_seed, model, "XGBoost-company-features", metric)
            lines.append(
                f"Delta {model} - XGBoost ({metric}): "
                f"{d['mean']:+.4f}+/-{d['std']:.4f}, positives {d['n_positive']}/{d['n_total']}"
            )
    if "RelationGatedFiscalGNN" in MODEL_KEYS:
        for metric in ["auc_pr", "auc_roc", "f1_macro", "gmean"]:
            d = paired_delta(per_seed, "RelationGatedFiscalGNN", "FiscalHeteroGNN", metric)
            lines.append(
                f"Delta RelationGated - equal-sum HeteroGNN ({metric}): "
                f"{d['mean']:+.4f}+/-{d['std']:.4f}, positives {d['n_positive']}/{d['n_total']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_companies = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    if n_seeds <= 1:
        print(format_single_result(run_synthetic_is_experiment(0, n_companies, epochs)))
    else:
        seeds = tuple(range(n_seeds))
        summary, per_seed = run_synthetic_is_benchmark(seeds=seeds, n_companies=n_companies, epochs=epochs)
        print(format_benchmark(summary, per_seed, n_seeds))

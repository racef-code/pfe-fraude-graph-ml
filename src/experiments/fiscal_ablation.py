"""Relation ablation benchmark for the synthetic IS heterogeneous graph.

This is the key Graph-ML experiment: train the same HeteroGNN after removing one
relation family at a time, then measure which relations actually create lift.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.config import TrainConfig, set_seed
from src.data.fiscal_graph import RELATION_TYPES, build_fiscal_graph, make_synthetic_is_fiscal_tables
from src.data.transforms import standardize_hetero_features
from src.eval.metrics import compute_metrics
from src.models.hetero_fiscal_gnn import FiscalHeteroGNN
from src.train.train_gnn import class_weights_from_labels
from src.train.train_hetero import predict_hetero_company_scores, train_hetero_company_gnn

METRIC_KEYS = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]


def _drop_relation_from_edges(edges_df, relation: str):
    return edges_df[edges_df["type_relation"] != relation].copy()


def run_one_ablation(seed: int, n_companies: int = 240, epochs: int = 40) -> dict[str, dict]:
    """Return metrics for all-relations and leave-one-relation-out variants."""
    set_seed(seed)
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=n_companies, seed=seed)
    variants = {"all_relations": edges}
    for rel in RELATION_TYPES:
        if rel in set(edges["type_relation"]):
            variants[f"without_{rel}"] = _drop_relation_from_edges(edges, rel)

    out = {}
    for name, variant_edges in variants.items():
        set_seed(seed)
        data = build_fiscal_graph(nodes, variant_edges, seed=seed)
        data = standardize_hetero_features(data, target_node_type="company")
        cfg = TrainConfig(hidden_dim=32, lr=0.01, epochs=epochs, patience=10, dropout=0.2, early_stop_metric="auc_pr")
        model = FiscalHeteroGNN(data.metadata(), hidden_dim=cfg.hidden_dim, out_dim=2, dropout=cfg.dropout)
        _ = model(data.x_dict, data.edge_index_dict)
        cw = class_weights_from_labels(data["company"].y[data["company"].train_mask])
        model = train_hetero_company_gnn(model, data, cfg, class_weight=cw)
        scores = predict_hetero_company_scores(model, data)
        y = data["company"].y.cpu().numpy()
        test = data["company"].test_mask.cpu().numpy()
        out[name] = compute_metrics(y[test], scores[test])
    return out


def run_ablation_benchmark(seeds=(0, 1, 2), n_companies: int = 240, epochs: int = 40):
    per_seed = []
    acc = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        res = run_one_ablation(seed=seed, n_companies=n_companies, epochs=epochs)
        per_seed.append(res)
        for variant, metrics in res.items():
            for k, v in metrics.items():
                acc[variant][k].append(v)
    summary = {
        variant: {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}
        for variant, metrics in acc.items()
    }
    return summary, per_seed


def relation_importance(summary: dict) -> list[dict]:
    """Positive drop means removing the relation hurt performance => useful relation."""
    base = summary["all_relations"]
    rows = []
    for variant, metrics in summary.items():
        if not variant.startswith("without_"):
            continue
        rel = variant.removeprefix("without_")
        rows.append({
            "relation": rel,
            "auc_pr_drop": base["auc_pr"][0] - metrics["auc_pr"][0],
            "auc_roc_drop": base["auc_roc"][0] - metrics["auc_roc"][0],
            "recall_at_k_drop": base["recall_at_k"][0] - metrics["recall_at_k"][0],
        })
    return sorted(rows, key=lambda r: r["auc_pr_drop"], reverse=True)


def format_ablation(summary: dict) -> str:
    variants = ["all_relations"] + sorted(v for v in summary if v != "all_relations")
    lines = ["Fiscal HeteroGNN relation ablation", ""]
    header = "Variant".ljust(28) + "".join(k.ljust(20) for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for variant in variants:
        row = variant.ljust(28)
        for k in METRIC_KEYS:
            mean, std = summary[variant][k]
            row += f"{mean:.4f}+/-{std:.4f}".ljust(20)
        lines.append(row)

    lines.append("\nRelation importance (drop vs all_relations; positive = useful)")
    lines.append("Relation".ljust(22) + "AUC-PR drop".ljust(18) + "AUC-ROC drop".ljust(18) + "Recall@k drop")
    lines.append("-" * 72)
    for r in relation_importance(summary):
        lines.append(
            r["relation"].ljust(22)
            + f"{r['auc_pr_drop']:+.4f}".ljust(18)
            + f"{r['auc_roc_drop']:+.4f}".ljust(18)
            + f"{r['recall_at_k_drop']:+.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_companies = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    summary, _ = run_ablation_benchmark(seeds=tuple(range(n_seeds)), n_companies=n_companies, epochs=epochs)
    print(format_ablation(summary))

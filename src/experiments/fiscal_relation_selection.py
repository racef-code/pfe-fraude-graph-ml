"""Relation-selection benchmark for the synthetic fiscal graph.

Unlike leave-one-out ablation, this asks whether a curated relation subset beats
using every edge type. It should be interpreted as pipeline/method validation on
synthetic data, not as a claim about real fiscal fraud.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.config import TrainConfig, set_seed
from src.data.fiscal_graph import build_fiscal_graph, make_synthetic_is_fiscal_tables
from src.data.transforms import standardize_hetero_features
from src.eval.metrics import compute_metrics
from src.features.fiscal_graph_features import compute_fiscal_graph_features
from src.models.relation_gated_fiscal_gnn import RelationGatedFiscalGNN
from src.train.baseline_xgb import predict_xgb, train_xgb
from src.train.train_gnn import class_weights_from_labels
from src.train.train_hetero import predict_hetero_company_scores, train_hetero_company_gnn

METRIC_KEYS = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]
VARIANTS = {
    "all_relations": None,
    "transaction_only": ["transaction"],
    "transaction_accountant": ["transaction", "uses_accountant"],
    "transaction_accountant_address": ["transaction", "uses_accountant", "registered_at"],
    "no_sector": ["transaction", "has_director", "registered_at", "uses_accountant", "participates_in"],
    "no_sector_no_participation": ["transaction", "has_director", "registered_at", "uses_accountant"],
    "audit_selected": ["transaction", "uses_accountant", "registered_at"],
}
MODEL_KEYS = ["XGBoost+graph", "RelationGatedFiscalGNN"]


def _filter_edges(edges, relations):
    if relations is None:
        return edges.copy()
    return edges[edges["type_relation"].isin(relations)].copy()


def run_one_selection(seed: int = 0, n_companies: int = 240, epochs: int = 30):
    set_seed(seed)
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=n_companies, seed=seed)
    out = {}
    for variant, relations in VARIANTS.items():
        set_seed(seed)
        variant_edges = _filter_edges(edges, relations)
        data = build_fiscal_graph(nodes, variant_edges, seed=seed)
        data = standardize_hetero_features(data, target_node_type="company")
        y = data["company"].y.cpu().numpy()
        train = data["company"].train_mask.cpu().numpy()
        test = data["company"].test_mask.cpu().numpy()
        x_company = data["company"].x.cpu().numpy()
        company_ids = list(data["company"].node_id)
        train_ids = {cid for cid, is_train in zip(company_ids, train) if is_train}
        x_graph, _ = compute_fiscal_graph_features(
            nodes,
            variant_edges,
            company_ids=company_ids,
            train_company_ids=train_ids,
            relations=list(relations) if relations is not None else None,
        )
        x_tab_graph = np.concatenate([x_company, x_graph], axis=1)
        n_neg, n_pos = (y[train] == 0).sum(), (y[train] == 1).sum()
        spw = n_neg / max(n_pos, 1)
        xgb = train_xgb(x_tab_graph[train], y[train], scale_pos_weight=spw)
        xgb_scores = predict_xgb(xgb, x_tab_graph)

        cfg = TrainConfig(hidden_dim=32, lr=0.01, epochs=epochs, patience=10, dropout=0.2, early_stop_metric="auc_pr")
        model = RelationGatedFiscalGNN(data.metadata(), hidden_dim=cfg.hidden_dim, out_dim=2, dropout=cfg.dropout)
        cw = class_weights_from_labels(data["company"].y[data["company"].train_mask])
        model = train_hetero_company_gnn(model, data, cfg, class_weight=cw)
        gnn_scores = predict_hetero_company_scores(model, data)

        out[variant] = {
            "XGBoost+graph": compute_metrics(y[test], xgb_scores[test]),
            "RelationGatedFiscalGNN": compute_metrics(y[test], gnn_scores[test]),
        }
    return out


def run_selection_benchmark(seeds=(0, 1, 2), n_companies: int = 240, epochs: int = 30):
    per_seed = []
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed in seeds:
        res = run_one_selection(seed, n_companies, epochs)
        per_seed.append(res)
        for variant, models in res.items():
            for model, metrics in models.items():
                for metric, value in metrics.items():
                    acc[variant][model][metric].append(value)
    summary = {
        variant: {
            model: {metric: (float(np.mean(values)), float(np.std(values))) for metric, values in metrics.items()}
            for model, metrics in models.items()
        }
        for variant, models in acc.items()
    }
    return summary, per_seed


def format_selection(summary: dict, n_seeds: int) -> str:
    lines = ["Fiscal relation selection benchmark", ""]
    header = "Variant".ljust(34) + "Model".ljust(28) + "".join(k.ljust(20) for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for variant in VARIANTS:
        for model in MODEL_KEYS:
            row = variant.ljust(34) + model.ljust(28)
            for metric in METRIC_KEYS:
                mean, std = summary[variant][model][metric]
                row += f"{mean:.4f}+/-{std:.4f}".ljust(20)
            lines.append(row)
    lines.append(f"\n(moyenne +/- ecart-type sur {n_seeds} seeds; données synthétiques)")
    if n_seeds < 10:
        lines.append("WARNING: audit-grade reporting should use >=10 seeds; this run is a smoke/iteration run.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_companies = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    summary, _ = run_selection_benchmark(tuple(range(n_seeds)), n_companies, epochs)
    print(format_selection(summary, n_seeds))

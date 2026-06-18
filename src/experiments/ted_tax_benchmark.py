"""Run this project's baselines on the TED tax-evasion dataset format.

The repository does not vendor TED data. Pass a local TED dataset directory, e.g.:
    python -m src.experiments.ted_tax_benchmark /tmp/TED/TED/Data/T20H 1v9 40
"""
from __future__ import annotations

import sys

import numpy as np

from src.config import TrainConfig, set_seed
from src.data.ted_tax import TEDLabelSplit, load_ted_heterodata
from src.eval.metrics import compute_metrics
from src.features.fiscal_graph_features import compute_fiscal_graph_features
from src.models.hetero_fiscal_gnn import FiscalHeteroGNN
from src.models.relation_gated_fiscal_gnn import RelationGatedFiscalGNN
from src.train.baseline_xgb import predict_xgb, train_xgb
from src.train.train_gnn import class_weights_from_labels
from src.train.train_hetero import predict_hetero_company_scores, train_hetero_company_gnn

METRIC_KEYS = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]
MODEL_KEYS = ["XGBoost-company", "XGBoost+graph", "FiscalHeteroGNN", "RelationGatedFiscalGNN"]
TED_RELATIONS_T20H = [
    "transaction",
    "info_change",
    "company_sell_item",
    "company_buy_item",
    "person_company",
    "belong_to",
]


def split_from_ratio(ratio: str) -> TEDLabelSplit:
    ratio = ratio.replace("TaxPayer", "")
    return TEDLabelSplit(
        train_label_file=f"label_TaxPayer{ratio}.dat",
        test_label_file=f"label_TaxPayer{ratio}.dat.test",
        val_fraction=0.2,
    )


def run_ted_benchmark(dataset_dir: str, ratio: str = "1v9", epochs: int = 40, seed: int = 42):
    set_seed(seed)
    split = split_from_ratio(ratio)
    data, nodes, edges, meta = load_ted_heterodata(dataset_dir, split=split, seed=seed)
    y = data["company"].y.cpu().numpy()
    train = data["company"].train_mask.cpu().numpy()
    test = data["company"].test_mask.cpu().numpy()
    x_company = data["company"].x.cpu().numpy()
    company_ids = list(data["company"].node_id)
    train_ids = {cid for cid, is_train in zip(company_ids, train) if is_train}
    n_neg, n_pos = (y[train] == 0).sum(), (y[train] == 1).sum()
    spw = n_neg / max(n_pos, 1)

    xgb = train_xgb(x_company[train], y[train], scale_pos_weight=spw)
    xgb_scores = predict_xgb(xgb, x_company)

    x_graph, graph_feature_names = compute_fiscal_graph_features(
        nodes,
        edges,
        company_ids=company_ids,
        train_company_ids=train_ids,
        relations=TED_RELATIONS_T20H,
    )
    x_tab_graph = np.concatenate([x_company, x_graph], axis=1)
    xgb_graph = train_xgb(x_tab_graph[train], y[train], scale_pos_weight=spw)
    xgb_graph_scores = predict_xgb(xgb_graph, x_tab_graph)

    cfg = TrainConfig(hidden_dim=16, lr=0.01, epochs=epochs, patience=10, dropout=0.2, early_stop_metric="auc_pr")
    cw = class_weights_from_labels(data["company"].y[data["company"].train_mask])

    hetero = FiscalHeteroGNN(data.metadata(), hidden_dim=cfg.hidden_dim, out_dim=2, dropout=cfg.dropout)
    hetero = train_hetero_company_gnn(hetero, data, cfg, class_weight=cw)
    hetero_scores = predict_hetero_company_scores(hetero, data)

    gated = RelationGatedFiscalGNN(data.metadata(), hidden_dim=cfg.hidden_dim, out_dim=2, dropout=cfg.dropout)
    gated = train_hetero_company_gnn(gated, data, cfg, class_weight=cw)
    gated_scores = predict_hetero_company_scores(gated, data)

    results = {
        "XGBoost-company": compute_metrics(y[test], xgb_scores[test]),
        "XGBoost+graph": compute_metrics(y[test], xgb_graph_scores[test]),
        "FiscalHeteroGNN": compute_metrics(y[test], hetero_scores[test]),
        "RelationGatedFiscalGNN": compute_metrics(y[test], gated_scores[test]),
        "meta": meta["stats"] | {"n_graph_features": len(graph_feature_names), "ratio": ratio},
    }
    return results


def format_ted_result(results: dict) -> str:
    lines = ["TED tax-evasion benchmark", "", f"meta: {results['meta']}", ""]
    header = "Model".ljust(28) + "".join(k.ljust(14) for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * len(header))
    for model in MODEL_KEYS:
        row = model.ljust(28)
        for metric in METRIC_KEYS:
            row += f"{results[model][metric]:.4f}".ljust(14)
        lines.append(row)
    lines.append("\nNOTE: bundled TED GitHub sample is tiny; treat as adapter smoke test unless full T20H/T15S data is downloaded.")
    return "\n".join(lines)


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/TED/TED/Data/T20H"
    ratio = sys.argv[2] if len(sys.argv) > 2 else "1v9"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    print(format_ted_result(run_ted_benchmark(dataset_dir, ratio, epochs)))

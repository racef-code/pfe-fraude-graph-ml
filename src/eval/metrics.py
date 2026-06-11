"""Évaluation : KPIs adaptés au déséquilibre (jamais l'accuracy)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_metrics(y_true, scores, k: int | None = None, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    n_pos = int((y_true == 1).sum())
    if k is None:
        k = n_pos
    topk = np.argsort(scores)[::-1][:k]
    recall_at_k = int((y_true[topk] == 1).sum()) / n_pos if n_pos else 0.0

    return {
        "auc_roc": roc_auc_score(y_true, scores),
        "auc_pr": average_precision_score(y_true, scores),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "gmean": float(np.sqrt(sensitivity * specificity)),
        "recall_at_k": recall_at_k,
    }


def print_comparison(results: dict[str, dict]) -> None:
    """results = {'GraphSAGE': {...}, 'XGBoost': {...}}. Gabarit PC-GNN."""
    keys = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]
    header = "Model".ljust(12) + "".join(k.ljust(13) for k in keys)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = name.ljust(12) + "".join(f"{m[k]:.4f}".ljust(13) for k in keys)
        print(row)

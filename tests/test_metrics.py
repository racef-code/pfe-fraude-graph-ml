import numpy as np
from src.eval.metrics import compute_metrics, print_comparison


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, scores, k=2)
    assert m["auc_roc"] == 1.0
    assert m["auc_pr"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["gmean"] == 1.0
    assert m["recall_at_k"] == 1.0


def test_recall_at_k_partial():
    y_true = np.array([0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1])  # top-2 = idx 0,1 -> 1 of 3 positives
    m = compute_metrics(y_true, scores, k=2)
    assert abs(m["recall_at_k"] - 1 / 3) < 1e-9


def test_default_k_is_positive_count():
    y_true = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    m = compute_metrics(y_true, scores)  # k defaults to n_pos = 1
    assert m["recall_at_k"] == 1.0

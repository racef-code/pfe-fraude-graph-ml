import numpy as np
import pandas as pd

from src.features.fiscal_graph_features import compute_fiscal_graph_features


def _tiny_tables(label_test_value=0):
    nodes = pd.DataFrame([
        {"id": "c0", "type": "company", "label": 1, "ca": 1.0},
        {"id": "c1", "type": "company", "label": 0, "ca": 2.0},
        {"id": "c2", "type": "company", "label": label_test_value, "ca": 3.0},
        {"id": "a0", "type": "accountant", "risk_prior": 0.1},
    ])
    edges = pd.DataFrame([
        {"id_source": "c0", "id_cible": "c1", "type_relation": "transaction"},
        {"id_source": "c1", "id_cible": "c2", "type_relation": "transaction"},
        {"id_source": "c0", "id_cible": "a0", "type_relation": "uses_accountant"},
        {"id_source": "c2", "id_cible": "a0", "type_relation": "uses_accountant"},
    ])
    return nodes, edges


def test_fiscal_graph_features_have_expected_relation_columns():
    nodes, edges = _tiny_tables()
    X, names = compute_fiscal_graph_features(
        nodes,
        edges,
        train_company_ids={"c0", "c1"},
        relations=["transaction", "uses_accountant"],
    )
    assert X.shape == (3, 12)  # 2 relations * 4 features + 4 any-relation features
    assert "transaction__log_degree" in names
    assert "uses_accountant__train_fraud_neighbor_ratio" in names
    assert "any_relation__log_degree" in names


def test_fiscal_graph_features_do_not_leak_test_labels():
    nodes1, edges = _tiny_tables(label_test_value=0)
    nodes2, _ = _tiny_tables(label_test_value=1)
    X1, names1 = compute_fiscal_graph_features(
        nodes1,
        edges,
        train_company_ids={"c0", "c1"},
        relations=["transaction", "uses_accountant"],
    )
    X2, names2 = compute_fiscal_graph_features(
        nodes2,
        edges,
        train_company_ids={"c0", "c1"},
        relations=["transaction", "uses_accountant"],
    )
    assert names1 == names2
    assert np.allclose(X1, X2), "test-label leakage in fiscal graph features"


def test_train_fraud_neighbor_ratio_uses_only_train_neighbors():
    nodes, edges = _tiny_tables(label_test_value=1)
    X, names = compute_fiscal_graph_features(
        nodes,
        edges,
        company_ids=["c0", "c1", "c2"],
        train_company_ids={"c0", "c1"},
        relations=["transaction"],
    )
    ratio_col = names.index("transaction__train_fraud_neighbor_ratio")
    # c2 only sees train neighbor c1, whose train label is 0, even though c2 itself is test label 1.
    assert abs(X[2, ratio_col] - 0.0) < 1e-6

import torch

from src.config import TrainConfig, set_seed
from src.data.fiscal_graph import build_fiscal_graph, make_synthetic_is_fiscal_tables
from src.models.hetero_fiscal_gnn import FiscalHeteroGNN
from src.train.train_gnn import class_weights_from_labels
from src.train.train_hetero import predict_hetero_company_scores, train_hetero_company_gnn


def test_build_fiscal_graph_returns_heterodata_with_company_labels_and_reverse_edges():
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=80, seed=0)
    data = build_fiscal_graph(nodes, edges, seed=0)

    assert "company" in data.node_types
    assert data["company"].x.shape[0] == 80
    assert data["company"].y.shape == (80,)
    assert data["company"].train_mask.sum() > 0
    assert data["company"].val_mask.sum() > 0
    assert data["company"].test_mask.sum() > 0

    # Les reverse edges sont essentiels : ils permettent aux entreprises de
    # recevoir des messages depuis dirigeants/adresses/comptables.
    assert ("person", "rev_has_director", "company") in data.edge_types
    assert ("address", "rev_registered_at", "company") in data.edge_types
    assert ("accountant", "rev_uses_accountant", "company") in data.edge_types


def test_build_fiscal_graph_ignores_unlabeled_companies_in_masks():
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=40, seed=1)
    company_rows = nodes["type"] == "company"
    first_company_idx = nodes[company_rows].index[0]
    nodes.loc[first_company_idx, "label"] = float("nan")

    data = build_fiscal_graph(nodes, edges, seed=1)
    assert data["company"].y[0].item() == -1
    assert not data["company"].labeled_mask[0].item()


def test_fiscal_hetero_gnn_forward_and_training_smoke():
    set_seed(2)
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=60, seed=2)
    data = build_fiscal_graph(nodes, edges, seed=2)

    model = FiscalHeteroGNN(data.metadata(), hidden_dim=12, out_dim=2, dropout=0.1)
    # Lazy modules initialize on first forward.
    out = model(data.x_dict, data.edge_index_dict)
    assert out.shape == (data["company"].num_nodes, 2)

    cfg = TrainConfig(hidden_dim=12, epochs=3, patience=3, lr=0.01, dropout=0.1)
    cw = class_weights_from_labels(data["company"].y[data["company"].train_mask])
    trained = train_hetero_company_gnn(model, data, cfg, class_weight=cw)
    scores = predict_hetero_company_scores(trained, data)
    assert scores.shape == (data["company"].num_nodes,)
    assert ((scores >= 0) & (scores <= 1)).all()

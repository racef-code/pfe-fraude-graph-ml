import torch

from src.data.transforms import standardize_features, standardize_hetero_features
from src.data.fiscal_graph import build_fiscal_graph, make_synthetic_is_fiscal_tables


def test_standardize_features_uses_train_stats_only():
    x = torch.tensor([[1.0], [2.0], [1000.0]])
    train_mask = torch.tensor([True, True, False])
    out = standardize_features(x, train_mask)
    assert torch.allclose(out[train_mask].mean(dim=0), torch.tensor([0.0]), atol=1e-6)
    # If test value leaked into stats, this would be near 1.15 not huge.
    assert out[2, 0] > 100.0


def test_standardize_hetero_features_uses_company_train_stats():
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=30, seed=3)
    data = build_fiscal_graph(nodes, edges, seed=3)
    train_mask = data["company"].train_mask.clone()
    data = standardize_hetero_features(data)
    train_x = data["company"].x[train_mask]
    assert torch.allclose(train_x.mean(dim=0), torch.zeros(train_x.shape[1]), atol=1e-5)

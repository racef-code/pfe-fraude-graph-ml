import torch

from src.data.graph_features import compute_graph_features
from src.data.transforms import standardize_features


def _chain_edges(n):
    # undirected chain 0-1-2-...-(n-1), both directions
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_degree_feature_matches_chain():
    n = 4
    ei = _chain_edges(n)  # degrees: 1,2,2,1
    y = torch.tensor([0, 1, 0, 1])
    train = torch.ones(n, dtype=torch.bool)
    feats = compute_graph_features(ei, y, train, n)
    expected_logdeg = torch.log1p(torch.tensor([1.0, 2.0, 2.0, 1.0]))
    assert torch.allclose(feats[:, 0], expected_logdeg)


def test_no_test_label_leakage():
    """Changer UNIQUEMENT les labels de test ne doit PAS changer les features."""
    n = 5
    ei = _chain_edges(n)
    train = torch.tensor([True, True, True, False, False])  # 0,1,2 train; 3,4 test
    y1 = torch.tensor([0, 1, 0, 0, 0])
    y2 = torch.tensor([0, 1, 0, 1, 1])  # diffère seulement sur 3,4 (test)
    f1 = compute_graph_features(ei, y1, train, n)
    f2 = compute_graph_features(ei, y2, train, n)
    assert torch.allclose(f1, f2), "fuite de labels test détectée"


def test_neighbor_ratio_uses_train_only():
    n = 3
    # triangle: chaque nœud voisin des deux autres
    ei = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
    y = torch.tensor([1, 1, 0])
    train = torch.tensor([True, True, False])  # node 2 = test
    feats = compute_graph_features(ei, y, train, n)
    # node 0 voisins train = {1}, label 1 -> ratio 1.0
    # node 2 voisins train = {0,1}, labels 1,1 -> ratio 1.0
    assert abs(feats[0, 1].item() - 1.0) < 1e-6
    assert abs(feats[2, 1].item() - 1.0) < 1e-6


def test_standardize_train_zero_mean_unit_std():
    x = torch.tensor([[1.0], [3.0], [5.0], [100.0]])
    train = torch.tensor([True, True, True, False])
    xs = standardize_features(x, train)
    tm = xs[train]
    assert abs(tm.mean().item()) < 1e-5
    assert abs(tm.std().item() - 1.0) < 1e-4

import torch

from src.models.multi_rel_gnn import MultiRelGNN


def _rand_edges(n, m):
    return torch.randint(0, n, (2, m), dtype=torch.long)


def test_forward_shape_and_uses_all_relations():
    n = 12
    eis = [_rand_edges(n, 20), _rand_edges(n, 15), _rand_edges(n, 10)]
    model = MultiRelGNN(in_dim=6, hidden_dim=8, out_dim=2, edge_index_list=eis, dropout=0.0)
    assert model.num_rel == 3
    x = torch.randn(n, 6)
    out = model(x, None)  # 2e arg ignoré
    assert out.shape == (n, 2)


def test_trains_via_train_gnn():
    """Compatible avec train_gnn (forward(x, edge_index)) malgré les relations internes."""
    from torch_geometric.data import Data
    from src.config import TrainConfig, set_seed
    from src.train.train_gnn import train_gnn, predict_scores

    set_seed(0)
    n = 30
    eis = [_rand_edges(n, 60), _rand_edges(n, 40)]
    x = torch.randn(n, 5)
    y = torch.tensor([0, 1] * (n // 2))
    mask = torch.ones(n, dtype=torch.bool)
    data = Data(x=x, edge_index=eis[0], y=y,
                train_mask=mask, val_mask=mask, test_mask=mask)
    model = MultiRelGNN(5, 8, 2, edge_index_list=eis, dropout=0.0)
    model = train_gnn(model, data, TrainConfig(epochs=3, patience=3))
    scores = predict_scores(model, data)
    assert scores.shape == (n,)
    assert ((scores >= 0) & (scores <= 1)).all()

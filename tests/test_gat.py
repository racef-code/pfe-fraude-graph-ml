import torch

from src.models.gat import GAT


def test_forward_shape():
    model = GAT(in_dim=8, hidden_dim=16, out_dim=2, heads=4, dropout=0.5)
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (10, 2)


def test_two_gat_layers():
    from torch_geometric.nn import GATConv
    model = GAT(in_dim=4, hidden_dim=8, out_dim=2, heads=2, dropout=0.0)
    convs = [m for m in model.modules() if isinstance(m, GATConv)]
    assert len(convs) == 2


def test_compatible_with_train_gnn_interface():
    """GAT doit s'entraîner via le même train_gnn que GraphSAGE (forward(x, edge_index))."""
    from torch_geometric.data import Data
    from src.config import TrainConfig, set_seed
    from src.train.train_gnn import train_gnn, predict_scores

    set_seed(0)
    x = torch.randn(20, 5)
    edge_index = torch.randint(0, 20, (2, 60))
    y = torch.tensor([0, 1] * 10)
    mask = torch.ones(20, dtype=torch.bool)
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=mask, val_mask=mask, test_mask=mask)
    model = GAT(5, 8, 2, heads=2, dropout=0.0)
    model = train_gnn(model, data, TrainConfig(epochs=3, patience=3))
    scores = predict_scores(model, data)
    assert scores.shape == (20,)
    assert ((scores >= 0) & (scores <= 1)).all()

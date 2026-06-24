import torch
from src.config import TrainConfig, set_seed
from src.data.cora import load_cora
from src.models.graphsage import GraphSAGE
from src.train.train_gnn import train_gnn, predict_scores


def test_train_gnn_learns_on_cora():
    set_seed(42)
    data = load_cora()
    model = GraphSAGE(data.x.shape[1], 64, int(data.y.max()) + 1, dropout=0.5)
    cfg = TrainConfig(epochs=60, patience=60)
    model = train_gnn(model, data, cfg)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    assert acc > 0.75  # critère succès Étape 1


def test_predict_scores_binary_shape():
    set_seed(0)
    import torch_geometric
    from torch_geometric.data import Data
    x = torch.randn(20, 5)
    edge_index = torch.randint(0, 20, (2, 40))
    y = torch.tensor([0, 1] * 10)
    mask = torch.ones(20, dtype=torch.bool)
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=mask, val_mask=mask, test_mask=mask)
    model = GraphSAGE(5, 8, 2, dropout=0.0)
    scores = predict_scores(model, data)
    assert scores.shape == (20,)
    assert ((scores >= 0) & (scores <= 1)).all()

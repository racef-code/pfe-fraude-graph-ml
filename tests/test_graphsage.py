import torch
from src.models.graphsage import GraphSAGE


def test_forward_shape():
    model = GraphSAGE(in_dim=8, hidden_dim=16, out_dim=3, dropout=0.5)
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (10, 3)


def test_two_sageconv_layers():
    from torch_geometric.nn import SAGEConv
    model = GraphSAGE(in_dim=4, hidden_dim=8, out_dim=2, dropout=0.0)
    convs = [m for m in model.modules() if isinstance(m, SAGEConv)]
    assert len(convs) == 2

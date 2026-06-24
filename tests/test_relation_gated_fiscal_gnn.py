from src.config import TrainConfig, set_seed
from src.data.fiscal_graph import build_fiscal_graph, make_synthetic_is_fiscal_tables
from src.data.transforms import standardize_hetero_features
from src.models.relation_gated_fiscal_gnn import RelationGatedFiscalGNN
from src.train.train_gnn import class_weights_from_labels
from src.train.train_hetero import predict_hetero_company_scores, train_hetero_company_gnn


def test_relation_gated_fiscal_gnn_forward_and_weights():
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=50, seed=13)
    data = standardize_hetero_features(build_fiscal_graph(nodes, edges, seed=13))
    model = RelationGatedFiscalGNN(data.metadata(), hidden_dim=8, out_dim=2, dropout=0.1)
    out = model(data.x_dict, data.edge_index_dict)
    assert out.shape == (data["company"].num_nodes, 2)
    weights = model.relation_weights(layer=2)
    assert weights
    assert any("company" in k for k in weights)


def test_relation_gated_fiscal_gnn_training_smoke():
    set_seed(14)
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=50, seed=14)
    data = standardize_hetero_features(build_fiscal_graph(nodes, edges, seed=14))
    model = RelationGatedFiscalGNN(data.metadata(), hidden_dim=8, out_dim=2, dropout=0.1)
    _ = model(data.x_dict, data.edge_index_dict)
    cfg = TrainConfig(hidden_dim=8, epochs=2, patience=2, dropout=0.1)
    cw = class_weights_from_labels(data["company"].y[data["company"].train_mask])
    model = train_hetero_company_gnn(model, data, cfg, class_weight=cw)
    scores = predict_hetero_company_scores(model, data)
    assert scores.shape == (data["company"].num_nodes,)

from src.data.fiscal_graph import make_synthetic_is_fiscal_tables
from src.data.real_fiscal import load_fiscal_csv_graph


def test_load_fiscal_csv_graph_roundtrip(tmp_path):
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=30, seed=4)
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    data = load_fiscal_csv_graph(nodes_path, edges_path, seed=4)
    assert "company" in data.node_types
    assert data["company"].num_nodes == 30
    assert data["company"].train_mask.any()

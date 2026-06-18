from src.analysis.graph_diagnostics import (
    company_pairs_for_relation,
    format_diagnostics,
    graph_diagnostics,
    relation_diagnostics,
)
from src.data.fiscal_graph import make_synthetic_is_fiscal_tables


def test_company_pairs_for_shared_support_relation():
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=40, seed=10)
    pairs = company_pairs_for_relation(nodes, edges, "has_director")
    assert pairs
    assert all(a != b for a, b in pairs)


def test_relation_diagnostics_contains_graph_advantage_metrics():
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=60, seed=11)
    row = relation_diagnostics(nodes, edges, "uses_accountant")
    assert row["raw_edges"] > 0
    assert 0.0 <= row["coverage"] <= 1.0
    assert 0.0 <= row["isolated_rate"] <= 1.0
    assert row["company_pairs"] > 0
    assert "fraud_neighbor_lift" in row


def test_graph_diagnostics_formats_all_relations():
    nodes, edges = make_synthetic_is_fiscal_tables(n_companies=50, seed=12)
    rows = graph_diagnostics(nodes, edges)
    text = format_diagnostics(rows)
    assert "Graph diagnostics by relation" in text
    assert "transaction" in text
    assert {r["relation"] for r in rows} == set(edges["type_relation"].unique())

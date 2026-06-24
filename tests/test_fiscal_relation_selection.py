from src.experiments.fiscal_relation_selection import (
    format_selection,
    run_one_selection,
    run_selection_benchmark,
)


def test_run_one_selection_returns_variants_and_models():
    res = run_one_selection(seed=0, n_companies=50, epochs=2)
    assert "all_relations" in res
    assert "audit_selected" in res
    assert "XGBoost+graph" in res["all_relations"]
    assert "RelationGatedFiscalGNN" in res["audit_selected"]
    assert "auc_pr" in res["all_relations"]["XGBoost+graph"]


def test_selection_benchmark_formats():
    summary, _ = run_selection_benchmark(seeds=(0,), n_companies=50, epochs=2)
    text = format_selection(summary, n_seeds=1)
    assert "Fiscal relation selection benchmark" in text
    assert "audit_selected" in text
    assert "WARNING" in text

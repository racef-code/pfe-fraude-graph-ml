from src.experiments.fiscal_ablation import (
    format_ablation,
    relation_importance,
    run_ablation_benchmark,
    run_one_ablation,
)


def test_run_one_ablation_returns_all_and_without_variants():
    res = run_one_ablation(seed=0, n_companies=50, epochs=2)
    assert "all_relations" in res
    assert "without_transaction" in res
    assert "auc_pr" in res["all_relations"]


def test_relation_importance_sorts_by_auc_pr_drop():
    summary, _ = run_ablation_benchmark(seeds=(0,), n_companies=50, epochs=2)
    rows = relation_importance(summary)
    assert rows
    drops = [r["auc_pr_drop"] for r in rows]
    assert drops == sorted(drops, reverse=True)
    text = format_ablation(summary)
    assert "Relation importance" in text

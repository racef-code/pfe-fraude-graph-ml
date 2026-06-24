from pathlib import Path

from src.data.ted_tax import load_ted_tables
from src.experiments.ted_tax_benchmark import format_ted_result, run_ted_benchmark

TED_SAMPLE = Path("/tmp/TED/TED/Data/T20H")


def test_load_ted_tables_if_sample_available():
    if not TED_SAMPLE.exists():
        return
    nodes, edges, meta = load_ted_tables(TED_SAMPLE, seed=0)
    assert len(nodes) > 0
    assert len(edges) > 0
    assert meta["stats"]["n_train"] > 0
    assert set(nodes["type"]).issuperset({"company", "item", "person"})


def test_ted_benchmark_smoke_if_sample_available():
    if not TED_SAMPLE.exists():
        return
    res = run_ted_benchmark(str(TED_SAMPLE), ratio="1v9", epochs=2, seed=0)
    assert "XGBoost+graph" in res
    assert "auc_pr" in res["RelationGatedFiscalGNN"]
    assert "tiny" in format_ted_result(res)

import importlib


def test_yelpchi_module_imports_without_dgl():
    """Module must import even when DGL absent (lazy import inside function)."""
    mod = importlib.import_module("src.data.yelpchi")
    assert hasattr(mod, "load_yelpchi")

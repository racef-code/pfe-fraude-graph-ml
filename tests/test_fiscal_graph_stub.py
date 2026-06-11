import pytest
import pandas as pd
from src.data.fiscal_graph import build_fiscal_graph


def test_stub_raises_not_implemented():
    nodes = pd.DataFrame({"id": [1, 2]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "type": ["client_fournisseur"]})
    with pytest.raises(NotImplementedError):
        build_fiscal_graph(nodes, edges)

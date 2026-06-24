import pandas as pd
import pytest

from src.data.fiscal_graph import build_fiscal_graph


def test_build_fiscal_graph_validates_required_columns():
    nodes = pd.DataFrame({"id": [1, 2]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "type": ["transaction"]})
    with pytest.raises(ValueError, match="nodes_df missing columns"):
        build_fiscal_graph(nodes, edges)

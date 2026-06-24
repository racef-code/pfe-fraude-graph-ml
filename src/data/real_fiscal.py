"""CSV adapter for real/anonymized fiscal IS tables.

Keeps I/O separate from graph construction so notebooks and scripts can load the
same contract documented in ``docs/contrat_donnees_is.md``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch_geometric.data import HeteroData

from src.data.fiscal_graph import build_fiscal_graph
from src.data.transforms import standardize_hetero_features


def load_fiscal_csv_graph(
    nodes_path: str | Path,
    edges_path: str | Path,
    *,
    seed: int = 42,
    standardize: bool = True,
) -> HeteroData:
    """Load `nodes.csv` + `edges.csv` and return a fiscal HeteroData graph."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    data = build_fiscal_graph(nodes_df, edges_df, seed=seed)
    if standardize:
        data = standardize_hetero_features(data, target_node_type="company")
    return data

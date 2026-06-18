"""Leak-free graph features for the heterogeneous fiscal/company graph.

These features answer the audit question: does graph *information* help a strong
tabular model before we claim that message passing is necessary?

Any feature using labels must use TRAIN labels only. Val/test labels must never
change the generated matrix.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.analysis.graph_diagnostics import company_pairs_for_relation
from src.data.fiscal_graph import RELATION_TYPES

COMPANY = "company"
LABEL = "label"


def _company_ids(nodes_df: pd.DataFrame) -> list[object]:
    return nodes_df.loc[nodes_df["type"] == COMPANY, "id"].tolist()


def _train_label_map(
    nodes_df: pd.DataFrame,
    train_company_ids: set[object],
    train_labels: dict[object, int] | None = None,
) -> dict[object, int]:
    if train_labels is not None:
        return {cid: int(y) for cid, y in train_labels.items() if cid in train_company_ids}
    companies = nodes_df[
        (nodes_df["type"] == COMPANY)
        & (nodes_df["id"].isin(train_company_ids))
        & (nodes_df[LABEL].notna())
    ]
    return dict(zip(companies["id"], companies[LABEL].astype(int)))


def compute_fiscal_graph_features(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    company_ids: list[object] | None = None,
    train_company_ids: set[object] | list[object],
    train_labels: dict[object, int] | None = None,
    relations: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build relation-aware graph features aligned to ``company_ids``.

    Parameters
    ----------
    nodes_df / edges_df:
        Fiscal graph tables.
    company_ids:
        Desired output order. If omitted, uses the order of company rows in
        ``nodes_df``.
    train_company_ids:
        Companies whose labels are allowed to influence label-derived features.
    train_labels:
        Optional explicit mapping for train labels. If absent, labels are read
        from ``nodes_df`` but only for ``train_company_ids``.
    relations:
        Relation types to featurize. Defaults to known fiscal relation types
        followed by any extra relation observed in ``edges_df``.

    Returns
    -------
    X_graph, feature_names
        ``X_graph`` is ``[n_company, n_features]``. For each relation, features
        are: log1p degree in projected company-company graph, coverage flag,
        train-neighbor count, and train-neighbor fraud ratio. Final global
        features summarize any relation.
    """
    if company_ids is None:
        company_ids = _company_ids(nodes_df)
    train_company_ids = set(train_company_ids)
    labels = _train_label_map(nodes_df, train_company_ids, train_labels)
    global_train_rate = sum(labels.values()) / max(len(labels), 1)

    observed = list(edges_df["type_relation"].dropna().unique()) if len(edges_df) else []
    if relations is None:
        relations = list(RELATION_TYPES) + [r for r in observed if r not in RELATION_TYPES]

    idx = {cid: i for i, cid in enumerate(company_ids)}
    columns: list[np.ndarray] = []
    names: list[str] = []
    any_degree = np.zeros(len(company_ids), dtype=np.float32)
    any_train_known = np.zeros(len(company_ids), dtype=np.float32)
    any_train_pos = np.zeros(len(company_ids), dtype=np.float32)

    for rel in relations:
        pairs = company_pairs_for_relation(nodes_df, edges_df, str(rel))
        neighbors: dict[object, set[object]] = defaultdict(set)
        for a, b in pairs:
            if a in idx and b in idx:
                neighbors[a].add(b)
                neighbors[b].add(a)

        degree = np.zeros(len(company_ids), dtype=np.float32)
        train_known = np.zeros(len(company_ids), dtype=np.float32)
        train_pos = np.zeros(len(company_ids), dtype=np.float32)
        for cid in company_ids:
            i = idx[cid]
            neigh = neighbors.get(cid, set())
            degree[i] = len(neigh)
            train_neigh = [n for n in neigh if n in labels]
            train_known[i] = len(train_neigh)
            train_pos[i] = sum(labels[n] == 1 for n in train_neigh)

        ratio = np.where(train_known > 0, train_pos / np.maximum(train_known, 1.0), global_train_rate)
        columns.extend([
            np.log1p(degree),
            (degree > 0).astype(np.float32),
            np.log1p(train_known),
            ratio.astype(np.float32),
        ])
        names.extend([
            f"{rel}__log_degree",
            f"{rel}__has_edge",
            f"{rel}__log_train_neighbor_count",
            f"{rel}__train_fraud_neighbor_ratio",
        ])
        any_degree += degree
        any_train_known += train_known
        any_train_pos += train_pos

    any_ratio = np.where(any_train_known > 0, any_train_pos / np.maximum(any_train_known, 1.0), global_train_rate)
    columns.extend([
        np.log1p(any_degree),
        (any_degree > 0).astype(np.float32),
        np.log1p(any_train_known),
        any_ratio.astype(np.float32),
    ])
    names.extend([
        "any_relation__log_degree",
        "any_relation__has_edge",
        "any_relation__log_train_neighbor_count",
        "any_relation__train_fraud_neighbor_ratio",
    ])

    if not columns:
        return np.zeros((len(company_ids), 0), dtype=np.float32), []
    return np.stack(columns, axis=1).astype(np.float32), names

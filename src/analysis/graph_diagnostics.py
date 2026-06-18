"""Graph diagnostics for the fiscal IS heterogeneous graph.

These diagnostics answer the PFE question behind the model scores: is there
actually exploitable relational signal in the graph, and which relation carries
it?
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

COMPANY = "company"
LABEL = "label"


def _company_labels(nodes_df: pd.DataFrame) -> dict[object, int]:
    companies = nodes_df[(nodes_df["type"] == COMPANY) & (nodes_df[LABEL].notna())]
    return dict(zip(companies["id"], companies[LABEL].astype(int)))


def _company_ids(nodes_df: pd.DataFrame) -> set[object]:
    return set(nodes_df.loc[nodes_df["type"] == COMPANY, "id"])


def company_pairs_for_relation(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, relation: str) -> set[tuple[object, object]]:
    """Project one heterogeneous relation into undirected company-company pairs.

    - company -> company relations become direct pairs.
    - company -> support-node relations become co-occurrence pairs: two companies
      sharing the same director/address/accountant/sector become neighbors.
    """
    company_ids = _company_ids(nodes_df)
    rel_edges = edges_df[edges_df["type_relation"] == relation]
    pairs: set[tuple[object, object]] = set()

    # Direct company-company relations.
    direct = rel_edges[
        rel_edges["id_source"].isin(company_ids) & rel_edges["id_cible"].isin(company_ids)
    ]
    for src, dst in zip(direct["id_source"], direct["id_cible"]):
        if src != dst:
            pairs.add(tuple(sorted((src, dst))))

    # Bipartite company-support relations projected by shared support node.
    bip = rel_edges[
        rel_edges["id_source"].isin(company_ids) & ~rel_edges["id_cible"].isin(company_ids)
    ]
    for _, group in bip.groupby("id_cible"):
        companies = sorted(set(group["id_source"]), key=str)
        for a, b in combinations(companies, 2):
            pairs.add(tuple(sorted((a, b))))
    return pairs


def relation_diagnostics(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, relation: str) -> dict:
    labels = _company_labels(nodes_df)
    company_ids = _company_ids(nodes_df)
    pairs = company_pairs_for_relation(nodes_df, edges_df, relation)

    connected = {c for pair in pairs for c in pair}
    labeled_pairs = [(a, b) for a, b in pairs if a in labels and b in labels]
    same = sum(int(labels[a] == labels[b]) for a, b in labeled_pairs)
    homophily = same / len(labeled_pairs) if labeled_pairs else float("nan")

    fraud_ids = {c for c, y in labels.items() if y == 1}
    nonfraud_ids = {c for c, y in labels.items() if y == 0}
    global_rate = len(fraud_ids) / max(len(labels), 1)

    # Neighbor fraud lift: among neighbors of fraud companies, how enriched are
    # fraud labels compared with the global labeled fraud rate?
    fraud_neighbors = set()
    nonfraud_neighbors = set()
    for a, b in pairs:
        if a in fraud_ids:
            fraud_neighbors.add(b)
        if b in fraud_ids:
            fraud_neighbors.add(a)
        if a in nonfraud_ids:
            nonfraud_neighbors.add(b)
        if b in nonfraud_ids:
            nonfraud_neighbors.add(a)
    labeled_fraud_neighbors = [n for n in fraud_neighbors if n in labels]
    neighbor_fraud_rate = (
        sum(labels[n] == 1 for n in labeled_fraud_neighbors) / len(labeled_fraud_neighbors)
        if labeled_fraud_neighbors else float("nan")
    )
    fraud_neighbor_lift = neighbor_fraud_rate / global_rate if global_rate and not np.isnan(neighbor_fraud_rate) else float("nan")

    degrees = {c: 0 for c in company_ids}
    for a, b in pairs:
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1

    return {
        "relation": relation,
        "raw_edges": int((edges_df["type_relation"] == relation).sum()),
        "company_pairs": int(len(pairs)),
        "coverage": len(connected) / max(len(company_ids), 1),
        "isolated_rate": 1.0 - (len(connected) / max(len(company_ids), 1)),
        "avg_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "homophily": float(homophily),
        "fraud_neighbor_lift": float(fraud_neighbor_lift),
        "neighbor_fraud_rate": float(neighbor_fraud_rate),
        "global_fraud_rate": float(global_rate),
    }


def graph_diagnostics(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> list[dict]:
    return [
        relation_diagnostics(nodes_df, edges_df, rel)
        for rel in sorted(edges_df["type_relation"].dropna().unique())
    ]


def format_diagnostics(rows: list[dict]) -> str:
    cols = [
        "relation", "raw_edges", "company_pairs", "coverage", "isolated_rate",
        "avg_degree", "homophily", "fraud_neighbor_lift",
    ]
    lines = ["Graph diagnostics by relation"]
    header = "".join(c.ljust(22) for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        line = ""
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = "nan" if np.isnan(v) else f"{v:.4f}"
            line += str(v).ljust(22)
        lines.append(line)
    return "\n".join(lines)

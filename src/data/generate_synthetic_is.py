"""Génère un dataset synthétique IS (selon les règles de make_synthetic_is_
fiscal_tables) et le SAUVE en local sous data/synthetic_is/.

Le dossier data/ est gitignoré → le dataset reste local, pas poussé sur GitHub.

Usage :
    python -m src.data.generate_synthetic_is [n_companies] [fraud_rate] [seed] [out_dir]
Exemple :
    python -m src.data.generate_synthetic_is 2000 0.15 42
"""
import os
import sys

from src.data.fiscal_graph import make_synthetic_is_fiscal_tables

DEFAULT_OUT = os.path.join("data", "synthetic_is")


def generate(n_companies=2000, fraud_rate=0.15, seed=42, out_dir=DEFAULT_OUT):
    nodes, edges = make_synthetic_is_fiscal_tables(
        n_companies=n_companies, fraud_rate=fraud_rate, seed=seed
    )
    os.makedirs(out_dir, exist_ok=True)
    nodes_path = os.path.join(out_dir, "nodes.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    n_comp = int((nodes["type"] == "company").sum())
    n_fraud = int((nodes["label"] == 1).sum())
    print(f"Dataset synthétique IS sauvé :")
    print(f"  {nodes_path}  ({len(nodes)} nœuds, dont {n_comp} entreprises)")
    print(f"  {edges_path}  ({len(edges)} arêtes)")
    print(f"  fraude : {n_fraud}/{n_comp} entreprises ({100 * n_fraud / n_comp:.1f}%)")
    print(f"  types de nœuds : {sorted(nodes['type'].unique())}")
    print(f"  types de relations : {sorted(edges['type_relation'].unique())}")
    return nodes_path, edges_path


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    fr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    out = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT
    generate(n, fr, seed, out)

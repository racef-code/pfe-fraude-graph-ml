"""Score un dataset fiscal et AFFICHE le classement des entreprises par risque.

Usage :
    python -m src.experiments.score_companies [nodes.csv] [edges.csv] [top_k]
Défaut : data/synthetic_is/{nodes,edges}.csv, top 20.

Pipeline : charge les CSV -> HeteroData -> entraîne FiscalHeteroGNN ->
score chaque entreprise -> tableau trié (rang, id, % de risque) -> sauve en CSV.
"""
import os
import sys

from src.analysis.score_report import build_company_score_report
from src.config import TrainConfig, set_seed
from src.data.real_fiscal import load_fiscal_csv_graph
from src.models.hetero_fiscal_gnn import FiscalHeteroGNN
from src.train.train_gnn import class_weights_from_labels
from src.train.train_hetero import train_hetero_company_gnn


def main():
    nodes = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "synthetic_is", "nodes.csv")
    edges = sys.argv[2] if len(sys.argv) > 2 else os.path.join("data", "synthetic_is", "edges.csv")
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    set_seed(42)
    data = load_fiscal_csv_graph(nodes, edges)
    cfg = TrainConfig(epochs=40, lr=0.01, hidden_dim=32, dropout=0.2, patience=40)

    company = data["company"]
    cw = class_weights_from_labels(company.y[company.train_mask])
    model = FiscalHeteroGNN(data.metadata(), hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    model = train_hetero_company_gnn(model, data, cfg, class_weight=cw)

    df = build_company_score_report(model, data)
    out = os.path.join("data", "scores_companies.csv")
    df.to_csv(out, index=False)

    print(f"Classement complet sauvé : {out}  ({len(df)} entreprises)\n")
    print(f"=== TOP {k} entreprises les plus à risque ===")
    print(df.head(k).to_string(index=False))


if __name__ == "__main__":
    main()

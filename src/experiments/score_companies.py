"""Score un dataset fiscal et AFFICHE le classement des entreprises par risque.

Usage :
    python -m src.experiments.score_companies [nodes.csv] [edges.csv] [top_k] [--split=...]
Défaut : data/synthetic_is/{nodes,edges}.csv, top 20, AUCUN filtre (affiche tout).

Filtre optionnel (non activé par défaut) :
    --split=test            n'affiche que les entreprises test (pour juger la perf)
    --split=non_enquete     n'affiche que les non-enquêtées (triage réel)
    --split=test,non_enquete   combine (jamais train/val = scores mémorisés)
Le rang affiché reste la position dans le classement GLOBAL. Le CSV sauvé contient
toujours TOUTES les entreprises (le filtre n'affecte que l'affichage).

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
    # positionnels (hors flags --) et flags --cle=valeur
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {}
    for a in sys.argv[1:]:
        if a.startswith("--"):
            key, _, val = a[2:].partition("=")
            flags[key] = val or True

    nodes = pos[0] if len(pos) > 0 else os.path.join("data", "synthetic_is", "nodes.csv")
    edges = pos[1] if len(pos) > 1 else os.path.join("data", "synthetic_is", "edges.csv")
    k = int(pos[2]) if len(pos) > 2 else 20
    split_filter = flags.get("split")  # None par défaut -> aucun filtre

    set_seed(42)
    data = load_fiscal_csv_graph(nodes, edges)
    cfg = TrainConfig(epochs=40, lr=0.01, hidden_dim=32, dropout=0.2, patience=40)

    company = data["company"]
    cw = class_weights_from_labels(company.y[company.train_mask])
    model = FiscalHeteroGNN(data.metadata(), hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    model = train_hetero_company_gnn(model, data, cfg, class_weight=cw)

    df = build_company_score_report(model, data)
    out = os.path.join("data", "scores_companies.csv")
    df.to_csv(out, index=False)  # CSV = toujours TOUTES les entreprises

    view = df
    title = f"=== TOP {k} entreprises les plus à risque (toutes) ==="
    if split_filter:
        wanted = [s.strip() for s in str(split_filter).split(",")]
        view = df[df["split"].isin(wanted)]
        title = f"=== TOP {k} les plus à risque [split={','.join(wanted)}] (rang = position globale) ==="

    print(f"Classement complet sauvé : {out}  ({len(df)} entreprises)\n")
    print(title)
    print(view.head(k).to_string(index=False))


if __name__ == "__main__":
    main()

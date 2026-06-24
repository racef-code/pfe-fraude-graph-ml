# Plan — prouver l'avantage Graph ML

Objectif : ne pas seulement « utiliser un GNN », mais démontrer quand et pourquoi le graphe apporte un signal de fraude IS au-delà des features tabulaires d'entreprise.

## 1. Questions à répondre

1. Les relations améliorent-elles AUC-PR / Recall@k vs XGBoost isolé ?
2. Le gain vient-il du message-passing ou de simples features de graphe ?
3. Quelles relations sont utiles : transaction, dirigeant, adresse, comptable, participation, secteur ?
4. Quelles relations sont bruitées ou trop sparse ?
5. Que faut-il demander en priorité dans les vraies données fiscales ?

## 2. Expériences ajoutées

### Diagnostics graphe

Module : `src/analysis/graph_diagnostics.py`

Mesures par relation :

- `coverage` : part d'entreprises touchées par la relation ;
- `isolated_rate` : part d'entreprises non atteintes ;
- `avg_degree` ;
- `homophily` ;
- `fraud_neighbor_lift` : enrichissement en fraude dans le voisinage des fraudeurs.

### Ablation relationnelle

Module : `src/experiments/fiscal_ablation.py`

Principe : entraîner le même `FiscalHeteroGNN` avec toutes les relations, puis retirer une relation à la fois :

```text
all_relations
without_transaction
without_has_director
without_registered_at
without_uses_accountant
without_participates_in
without_in_sector
```

Si retirer une relation baisse l'AUC-PR, cette relation porte du signal utile.

### Modèle relation-gated

Module : `src/models/relation_gated_fiscal_gnn.py`

`RelationGatedFiscalGNN` apprend un poids par type d'arête au lieu de sommer toutes les relations également. C'est une réponse directe au problème observé en fraude : certaines relations sont informatives, d'autres bruitées.

## 3. Commandes

```bash
# Diagnostics sur le synthétique
python - <<'PY'
from src.data.fiscal_graph import make_synthetic_is_fiscal_tables
from src.analysis.graph_diagnostics import graph_diagnostics, format_diagnostics
nodes, edges = make_synthetic_is_fiscal_tables(n_companies=240, seed=0)
print(format_diagnostics(graph_diagnostics(nodes, edges)))
PY

# Benchmark modèles : tabulaire vs GNN equal-sum vs GNN gated
python -m src.experiments.fiscal_synthetic 3 240 40

# Ablation des relations
python -m src.experiments.fiscal_ablation 3 240 40
```

## 4. Lecture attendue pour le mémoire

Le meilleur argument PFE n'est pas « le GNN est moderne ». C'est :

> Les relations fiscales créent un voisinage à risque mesurable ; le GNN exploite ce voisinage via message-passing ; les ablations montrent quelles relations apportent réellement le gain.

Sur données réelles, on devra répéter exactement ces diagnostics avant de conclure.

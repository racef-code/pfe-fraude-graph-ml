# Résultats — Graph advantage suite

> Données synthétiques uniquement. Ces résultats valident le protocole qui sera réutilisé sur données IS réelles ou dataset fiscal/corporate public.

## 1. Diagnostics relationnels

Commande :

```bash
python - <<'PY'
from src.data.fiscal_graph import make_synthetic_is_fiscal_tables
from src.analysis.graph_diagnostics import graph_diagnostics, format_diagnostics
nodes, edges = make_synthetic_is_fiscal_tables(n_companies=240, seed=0)
print(format_diagnostics(graph_diagnostics(nodes, edges)))
PY
```

Sortie vérifiée :

| Relation | Raw edges | Company pairs | Coverage | Isolated | Avg degree | Homophily | Fraud neighbor lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| has_director | 240 | 392 | 0.9417 | 0.0583 | 3.2667 | 0.7500 | 1.8919 |
| in_sector | 240 | 3652 | 1.0000 | 0.0000 | 30.4333 | 0.7525 | 0.9763 |
| participates_in | 39 | 39 | 0.2792 | 0.7208 | 0.3250 | 0.7179 | 3.9506 |
| registered_at | 240 | 1239 | 1.0000 | 0.0000 | 10.3250 | 0.7756 | 1.6129 |
| transaction | 776 | 760 | 1.0000 | 0.0000 | 6.3333 | 0.7763 | 1.6107 |
| uses_accountant | 240 | 2556 | 1.0000 | 0.0000 | 21.3000 | 0.7191 | 1.0016 |

## 2. Modèles : tabulaire vs Graph ML

Commande :

```bash
python -m src.experiments.fiscal_synthetic 3 240 40
```

| Model | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---:|---:|---:|---:|---:|
| XGBoost-company-features | 0.8246 ± 0.0146 | 0.5193 ± 0.0255 | 0.7260 ± 0.0362 | 0.6905 ± 0.0795 | 0.5714 ± 0.0000 |
| FiscalHeteroGNN | 0.9477 ± 0.0103 | 0.7724 ± 0.0940 | 0.7097 ± 0.0826 | 0.8396 ± 0.0595 | 0.6667 ± 0.0673 |
| RelationGatedFiscalGNN | **0.9570 ± 0.0214** | **0.7726 ± 0.1447** | **0.8248 ± 0.0482** | **0.9136 ± 0.0389** | 0.6667 ± 0.0673 |

Deltas vs XGBoost :

| Model | Metric | Delta | Positive seeds |
|---|---|---:|---:|
| FiscalHeteroGNN | AUC-PR | +0.2531 ± 0.1083 | 3/3 |
| FiscalHeteroGNN | AUC-ROC | +0.1231 ± 0.0183 | 3/3 |
| FiscalHeteroGNN | Recall@k | +0.0952 ± 0.0673 | 2/3 |
| RelationGatedFiscalGNN | AUC-PR | +0.2533 ± 0.1590 | 3/3 |
| RelationGatedFiscalGNN | AUC-ROC | +0.1324 ± 0.0335 | 3/3 |
| RelationGatedFiscalGNN | Recall@k | +0.0952 ± 0.0673 | 2/3 |

Relation-gated vs equal-sum HeteroGNN :

| Metric | Delta |
|---|---:|
| AUC-PR | +0.0002 ± 0.0509 |
| AUC-ROC | +0.0093 ± 0.0157 |
| F1-macro | +0.1151 ± 0.0592 |
| GMean | +0.0740 ± 0.0518 |

## 3. Relation ablation

Commande :

```bash
python -m src.experiments.fiscal_ablation 3 240 40
```

| Variant | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---:|---:|---:|---:|---:|
| all_relations | 0.9477 ± 0.0103 | 0.7724 ± 0.0940 | 0.7097 ± 0.0826 | 0.8396 ± 0.0595 | 0.6667 ± 0.0673 |
| without_transaction | 0.9373 ± 0.0243 | 0.7293 ± 0.1236 | 0.7082 ± 0.0723 | 0.8407 ± 0.0535 | 0.6190 ± 0.0673 |
| without_uses_accountant | 0.9048 ± 0.0427 | 0.7313 ± 0.1356 | 0.7676 ± 0.0962 | 0.8136 ± 0.0690 | 0.7143 ± 0.1166 |
| without_registered_at | 0.9396 ± 0.0460 | 0.7607 ± 0.1521 | 0.7181 ± 0.0485 | 0.8366 ± 0.0623 | 0.7143 ± 0.1166 |
| without_has_director | 0.9501 ± 0.0339 | 0.8055 ± 0.1164 | 0.7601 ± 0.0791 | 0.8607 ± 0.0608 | 0.6667 ± 0.0673 |
| without_participates_in | 0.9477 ± 0.0289 | 0.8090 ± 0.1425 | 0.8190 ± 0.0575 | 0.8702 ± 0.0335 | 0.8095 ± 0.0673 |
| without_in_sector | 0.9466 ± 0.0559 | 0.8155 ± 0.1660 | 0.7717 ± 0.0709 | 0.8335 ± 0.0958 | 0.7619 ± 0.1347 |

Importance by AUC-PR drop vs all relations:

| Relation | AUC-PR drop | Interpretation |
|---|---:|---|
| transaction | +0.0431 | useful |
| uses_accountant | +0.0411 | useful for AUC-PR/AUC-ROC, but threshold metrics mixed |
| registered_at | +0.0117 | small useful signal |
| has_director | -0.0331 | noisy in this synthetic run |
| participates_in | -0.0367 | high lift but sparse/noisy; needs better weighting |
| in_sector | -0.0431 | mostly broad context/noise |

## 4. Upgrade identified

Equal-sum HeteroGNN is too naive because it treats all relations equally. The relation-gated model improves F1/GMean strongly and slightly improves AUC-ROC, but AUC-PR remains similar. This suggests the next serious upgrade should be **relation selection/regularization** or **attention constrained by diagnostics**, not simply deeper GNN layers.

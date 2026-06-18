# Résultats — Baseline XGBoost + graph features fiscales

> Résultat demandé après audit Claude Code : ajouter la baseline décisive qui sépare **information de graphe** et **message passing GNN**.

## Pourquoi cette baseline existe

Avant cette étape, le benchmark comparait :

| Modèle | Features tabulaires | Graphe | Message passing |
|---|---:|---:|---:|
| XGBoost-company-features | ✅ | ❌ | ❌ |
| FiscalHeteroGNN | ✅ | ✅ | ✅ |
| RelationGatedFiscalGNN | ✅ | ✅ | ✅ |

Cette comparaison ne prouvait pas que le GNN était nécessaire. Elle prouvait seulement que **le graphe peut contenir de l'information**.

La nouvelle baseline ajoute :

| Modèle | Features tabulaires | Features de graphe | Message passing |
|---|---:|---:|---:|
| XGBoost-company+graph-features | ✅ | ✅ | ❌ |

Elle répond à la critique :

> Peut-être que des features de graphe simples suffisent ; pourquoi utiliser un GNN ?

## Implémentation

Nouveau module :

```text
src/features/fiscal_graph_features.py
```

Features calculées par relation :

- `relation__log_degree`
- `relation__has_edge`
- `relation__log_train_neighbor_count`
- `relation__train_fraud_neighbor_ratio`

Features globales :

- `any_relation__log_degree`
- `any_relation__has_edge`
- `any_relation__log_train_neighbor_count`
- `any_relation__train_fraud_neighbor_ratio`

Règle anti-fuite :

> Toute feature qui utilise les labels de fraude n'utilise que les labels du train. Changer les labels val/test ne doit pas modifier la matrice de features.

Tests ajoutés :

```text
tests/test_fiscal_graph_features.py
```

## Résultat smoke vérifié

Commande :

```bash
python -m src.experiments.fiscal_synthetic 3 240 40
```

Résultat :

| Model | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---:|---:|---:|---:|---:|
| XGBoost-company-features | 0.8246 ± 0.0146 | 0.5193 ± 0.0255 | 0.7260 ± 0.0362 | 0.6905 ± 0.0795 | 0.5714 ± 0.0000 |
| XGBoost-company+graph-features | **0.9628 ± 0.0431** | **0.8994 ± 0.1010** | **0.8546 ± 0.0553** | 0.8285 ± 0.0687 | **0.8095 ± 0.1782** |
| FiscalHeteroGNN | 0.9477 ± 0.0103 | 0.7724 ± 0.0940 | 0.7097 ± 0.0826 | 0.8396 ± 0.0595 | 0.6667 ± 0.0673 |
| RelationGatedFiscalGNN | 0.9570 ± 0.0214 | 0.7726 ± 0.1447 | 0.8248 ± 0.0482 | **0.9136 ± 0.0389** | 0.6667 ± 0.0673 |

Deltas importants :

| Comparison | AUC-PR delta | Interpretation |
|---|---:|---|
| XGBoost+graph - XGBoost-company | +0.3801 ± 0.1097 | Le graphe contient beaucoup de signal dans le synthétique |
| FiscalHeteroGNN - XGBoost+graph | -0.1270 ± 0.0257 | Le GNN ne bat pas encore la baseline graph-features |
| RelationGatedFiscalGNN - XGBoost+graph | -0.1268 ± 0.0562 | Le modèle gated améliore certains seuils mais pas AUC-PR |

## Nouvelle conclusion honnête

Avant :

> Le GNN apporte un gros gain vs XGBoost company-only.

Après audit + baseline :

> Le graphe apporte un gros gain, mais sur le benchmark synthétique actuel, des features de graphe simples avec XGBoost battent les GNNs en AUC-PR. Donc la prochaine question scientifique n'est pas encore “quel GNN est meilleur”, mais “dans quelles conditions le message passing bat-il une baseline graph-features forte ?”.

Cette conclusion est plus solide et plus défendable.

## Conséquence pour le PFE

Le projet reste dans la bonne direction, mais le mémoire doit être formulé ainsi :

1. La fraude fiscale entreprise est naturellement relationnelle.
2. Les diagnostics montrent si une relation porte du signal.
3. XGBoost+graph-features teste si le graphe aide sans GNN.
4. Le GNN est justifié seulement s'il apporte au-dessus de cette baseline forte.
5. Sur synthétique actuel, le graphe aide fortement, mais le message passing n'est pas encore supérieur.

## Prochaine étape technique

Créer un benchmark plus robuste où le GNN peut gagner ou perdre honnêtement :

```bash
python -m src.experiments.fiscal_synthetic 10 2000 40
```

Puis tester :

- relation selection ;
- harder synthetic data with stronger tabular signal ;
- label noise / hidden fraud labels ;
- public corporate/fiscal dataset if available.

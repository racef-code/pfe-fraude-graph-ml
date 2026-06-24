# Corrections méthodologiques appliquées après audit

Ce fichier liste les corrections appliquées suite à l'audit Claude Code.

## 1. Baseline XGBoost + graph features

Statut : ✅ appliqué

Fichiers :

```text
src/features/fiscal_graph_features.py
tests/test_fiscal_graph_features.py
src/experiments/fiscal_synthetic.py
docs/RESULTATS_GRAPH_FEATURE_BASELINE.md
```

But : séparer deux questions :

1. Le graphe contient-il de l'information utile ?
2. Le message passing GNN apporte-t-il plus qu'une baseline XGBoost avec features de graphe ?

Conclusion actuelle sur le synthétique :

> Le graphe aide fortement, mais XGBoost+graph-features bat les GNNs en AUC-PR sur ce benchmark. Donc le prochain objectif est de comprendre quand le message passing dépasse une baseline graph-features forte.

## 2. Anti-fuite sur les features de graphe

Statut : ✅ appliqué

Règle : toute feature qui utilise les labels utilise uniquement les labels des entreprises train.

Test ajouté :

```text
test_fiscal_graph_features_do_not_leak_test_labels
```

Ce test change uniquement les labels test et vérifie que les features restent identiques.

## 3. Diagnostics corrigés : homophily vs null baseline

Statut : ✅ appliqué

Avant, on affichait seulement :

```text
homophily
```

Mais avec un taux de fraude de 15 %, l'homophilie attendue au hasard est déjà :

```text
p² + (1-p)² ≈ 0.745
```

Donc une homophilie brute autour de 0.75 ne prouve rien.

Maintenant les diagnostics affichent :

```text
homophily
homophily_null
homophily_over_null
fraud_neighbor_lift
```

Le signal principal à discuter est plutôt :

```text
fraud_neighbor_lift
```

## 4. Diagnostics label-scope train-only

Statut : ✅ appliqué

`graph_diagnostics(...)` accepte maintenant :

```python
label_company_ids=train_ids
```

Cela permet de calculer les diagnostics de label sans regarder les labels val/test.

Test ajouté :

```text
test_relation_diagnostics_label_scope_prevents_test_label_leakage
```

## 5. Discipline d'évaluation

Statut : partiellement appliqué ✅/🟠

Le benchmark affiche maintenant un warning si le nombre de seeds est inférieur à 10 :

```text
WARNING: audit-grade reporting should use >=10 seeds; this run is a smoke/iteration run.
```

Il reste à lancer un run plus lourd, recommandé :

```bash
python -m src.experiments.fiscal_synthetic 10 2000 40
```

Ce run peut être plus long. Il doit servir aux résultats sérieux, pas aux itérations rapides.

## 6. Encore à faire

Non encore appliqué :

1. relation selection experiment ;
2. early stopping sur validation AUC-PR au lieu de val loss ;
3. temporal split pour données réelles ;
4. inductive evaluation ;
5. dataset public plus proche fiscal/corporate : TED T20H/T15S ou KeHGN-R.

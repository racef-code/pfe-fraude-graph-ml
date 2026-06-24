# Résultats — benchmark synthétique renforcé post-audit

Commande vérifiée :

```bash
python -m src.experiments.fiscal_synthetic 10 2000 20
```

Pourquoi ce run : l'audit Claude Code demandait de ne pas s'appuyer sur `3 seeds / 240 entreprises`. Ce run utilise :

- 10 seeds ;
- 2000 entreprises synthétiques ;
- 20 epochs pour garder le coût raisonnable ;
- baseline décisive `XGBoost-company+graph-features`.

## Résultats

| Model | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---:|---:|---:|---:|---:|
| XGBoost-company-features | 0.7843 ± 0.0310 | 0.4651 ± 0.0460 | 0.6726 ± 0.0289 | 0.6045 ± 0.0373 | 0.4533 ± 0.0547 |
| XGBoost-company+graph-features | **0.9748 ± 0.0105** | **0.9147 ± 0.0265** | **0.9068 ± 0.0213** | **0.9070 ± 0.0279** | **0.8467 ± 0.0340** |
| FiscalHeteroGNN | 0.9339 ± 0.0191 | 0.7965 ± 0.0506 | 0.8075 ± 0.0303 | 0.8551 ± 0.0293 | 0.7167 ± 0.0357 |
| RelationGatedFiscalGNN | 0.9516 ± 0.0125 | 0.8356 ± 0.0405 | 0.8263 ± 0.0255 | 0.8800 ± 0.0298 | 0.7417 ± 0.0423 |

## Deltas importants

| Comparison | AUC-PR delta | AUC-ROC delta | Recall@k delta | Positive seeds |
|---|---:|---:|---:|---:|
| XGBoost+graph - XGBoost-company | +0.4496 ± 0.0594 | +0.1905 ± 0.0351 | +0.3933 ± 0.0688 | 10/10 |
| FiscalHeteroGNN - XGBoost-company | +0.3314 ± 0.0562 | +0.1496 ± 0.0299 | +0.2633 ± 0.0657 | 10/10 |
| RelationGatedFiscalGNN - XGBoost-company | +0.3705 ± 0.0566 | +0.1674 ± 0.0254 | +0.2883 ± 0.0548 | 10/10 |
| FiscalHeteroGNN - XGBoost+graph | -0.1182 ± 0.0408 | -0.0409 ± 0.0181 | -0.1300 ± 0.0420 | 0/10 |
| RelationGatedFiscalGNN - XGBoost+graph | -0.0791 ± 0.0332 | -0.0232 ± 0.0155 | -0.1050 ± 0.0573 | 0/10 AUC-PR |
| RelationGated - equal-sum HeteroGNN | +0.0391 ± 0.0260 | +0.0177 ± 0.0143 | — | 10/10 AUC-PR |

## Conclusion honnête

1. Le graphe contient bien du signal sur le synthétique : `XGBoost+graph` bat largement `XGBoost-company`.
2. Le modèle relation-gated bat le HeteroGNN equal-sum sur 10/10 seeds en AUC-PR.
3. Mais les GNNs ne battent pas encore la baseline `XGBoost+graph-features`.

Donc la bonne question de recherche devient :

> Quand le message passing est-il nécessaire par rapport à de fortes features de graphe ?

Ce résultat est plus défendable qu'une simple victoire contre XGBoost company-only.

# Résultats — prototype synthétique IS hétérogène

> Données synthétiques uniquement. Ce tableau valide l'ingénierie de la pipeline fiscale hétérogène ; il ne constitue pas une preuve scientifique sur la fraude fiscale réelle.

Commande :

```bash
python -m src.experiments.fiscal_synthetic 3 240 40
```

Résultat vérifié le 2026-06-17 :

| Modèle | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---:|---:|---:|---:|---:|
| XGBoost-company-features | 0.8246 ± 0.0146 | 0.5193 ± 0.0255 | 0.7260 ± 0.0362 | 0.6905 ± 0.0795 | 0.5714 ± 0.0000 |
| FiscalHeteroGNN | **0.9686 ± 0.0419** | **0.8638 ± 0.1802** | **0.8311 ± 0.0926** | **0.9144 ± 0.0690** | **0.8571 ± 0.1166** |

Deltas appariés `FiscalHeteroGNN - XGBoost-company-features` :

| Métrique | Delta moyen | Seeds positives |
|---|---:|---:|
| AUC-PR | +0.3445 ± 0.1944 | 3/3 |
| AUC-ROC | +0.1440 ± 0.0538 | 3/3 |
| Recall@k | +0.2857 ± 0.1166 | 3/3 |

## Interprétation correcte

Le générateur synthétique est conçu pour être non trivial : le signal tabulaire est bruité, tandis que les relations dirigeant/adresse/comptable/transactions créent des communautés à risque. Le fait que `FiscalHeteroGNN` batte XGBoost ici montre que la pipeline est capable d'exploiter une information relationnelle quand elle existe.

À ne pas dire : « le GNN détecte mieux la fraude fiscale réelle ». La vraie conclusion viendra seulement après données fiscales réelles ou dataset public fiscal/corporate proche comme TED ou KeHGN-R.

# Dataset alternatif — TED tax evasion heterogeneous graph

## Pourquoi TED est meilleur que YelpChi pour notre PFE

YelpChi reste utile comme benchmark public de fraude, mais ce n'est pas fiscal/company. TED est beaucoup plus proche de notre sujet :

- domaine : **tax evasion detection** ;
- structure : **heterogeneous graph** ;
- nœud cible : company/taxpayer ;
- relations : transactions, personnes, items, événements ;
- papier : *TED: Related Party Transaction guided Tax Evasion Detection on Heterogeneous Graph*, DMKD 2025 ;
- code/data : <https://github.com/yimingxu24/TED>.

Le repo annonce deux datasets :

| Dataset | Node types | Nodes | Edge types | Edges | Attributes | Label type |
|---|---:|---:|---:|---:|---:|---:|
| T20H | 4 | 112,015 | 6 | 198,903 | 300 | 2 |
| T15S | 2 | 132,522 | 2 | 467,273 | 300 | 2 |

## Ce qui a été adapté

Nouveaux fichiers :

```text
src/data/ted_tax.py
src/experiments/ted_tax_benchmark.py
tests/test_ted_tax.py
```

Mapping T20H :

| TED type | Our type |
|---:|---|
| 0 | company |
| 1 | event |
| 2 | item |
| 3 | person |

Relations T20H :

| TED link | Relation |
|---:|---|
| 0 | transaction |
| 1 | info_change |
| 2 | company_sell_item |
| 3 | company_buy_item |
| 4 | person_company |
| 5 | belong_to |

Le benchmark adapté lance :

```text
XGBoost-company
XGBoost+graph
FiscalHeteroGNN
RelationGatedFiscalGNN
```

## Limite importante

Le clone GitHub récupéré localement contient seulement un **mini-échantillon** T20H :

```text
31 nodes
70 edges
5 train labels
1 val label
4 test labels
```

Donc les résultats ci-dessous sont un **adapter smoke test**, pas un benchmark scientifique. Le full dataset doit être récupéré via le lien Baidu fourni dans le repo TED.

## Commande exécutée

```bash
python -m src.experiments.ted_tax_benchmark /tmp/TED/TED/Data/T20H 1v9 40
```

## Résultat sur le mini-échantillon GitHub

| Model | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---:|---:|---:|---:|---:|
| XGBoost-company | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 0.0000 |
| XGBoost+graph | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 0.0000 |
| FiscalHeteroGNN | 1.0000 | 1.0000 | 0.3333 | 0.0000 | 1.0000 |
| RelationGatedFiscalGNN | 1.0000 | 1.0000 | 0.3333 | 0.0000 | 1.0000 |

## Interpretation

These numbers are **not reliable** because test has only 4 companies. AUC can jump to 1.0 by ranking two positives above two negatives, while F1/GMean remain poor at threshold 0.5.

What is valuable here is not the metric; it is that:

1. the TED format was parsed ;
2. labels/splits were adapted ;
3. our XGBoost and heterogeneous GNN pipeline runs on a real tax-evasion graph format ;
4. we can reuse the same code when full T20H/T15S is downloaded.

## Next step for TED

Download the full TED data from the official repo link, then run:

```bash
python -m src.experiments.ted_tax_benchmark /path/to/T20H 1v9 40
python -m src.experiments.ted_tax_benchmark /path/to/T20H 5v5 40
```

For the PFE, TED is currently the best alternative dataset direction because it is tax-specific and heterogeneous.

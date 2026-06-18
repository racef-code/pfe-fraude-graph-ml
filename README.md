# PFE — Détection de fraude fiscale par Graph ML

Outil de triage : score de risque de fraude (GraphSAGE) vs baseline sans graphe (XGBoost).

## Structure
- `src/` — modules (source de vérité)
- `notebooks/` — wrappers Colab (Étape 1 Cora, Étape 2 YelpChi)
- `docs/superpowers/` — spec + plan
- `tests/` — suite pytest (CPU-local)

## Dev local (CPU)
```
pip install -r requirements.txt
pytest -v
```
DGL/YelpChi tournent seulement sur Colab (`notebooks/02_yelpchi_pipeline.ipynb`).

## Question de recherche
GraphSAGE / GNN hétérogène (relations) vs XGBoost (isolé) sur AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k.

## Direction actuelle
- YelpChi = banc d'essai public pour valider la pipeline Graph ML.
- Cible PFE = **fraude IS sur entreprises** avec graphe hétérogène `company` + entités de contexte (`person`, `address`, `accountant`, `sector`).
- Prototype fiscal : `src/data/fiscal_graph.py`, `src/data/real_fiscal.py`, `src/models/hetero_fiscal_gnn.py`, `src/experiments/fiscal_synthetic.py`.
- Contrat données réelles : `docs/contrat_donnees_is.md`.
- Recherche datasets : `docs/recherche_datasets_is.md`.
- Résultats prototype synthétique : `docs/RESULTATS_SYNTHETIC_IS.md`.

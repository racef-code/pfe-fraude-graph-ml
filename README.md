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
GraphSAGE (relations) vs XGBoost (isolé) sur AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k.

# Design — PFE : Détection de fraude fiscale par Graph ML

**Date** : 2026-06-11
**Statut** : approuvé (design), en attente revue du spec écrit

---

## 1. Objectif

Outil de triage attribuant un **score de risque de fraude** (0–1) à des contribuables,
en exploitant leurs **relations** (graphe) en plus de leurs caractéristiques individuelles.
Aide à la décision, pas décision. Tâche = classification binaire déséquilibrée.

**Question de recherche** : exploiter les relations (GraphSAGE) améliore-t-il la détection
vs une approche isolée (XGBoost) ?

Référence cahier des charges : `C:/Users/rachi/Downloads/FEUILLE_DE_ROUTE.md`.

---

## 2. Décisions de design (confirmées)

| # | Décision | Détail |
|---|----------|--------|
| D1 | **Modules `.py` = source de vérité** | Notebooks = wrappers Colab fins qui importent les modules. Zéro logique dupliquée. |
| D2 | **DGL/YelpChi = Colab-only** | `yelpchi.py` s'importe correctement mais ne tourne réellement que sur Colab (DGL pénible sur Windows local). Cora tourne en CPU local. |
| D3 | **Étape 3 fiscal = stub** | `fiscal_graph.py` : signature `tables → HeteroData` + TODOs explicites, pas de corps fonctionnel. Données réelles pas encore arrivées. |
| D4 | **GraphSAGE homogène** | Graphe homogène pour Cora + YelpChi. `HeteroData` seulement dans le stub fiscal. |
| D5 | **Modèle principal** | GraphSAGE 2 couches (`SAGEConv`, PyG). |
| D6 | **Baseline** | XGBoost sur les **mêmes features** (`data.x`), sans `edge_index`. |
| D7 | **Environnement** | Colab (GPU) pour entraînement ; device auto `cuda`/`cpu`. |

---

## 3. Architecture

```
pfe-fraude-graph-ml/
├── src/
│   ├── config.py            # seed, hyperparams, device auto (cuda/cpu)
│   ├── data/
│   │   ├── cora.py          # Étape 1 : Cora → PyG Data
│   │   ├── yelpchi.py       # Étape 2 : DGL FraudDataset → PyG Data (Colab-only)
│   │   └── fiscal_graph.py  # Étape 3 : 3 tables → HeteroData (STUB)
│   ├── models/
│   │   └── graphsage.py     # GraphSAGE 2 couches
│   ├── train/
│   │   ├── train_gnn.py     # boucle + pos_weight (déséquilibre)
│   │   └── baseline_xgb.py  # XGBoost, mêmes features, SANS graphe
│   └── eval/
│       └── metrics.py       # AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k
├── notebooks/
│   ├── 01_cora_demo.ipynb       # Étape 1
│   └── 02_yelpchi_pipeline.ipynb # Étape 2 (GraphSAGE vs XGBoost)
├── requirements.txt
└── README.md
```

---

## 4. Contrats des modules (interfaces)

### `config.py`
- `SEED = 42`, fonction `set_seed(seed)`.
- `DEVICE` = `cuda` si dispo sinon `cpu`.
- Dataclass `TrainConfig` (hidden_dim, lr, epochs, weight_decay, dropout).

### `data/cora.py`
- `load_cora() -> Data` : renvoie PyG `Data` (`x`, `edge_index`, `y`, `train/val/test_mask`).
- Multi-classe (7 classes) — sert juste à valider la mécanique GNN.

### `data/yelpchi.py` (Colab-only)
- `load_yelpchi(relations="homo") -> Data` : charge `dgl.data.FraudDataset('yelp')`,
  convertit le graphe en PyG `Data` homogène, crée des masks train/val/test stratifiés.
- Binaire, déséquilibré. Expose `data.x`, `data.edge_index`, `data.y`.

### `data/fiscal_graph.py` (STUB)
- `build_fiscal_graph(nodes_df, edges_df) -> HeteroData` : signature + docstring listant
  nœuds (entreprise/particulier), features attendues, types de relations
  (client-fournisseur, dirigeant, adresse, comptable, participation). Corps = `raise NotImplementedError` + TODOs.

### `models/graphsage.py`
- `class GraphSAGE(nn.Module)` : 2 `SAGEConv`, ReLU + dropout entre couches.
  `__init__(in_dim, hidden_dim, out_dim, dropout)`, `forward(x, edge_index) -> logits`.

### `train/train_gnn.py`
- `train_gnn(model, data, cfg, pos_weight=None) -> trained_model` : boucle full-batch,
  `CrossEntropyLoss`/`BCEWithLogitsLoss(pos_weight=...)` pour le binaire, early-stop sur val.
- `predict_scores(model, data) -> np.ndarray` : scores classe positive.

### `train/baseline_xgb.py`
- `train_xgb(X_train, y_train, scale_pos_weight) -> model`.
- `predict_xgb(model, X) -> scores`. Entrée = `data.x` (numpy), aucune info de graphe.

### `eval/metrics.py`
- `compute_metrics(y_true, scores, k=...) -> dict` : AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k.
- `print_comparison(results: dict[str, dict])` : tableau GraphSAGE vs XGBoost (gabarit PC-GNN).

---

## 5. Flux de données (par expérience)

```
load_*()  →  PyG Data
                ├─ GraphSAGE : train_gnn(pos_weight) → predict_scores ┐
                │                                                      ├→ metrics.compute_metrics
                └─ XGBoost   : data.x → train_xgb → predict_xgb ───────┘
                                                                        → print_comparison
```

GraphSAGE voit `x` + `edge_index`. XGBoost voit seulement `x`. Mêmes splits, mêmes métriques
→ comparaison équitable = réponse à la question de recherche.

---

## 6. Gestion du déséquilibre (piège n°2 du cahier des charges)

- GNN : `pos_weight = n_neg / n_pos` dans `BCEWithLogitsLoss`.
- XGBoost : `scale_pos_weight = n_neg / n_pos`.
- **Jamais d'accuracy** comme métrique principale (piège n°1).

---

## 7. Notebooks (wrappers Colab)

Chaque notebook : (1) `pip install` deps, (2) `git clone` / `sys.path` vers `src/`,
(3) appels aux fonctions, (4) affichage tableaux + courbes. Aucune logique métier dans le notebook.

- `01_cora_demo.ipynb` : `load_cora` → `GraphSAGE` → train → accuracy (mécanique seulement).
- `02_yelpchi_pipeline.ipynb` : `load_yelpchi` → GraphSAGE vs XGBoost → 5 KPIs → tableau comparatif.

---

## 8. Tests / validation

- `cora.py` + `graphsage.py` + `train_gnn.py` + `metrics.py` : testables en CPU local (Cora petit).
- Critère de succès Étape 1 : GraphSAGE atteint accuracy raisonnable (>0.75) sur test Cora.
- Critère Étape 2 (Colab) : pipeline produit le tableau 5-KPI GraphSAGE vs XGBoost sans erreur.
- `yelpchi.py` : validation manuelle sur Colab (pas de CI Windows).

---

## 9. Hors périmètre (YAGNI)

- Pas de reproduction PC-GNN / CARE-GNN (citation seulement).
- Pas de base graphe (Arango/Neo4j) — graphe en mémoire PyG.
- Pas de corps fonctionnel pour le graphe fiscal tant que les vraies données ne sont pas là.
- TVA, blanchiment, banque : exclus.

---

## 10. Limites à documenter dans le mémoire

- **Biais de sélection** : seuls les enquêtés ont un label ; les non-enquêtés sont "inconnus", pas négatifs.
- Validation sur datasets publics (YelpChi) avant données fiscales réelles.

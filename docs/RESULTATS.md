# Résultats expérimentaux

## Étape 1 — Cora (validation mécanique GNN)

GraphSAGE 2 couches, test accuracy = **0.798** (>0.75). Sert seulement à valider
la pipeline PyG, pas à répondre à la question de recherche.

## Étape 2 — YelpChi (GraphSAGE vs XGBoost)

Dataset : 45 954 nœuds, 7 693 958 arêtes (relation `homo`), 32 features,
taux de fraude ≈ 14,5 %. Splits stratifiés 60/20/20, seed 42.
Gestion du déséquilibre : class weights (GNN) + `scale_pos_weight` (XGBoost).

| Modèle | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---|---|---|---|---|
| GraphSAGE | 0.8012 | 0.4482 | 0.6301 | 0.7302 | 0.4449 |
| **XGBoost** | **0.9417** | **0.8085** | **0.8263** | **0.8588** | **0.7236** |

### Lecture

Sur YelpChi, **XGBoost (sans graphe) bat largement GraphSAGE (avec graphe)**.

Ce n'est pas un échec de l'implémentation mais un résultat connu de la littérature :
le graphe `homo` de YelpChi est **hétérophile** — les fraudeurs sont camouflés au
milieu de voisins non-fraudeurs. GraphSAGE, qui agrège les voisins de façon
uniforme, **lisse** le signal discriminant (over-smoothing / camouflage). C'est
précisément la motivation des GNN spécialisés fraude (**CARE-GNN**, **PC-GNN**),
qui pondèrent/échantillonnent les voisins pour contrer ce camouflage.

### Réponse à la question de recherche (sur ce banc d'essai)

Exploiter les relations avec un **GNN générique (GraphSAGE)** n'améliore PAS la
détection ici ; un modèle tabulaire fort (XGBoost) sur les seules features fait
mieux. Conclusion nuancée pour le mémoire : *le graphe seul ne suffit pas — la
valeur vient de la façon de l'exploiter*, d'où l'intérêt des méthodes dédiées.

### Limites

- Un seul dataset public, GraphSAGE vanilla, hyperparamètres par défaut.
- Le résultat ne se transpose pas mécaniquement aux données fiscales réelles
  (structure de graphe différente, biais de sélection des labels).
- Reproductible : `notebooks/02_yelpchi_pipeline.ipynb` (≈15 min CPU).

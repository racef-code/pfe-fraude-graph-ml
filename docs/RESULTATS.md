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

### Pourquoi l'écart est crédible (analyse)

- L'écart est **cohérent sur les 5 métriques**, pas du bruit. Le plus parlant :
  **AUC-PR 0.81 vs 0.45** (×1.8) — c'est la métrique reine en fraude déséquilibrée.
- Les 32 features YelpChi sont des **features comportementales** déjà très
  discriminantes (littérature spam Yelp). XGBoost les exploite directement ;
  GraphSAGE les **dilue** en moyennant des voisins majoritairement non-fraudeurs.
- Le graphe `homo` est dense (degré moyen ≈ 167) → sur-lissage marqué.

### Réserves à assumer dans le mémoire (ne pas cacher)

1. **Un seul seed.** Pas de barre d'erreur → résultat attaquable par un jury.
   **À refaire sur ≥5 seeds, rapporter moyenne ± écart-type.**
2. **Pas de tuning du GNN** (64 hidden, 2 couches, full-batch, relation `homo`).
   Un GNN mieux réglé (relation unique, sampling de voisins) réduirait l'écart
   sans le combler probablement — à mentionner pour l'honnêteté.
3. **Ne pas comparer aux chiffres PC-GNN publiés** : protocole/splits différents
   (ici 60/20/20 stratifié). Rester sur la comparaison **interne** GraphSAGE vs
   XGBoost, même protocole. Ne PAS conclure « on bat PC-GNN ».

### Ce que ça dit vraiment

Pas « les graphes sont inutiles » mais « **l'agrégation naïve échoue sous
camouflage** ». Le pari de la thèse se joue sur les **données fiscales**, où les
relations (dirigeant/adresse/comptable communs) sont probablement plus
**homophiles** que YelpChi → le GNN pourrait y gagner. YelpChi = banc d'essai
qui montre les limites du GNN naïf, pas le verdict final.

### Pistes avant les données réelles

- **Multi-seed** (moyenne ± std) pour la crédibilité.
- **XGBoost + features de graphe** (degré, ratio de fraude des voisins) → tester
  si l'hybride bat les deux modèles purs.

### Limites

- Un seul dataset public, GraphSAGE vanilla, hyperparamètres par défaut, 1 seed.
- Le résultat ne se transpose pas mécaniquement aux données fiscales réelles
  (structure de graphe différente, biais de sélection des labels).
- Reproductible : `notebooks/02_yelpchi_pipeline.ipynb` (≈15 min CPU).

# Résultats expérimentaux

> Protocole : splits stratifiés 60/20/20, **moyenne ± écart-type sur 5 seeds**
> (0–4). Déséquilibre géré par class weights (GNN) et `scale_pos_weight` (XGBoost).
> Features normalisées (z-score, stats train uniquement) avant tous les modèles.
> Reproductible : `python -m src.experiments.benchmark 5`.

## Étape 1 — Cora (validation mécanique GNN)

GraphSAGE 2 couches, test accuracy = **0.798** (>0.75). Valide seulement la
pipeline PyG, ne répond pas à la question de recherche.

## Étape 2 — YelpChi : baseline (GraphSAGE vs XGBoost vs XGBoost+graph)

Dataset : 45 954 nœuds, 32 features, taux de fraude ≈ 14,5 %, relation `homo`
(7,7 M arêtes).

| Modèle | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---|---|---|---|---|
| GraphSAGE (homo) | 0.894 ± 0.002 | 0.676 ± 0.006 | 0.728 ± 0.003 | 0.812 ± 0.005 | 0.611 ± 0.008 |
| XGBoost | 0.946 ± 0.003 | 0.819 ± 0.010 | 0.831 ± 0.008 | 0.870 ± 0.008 | 0.742 ± 0.015 |
| **XGBoost+graph** | **0.951 ± 0.002** | **0.830 ± 0.009** | **0.838 ± 0.005** | **0.875 ± 0.007** | **0.751 ± 0.010** |

### Lectures

**1. La normalisation est cruciale pour le GNN.** Sans normalisation, GraphSAGE
plafonnait à 0.80 AUC. Avec (features brutes YelpChi à échelles très variées),
il monte à **0.894**. XGBoost, scale-invariant, n'en profite pas. → un GNN doit
recevoir des features préparées ; sinon on sous-estime le Graph ML.

**2. XGBoost > GraphSAGE sur `homo`.** Écart resserré mais réel (AUC 0.946 vs
0.894 ; AUC-PR 0.819 vs 0.676). Cause : le graphe `homo` est **quasi
non-homophile** (voir §homophilie) — les fraudeurs sont camouflés, et
l'agrégation uniforme de GraphSAGE lisse le signal.

**3. Les features de graphe aident le tabulaire (lift réel, modeste).**
Comparaison **appariée** XGBoost+graph vs XGBoost (même split par seed) :

| Métrique | Δ moyen | écart-type | seeds positifs |
|---|---|---|---|
| AUC-PR | **+0.0106** | 0.0045 | **5/5** |
| AUC-ROC | **+0.0055** | 0.0018 | **5/5** |

Positif sur les 5 seeds, std ≪ moyenne → effet consistant, pas du bruit. Mais
c'est du **ML tabulaire enrichi**, pas du Graph ML : info de graphe réduite à
2 features (degré, ratio de fraude des voisins-train, calculé sans fuite). Sert
de borne « combien le graphe apporte sans message-passing ».

## Homophilie par relation (le levier Graph ML)

YelpChi est **multi-relationnel** : 3 relations, qu'on a écrasées en `homo`.
Homophilie = % d'arêtes reliant 2 nœuds de même label. Baseline aléatoire
(fraude 14,5 %) = 0,85² + 0,15² ≈ **0,752**.

| relation | arêtes | degré moy. | homophilie |
|---|---|---|---|
| homo | 7 693 958 | 167.4 | 0.773 |
| net_rtr | 1 147 232 | 25.0 | 0.759 |
| net_rsr | 6 805 486 | 148.1 | 0.772 |
| **net_rur** | 98 630 | 2.1 | **0.996** |

→ `homo`/`rtr`/`rsr` ≈ aléatoire = **non homophiles** (camouflage) : c'est le
mauvais graphe pour un GNN. `net_rur` est **quasi parfaitement homophile** (mais
sparse). **Hypothèse centrale du volet Graph ML** : la faiblesse de GraphSAGE
vient du graphe fourni, pas du GNN. Sur `net_rur`, le message-passing devrait
enfin payer.

## Expériences Graph ML (en cours)

Volet GNN dédié, pour rester sur le sujet (Graph ML, pas du tabulaire) :

1. **Sweep de relations** — GraphSAGE par relation (test de l'hypothèse ci-dessus).
2. **GAT** — attention par arête, downweighte les voisins camouflés (`src/models/gat.py`).
3. **GNN multi-relationnel** — combine les 3 relations avec poids appris.

Résultats ajoutés ici à mesure.

## Réponse (provisoire) à la question de recherche

Sur la relation `homo`, un **GNN générique (GraphSAGE) n'améliore pas** la
détection vs XGBoost. Mais l'analyse d'homophilie montre que `homo` est un
mauvais graphe ; le volet Graph ML teste si le bon graphe / la bonne archi
inverse la conclusion. Verdict final = données fiscales (relations attendues
homophiles : dirigeant/adresse/comptable communs).

## Réserves à assumer dans le mémoire

1. **YelpChi reste un banc d'essai** ; les conclusions ne se transposent pas
   mécaniquement au fiscal (structure différente, biais de sélection des labels).
2. **Pas de tuning exhaustif** des GNN (hidden, couches, têtes) — comparaisons à
   budget égal, pas une recherche d'archi optimale.
3. **Ne pas comparer aux chiffres PC-GNN publiés** (protocole/splits différents) ;
   rester sur les comparaisons internes, même protocole.

# Avancement du projet — PFE Détection de fraude par Graph ML

> Journal de bord concis. Date : 2026-06-11. Branche git : `impl-pipeline` (PR #1).
> Cahier des charges initial : `FEUILLE_DE_ROUTE.md` (fourni séparément).

## 1. Objectif du projet

Outil de triage qui attribue un **score de risque de fraude** à des contribuables
en exploitant leurs **relations** (graphe) en plus de leurs features individuelles.
**Question de recherche** : exploiter les relations (GNN / GraphSAGE) améliore-t-il
la détection vs une approche isolée (XGBoost) ?

## 2. Méthode de travail suivie

1. **Spec** validée → `docs/superpowers/specs/2026-06-11-...-design.md`
2. **Plan** d'implémentation TDD → `docs/superpowers/plans/2026-06-11-...md`
3. **Implémentation** par tâches, en TDD (test d'abord), commits fréquents.
4. Décisions clés actées : modules `.py` = source de vérité, notebooks Colab =
   wrappers fins ; GraphSAGE homogène ; XGBoost baseline ; graphe en mémoire (PyG),
   pas de base de données graphe.

## 3. Étapes effectuées

### Étape 1 — Cora (validation pipeline GNN) ✅
GraphSAGE 2 couches sur Cora, **test accuracy 0.798** (>0.75). Valide la mécanique
PyG. Ne répond pas à la question de recherche (juste un « hello world » GNN).

### Étape 2 — YelpChi (banc d'essai fraude) ✅ (en approfondissement)
Dataset public de faux avis Yelp : 45 954 nœuds, 32 features, ~14,5 % de fraude.

**Galère DGL résolue à la racine.** Le plan prévoyait DGL pour charger YelpChi,
mais DGL casse sur Colab (`graphbolt` exige `torchdata.datapipes`, supprimé des
versions récentes). **Solution** : on a viré DGL — `src/data/yelpchi.py` télécharge
le `.mat` (data.dgl.ai) et le parse avec scipy. Marche en local **et** Colab,
sans dépendance fragile.

**Pipeline construite** (tout testé, 18 tests pytest) :
- `src/models/graphsage.py` — GraphSAGE 2 couches
- `src/models/gat.py` — GAT (attention) [volet Graph ML]
- `src/train/train_gnn.py` — boucle d'entraînement + early-stop + class weights (déséquilibre)
- `src/train/baseline_xgb.py` — XGBoost (baseline sans graphe)
- `src/eval/metrics.py` — 5 KPI : AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k
- `src/data/graph_features.py` — features de graphe **sans fuite de labels** (testé)
- `src/data/transforms.py` — normalisation z-score (stats train)
- `src/experiments/benchmark.py` — banc multi-seed + delta apparié
- `src/experiments/relation_sweep.py` — GNN par relation

**Résultats baseline (moyenne ± std sur 5 seeds)** — voir `docs/RESULTATS.md` :

| Modèle | AUC-ROC | AUC-PR |
|---|---|---|
| GraphSAGE (homo) | 0.894 ± 0.002 | 0.676 ± 0.006 |
| XGBoost | 0.946 ± 0.003 | 0.819 ± 0.010 |
| XGBoost+graph | 0.951 ± 0.002 | 0.830 ± 0.009 |

**Découvertes importantes :**
- **Normalisation cruciale** : sans elle GraphSAGE plafonne à 0.80 AUC ; avec, il
  monte à 0.894. Les features GNN doivent être préparées.
- **XGBoost > GraphSAGE sur `homo`** (écart réel mais resserré).
- **Features de graphe aident le tabulaire** : XGBoost+graph > XGBoost, +0.011
  AUC-PR, **positif sur 5/5 seeds** (comparaison appariée).
- **Analyse d'homophilie par relation** (le levier Graph ML) : `homo`/`net_rtr`/
  `net_rsr` ≈ aléatoire (~0.75) = **non homophiles** (fraude camouflée) ; mais
  **`net_rur` = 0.996** = ultra-homophile (mais sparse). Hypothèse : la faiblesse
  du GNN vient du **graphe fourni**, pas du GNN.

### Pivot Graph ML (en cours)
Recadrage : le but est le **Graph ML (GNN)**, pas le ML tabulaire. On construit un
volet GNN dédié pour rester sur le sujet :
1. **Sweep de relations** — GraphSAGE par relation (`net_rur` vs `homo`). **EN COURS.**
2. **GAT** — attention pour ignorer les voisins camouflés (codé, testé).
3. **GNN multi-relationnel** — combiner les 3 relations (à venir).

### Étape 3 — Graphe fiscal (stub)
`src/data/fiscal_graph.py` = signature + TODOs, `NotImplementedError`. Attend les
vraies données fiscales (relations attendues homophiles : dirigeant/adresse/
comptable communs → terrain où le GNN devrait gagner).

## 4. État actuel
- 18 tests pytest passent. Code poussé sur `impl-pipeline`, PR #1 ouverte.
- Job en cours : sweep GraphSAGE `net_rur`/`net_rtr` (5 seeds).
- Reste : finir sweep → GAT → multi-relationnel → (puis données fiscales).

## 5. Limites assumées
- YelpChi = banc d'essai, ne se transpose pas mécaniquement au fiscal.
- Pas de tuning exhaustif des GNN (comparaisons à budget égal).
- Ne pas comparer aux chiffres PC-GNN publiés (protocoles différents).
- Biais de sélection des labels (non-enquêtés ≠ non-fraudeurs) à documenter.

---

## Prompt à donner à une autre IA (explication détaillée du projet)

> Copie-colle le bloc ci-dessous à l'autre IA. Elle aura accès aux fichiers du
> projet, à ce `AVANCEMENT.md`, et au cahier des charges `FEUILLE_DE_ROUTE.md`.

```
Tu es un assistant pédagogue. J'ai un projet de fin d'études (PFE) sur la
détection de fraude fiscale par Graph Machine Learning. Tu as accès à tous les
fichiers du projet, au journal d'avancement `AVANCEMENT.md`, au cahier des
charges `FEUILLE_DE_ROUTE.md`, et aux résultats `docs/RESULTATS.md`.

Explique-moi le projet EN DÉTAIL et de façon PÉDAGOGIQUE, comme si je devais le
présenter à un jury sans tout maîtriser. Structure ta réponse ainsi :

1. LE PROBLÈME : c'est quoi la détection de fraude par graphe, et pourquoi un
   graphe plutôt qu'un modèle classique. Explique les concepts (nœud, arête,
   GNN, message-passing, GraphSAGE, GAT, homophilie/hétérophilie, camouflage)
   avec des analogies simples.

2. CE QUI A ÉTÉ FAIT : parcours `AVANCEMENT.md` et le code dans `src/`. Pour
   chaque module, explique son rôle en une phrase claire. Explique pourquoi on a
   abandonné DGL et chargé le .mat à la place.

3. LES RÉSULTATS : lis `docs/RESULTATS.md`. Explique le tableau (GraphSAGE vs
   XGBoost vs XGBoost+graph), ce que chaque métrique (AUC-ROC, AUC-PR, F1-macro,
   GMean, Recall@k) mesure et pourquoi on n'utilise PAS l'accuracy. Explique
   pourquoi XGBoost bat GraphSAGE ici, et le rôle de l'homophilie des relations.
   Explique ce qu'est une comparaison appariée (paired delta).

4. LA STRATÉGIE GRAPH ML : pourquoi le sweep de relations, GAT et le GNN
   multi-relationnel sont les bonnes prochaines étapes pour rester sur du vrai
   Graph ML, et pourquoi le vrai test sera sur les données fiscales réelles.

5. CE QUI RESTE À FAIRE et les limites/biais à mentionner dans le mémoire.

Vérifie tes affirmations en lisant les fichiers réels (ne devine pas). Si un
chiffre que tu cites vient d'un fichier, dis lequel. Pose-moi des questions si
un point du code n'est pas clair. Sois précis sur les termes techniques mais
explique-les simplement.
```

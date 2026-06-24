# Feuille de route — PFE : Détection de fraude fiscale par Graph ML

> Document de pilotage du projet. À lire et à utiliser comme cahier des charges / suivi.
> Ce n'est PAS du code exécutable : le code vit dans un notebook Google Colab (`.ipynb`).

---

## 1. Le projet en une phrase

Construire un **outil de triage** qui attribue un **score de risque de fraude** à des contribuables
(entreprises + particuliers), en exploitant non seulement leurs caractéristiques individuelles
mais aussi leurs **relations** (graphe), afin de **prioriser** les dossiers à examiner.
La sortie n'est PAS une décision : c'est une aide à la décision, revue ensuite par un humain.

- **Tâche** : classification binaire (fraudeur / non-fraudeur) → score entre 0 et 1.
- **Périmètre exclu** : TVA, blanchiment, banque/cartes bancaires.
- **Positionnement** : détection en amont, pour la priorisation.

---

## 2. Décisions déjà actées

| Sujet | Décision | Raison |
|---|---|---|
| Modèle principal | **GraphSAGE** (via PyTorch Geometric) | Simple, robuste, standard |
| Baseline de comparaison | **XGBoost** (+ quelques features de graphe) | Point de comparaison "sans graphe" |
| Papier de référence | **PC-GNN** (Liu et al., WWW 2021) | Méthodologie + protocole d'évaluation |
| À citer, pas à coder | CARE-GNN, PC-GNN | État de l'art, hors périmètre PFE |
| Environnement | **Google Colab** (GPU gratuit) | Zéro config, suffisant pour ces graphes |
| Base de données graphe (Arango/Neo4j) | **NON** | Inutile : graphe construit en mémoire par PyG |

---

## 3. La question de recherche

> Est-ce qu'exploiter les **relations** entre contribuables (approche graphe, GraphSAGE)
> améliore la détection de fraude par rapport à une approche qui regarde chaque
> contribuable **isolément** (XGBoost) ?

C'est l'axe central du mémoire. Tout le travail expérimental sert à y répondre.

---

## 4. Les données

### 4.1 Données réelles (à demander, en attente)
Trois tables :
- **Contribuables (nœuds)** : id anonymisé, type (entreprise/particulier), caractéristiques
  (CA/revenu, secteur, ancienneté, effectif, impôt déclaré, ratios), comportement déclaratif
  (retards, rectifications, écarts annuels), **label** (fraudeur/non-fraudeur, pour les enquêtés seulement).
- **Relations (arêtes)** : id_source, id_cible, type_relation
  (client-fournisseur, partage de dirigeant, partage d'adresse, partage de comptable, participation).
- **Métadonnées** : volume, format (CSV/Excel), anonymisation.

> ⚠️ **Biais de sélection** : seuls les contribuables déjà enquêtés ont un label.
> Les autres sont "inconnus", PAS "non-fraudeurs". À assumer comme limite dans le mémoire.

### 4.2 Données publiques (pour démarrer tout de suite)
- **Cora** : "hello world" du graph ML (graphe de citations). Pour comprendre PyG en 1 aprèm.
- **YelpChi** / **Amazon** : datasets de fraude (faux avis), déséquilibrés, structure graphe.
  Chargeables via `dgl.data.FraudDataset`. Servent de **banc d'essai** pour valider la pipeline.

> Valorisable dans le mémoire : "implémentation validée sur datasets de référence avant
> application aux données fiscales".

---

## 5. Plan de travail (étapes)

### Étape 0 — Environnement (30 min)
- [ ] Créer un compte / ouvrir Google Colab
- [ ] Activer le GPU (Exécution → Modifier le type d'exécution → GPU)
- [ ] Installer les librairies dans une cellule (voir §6)

### Étape 1 — Prise en main PyG sur Cora (½ journée)
- [ ] Charger Cora
- [ ] Comprendre l'objet `Data` (x = features, edge_index = arêtes, y = labels, masks)
- [ ] Coder un GraphSAGE à 2 couches
- [ ] Entraîner + évaluer (accuracy ici, c'est juste pour comprendre la mécanique)

### Étape 2 — Pipeline de fraude sur YelpChi (2-3 jours)
- [ ] Charger YelpChi (FraudDataset)
- [ ] GraphSAGE + entraînement avec gestion du déséquilibre (pondération de la loss)
- [ ] **Baseline XGBoost** sur les mêmes features (sans le graphe)
- [ ] Évaluation avec les VRAIS KPI : **AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k**
- [ ] Comparer GraphSAGE vs XGBoost → première réponse à la question de recherche

### Étape 3 — Modélisation du graphe fiscal (sur papier, en parallèle)
- [ ] Lister précisément nœuds, types, features, relations à créer depuis les vraies données
- [ ] Préparer le code de construction du graphe (tables → objet PyG/HeteroData)

### Étape 4 — Application aux données réelles (quand elles arrivent)
- [ ] Adapter le chargement (remplacer la source)
- [ ] Reconstruire le graphe à partir des 3 tables
- [ ] Relancer toute la pipeline (le reste du code ne change pas)
- [ ] Analyser les résultats, les limites, le biais de sélection

### Étape 5 — Rédaction (en continu)
- [ ] Plan du mémoire
- [ ] Revue de littérature (PC-GNN, CARE-GNN, GraphSAGE, survey GNN fraude)
- [ ] Méthodologie
- [ ] Expériences + résultats (gabarit = tableaux de PC-GNN)
- [ ] Limites + conclusion

---

## 6. Stack technique

| Outil | Rôle |
|---|---|
| Google Colab | Environnement d'exécution (GPU gratuit) |
| PyTorch | Base deep learning |
| PyTorch Geometric (PyG) | Couche graphe : objet graphe + `SAGEConv` |
| DGL | Uniquement pour charger YelpChi/Amazon (`FraudDataset`) |
| XGBoost | Baseline sans graphe |
| pandas / scikit-learn | Préparation données + métriques |

**Installation Colab (une cellule) :**
```
!pip install torch_geometric
!pip install dgl
!pip install xgboost scikit-learn pandas
```

**PAS besoin de :** ArangoDB, Neo4j, ni aucune base de données graphe.
Le graphe est construit en mémoire à partir des tables.

---

## 7. Pièges à éviter (rappels)

1. **Ne jamais évaluer en accuracy** → le déséquilibre la rend trompeuse (98% en prédisant
   "non-fraudeur" partout). Utiliser AUC, F1-macro, GMean, Recall@k.
2. **Gérer le déséquilibre** → pondération de la loss (`pos_weight`) ou échantillonnage.
3. **Ne pas reproduire PC-GNN/CARE-GNN** → trop complexe, hors périmètre. Les citer seulement.
4. **Ne pas faire tourner le vieux code GitHub de PC-GNN** → conflits de versions (2021).
   Coder GraphSAGE soi-même avec PyG (simple).
5. **Labels partiels** → les non-enquêtés ne sont pas des négatifs. Limite à documenter.
6. **2-3 couches GNN max** → au-delà, over-smoothing (les nœuds se ressemblent tous).

---

## 8. Références

- **PC-GNN** (papier) : https://ponderly.github.io/pub/PCGNN_WWW2021.pdf
- **PC-GNN** (code, référence seulement) : https://github.com/PonderLY/PC-GNN
- **Étude comparative simple** : https://arxiv.org/abs/2105.14568
- **PyTorch Geometric** (doc + tutos) : https://pytorch-geometric.readthedocs.io

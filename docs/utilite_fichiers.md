# Utilité de chaque fichier du projet

Ce document explique la structure du projet `pfe-fraude-graph-ml` et le rôle de chaque fichier important.

## Vue générale

```text
pfe-fraude-graph-ml/
├── README.md
├── FEUILLE_DE_ROUTE.md
├── AVANCEMENT.md
├── requirements.txt
├── pytest.ini
├── src/
├── tests/
├── notebooks/
└── docs/
```

Idée générale :

- `src/` contient le vrai code réutilisable.
- `tests/` contient les tests automatiques.
- `notebooks/` contient les notebooks Colab.
- `docs/` contient la documentation, les résultats et les rapports.
- Les fichiers `.md` à la racine expliquent le projet, son objectif et son avancement.

---

# 1. Fichiers à la racine

## `README.md`

C’est la page d’accueil du projet.

Il sert à expliquer rapidement :

- le but du projet ;
- la structure générale ;
- comment installer les dépendances ;
- comment lancer les tests ;
- la question de recherche.

Rôle simple :

> Donner une première vue rapide du projet à quelqu’un qui le découvre.

---

## `FEUILLE_DE_ROUTE.md`

C’est le cahier des charges initial du projet.

Il explique :

- le sujet du PFE ;
- l’objectif : attribuer un score de risque de fraude ;
- le fait que le modèle est une aide à la décision, pas une décision automatique ;
- les données attendues ;
- les étapes de travail ;
- les pièges à éviter ;
- les métriques à utiliser ;
- les références scientifiques comme PC-GNN, CARE-GNN et GraphSAGE.

Rôle simple :

> Définir ce qu’on veut construire et pourquoi.

---

## `AVANCEMENT.md`

C’est le journal de bord du projet.

Il explique :

- ce qui a déjà été fait ;
- pourquoi certains choix ont été pris ;
- les résultats obtenus ;
- les problèmes rencontrés ;
- les prochaines étapes.

Par exemple, il explique que DGL a été abandonné parce qu’il posait problème, et que YelpChi est maintenant chargé directement depuis le fichier `.mat`.

Rôle simple :

> Montrer où en est le projet actuellement.

---

## `requirements.txt`

C’est la liste des bibliothèques Python nécessaires au projet.

Exemples de dépendances :

- `torch` ;
- `torch_geometric` ;
- `xgboost` ;
- `scikit-learn` ;
- `pandas` ;
- `numpy` ;
- `pytest`.

Il sert à installer l’environnement avec :

```bash
pip install -r requirements.txt
```

Rôle simple :

> Installer toutes les librairies nécessaires pour faire tourner le projet.

---

## `pytest.ini`

C’est le fichier de configuration des tests.

Il indique à `pytest` :

- où chercher les tests ;
- comment les afficher.

Dans ce projet, les tests sont dans le dossier `tests/`.

Rôle simple :

> Configurer l’exécution des tests automatiques.

---

## `.gitignore`

Ce fichier indique à Git quels fichiers ne doivent pas être suivis.

Exemples :

- données téléchargées ;
- caches Python ;
- fichiers temporaires ;
- gros fichiers générés.

Rôle simple :

> Éviter d’envoyer dans Git des fichiers inutiles ou trop lourds.

---

# 2. Dossier `src/`

Le dossier `src/` contient le cœur du code.

```text
src/
├── config.py
├── data/
├── models/
├── train/
├── eval/
└── experiments/
```

Rôle simple :

> Contenir tout le code principal, propre, réutilisable et testable.

---

## `src/__init__.py`

Ce fichier peut être vide.

Il sert à indiquer à Python que `src` est un package importable.

Grâce à lui, on peut faire des imports comme :

```python
from src.config import TrainConfig
```

Rôle simple :

> Permettre à Python d’importer les fichiers du dossier `src`.

---

## `src/config.py`

Ce fichier contient la configuration globale.

Il définit :

- le seed aléatoire pour rendre les expériences reproductibles ;
- le device : CPU ou GPU ;
- la classe `TrainConfig` avec les hyperparamètres.

Exemples d’hyperparamètres :

- nombre d’époques ;
- learning rate ;
- dropout ;
- hidden dimension ;
- patience pour l’early stopping.

Rôle simple :

> Centraliser les paramètres d’entraînement.

---

# 3. Dossier `src/data/`

Ce dossier sert à charger et préparer les données.

```text
src/data/
├── cora.py
├── yelpchi.py
├── fiscal_graph.py
├── graph_features.py
├── transforms.py
└── __init__.py
```

---

## `src/data/__init__.py`

Fichier d’initialisation du package `data`.

Il permet d’importer les modules de données avec :

```python
from src.data.yelpchi import load_yelpchi
```

Rôle simple :

> Rendre le dossier `data` importable.

---

## `src/data/cora.py`

Ce fichier charge le dataset Cora.

Cora est un dataset classique de Graph ML, utilisé pour vérifier que GraphSAGE fonctionne.

Dans ce projet, Cora sert seulement à valider la mécanique :

- charger un graphe ;
- entraîner GraphSAGE ;
- vérifier que le modèle apprend.

Ce n’est pas le vrai dataset de fraude.

Rôle simple :

> Tester rapidement que la pipeline GNN fonctionne.

---

## `src/data/yelpchi.py`

Ce fichier charge YelpChi, un dataset public de fraude sur les faux avis Yelp.

Il fait plusieurs choses :

- télécharge `FraudYelp.zip` ;
- extrait `YelpChi.mat` ;
- lit les features ;
- lit les labels ;
- lit les relations du graphe :
  - `homo` ;
  - `net_rur` ;
  - `net_rtr` ;
  - `net_rsr` ;
- transforme les matrices d’adjacence en `edge_index` PyTorch Geometric ;
- crée les masques train/validation/test.

Important : dans le plan initial, YelpChi devait être chargé avec DGL. Dans le code actuel, DGL a été abandonné. Le fichier `.mat` est chargé directement avec `scipy`.

Rôle simple :

> Charger le dataset de fraude public YelpChi pour tester la détection de fraude par graphe.

---

## `src/data/fiscal_graph.py`

C’est un stub pour le futur graphe fiscal réel.

Un stub veut dire :

> Le fichier existe, la structure est prévue, mais le vrai corps n’est pas encore implémenté.

Ce fichier prépare l’idée du vrai graphe fiscal avec :

- entreprises ;
- particuliers ;
- relations client-fournisseur ;
- partage de dirigeant ;
- partage d’adresse ;
- partage de comptable ;
- participation.

Il utilise `HeteroData`, donc il prépare un graphe hétérogène.

Actuellement, la fonction principale lève volontairement :

```python
NotImplementedError
```

Ce n’est pas une erreur : c’est volontaire, car les vraies données fiscales ne sont pas encore disponibles.

Rôle simple :

> Préparer la future construction du vrai graphe fiscal hétérogène.

---

## `src/data/graph_features.py`

Ce fichier calcule des features de graphe pour enrichir XGBoost.

Il crée deux features par nœud :

1. le degré du nœud ;
2. le ratio de voisins frauduleux dans le train.

Mais il le fait sans fuite de labels.

Cela veut dire qu’il n’utilise pas les labels validation/test pour calculer ces features.

Pourquoi c’est important ?

Sinon, le modèle tricherait en utilisant des informations qu’il n’est pas censé connaître.

Rôle simple :

> Transformer une partie de l’information du graphe en features tabulaires pour XGBoost+graph.

---

## `src/data/transforms.py`

Ce fichier contient les transformations appliquées aux features.

Actuellement, il fait surtout la normalisation z-score :

```text
x_normalisé = (x - moyenne_train) / écart_type_train
```

La moyenne et l’écart-type sont calculés uniquement sur le train.

Pourquoi ?

Pour éviter une fuite de données depuis validation/test.

Rôle simple :

> Normaliser les features proprement avant l’entraînement.

---

# 4. Dossier `src/models/`

Ce dossier contient les modèles de Graph ML.

```text
src/models/
├── graphsage.py
├── gat.py
├── multi_rel_gnn.py
└── __init__.py
```

---

## `src/models/__init__.py`

Fichier d’initialisation du package `models`.

Rôle simple :

> Rendre le dossier `models` importable.

---

## `src/models/graphsage.py`

C’est le modèle principal du projet.

Il implémente GraphSAGE avec deux couches `SAGEConv`.

GraphSAGE sert à faire du message-passing :

- chaque nœud regarde ses propres features ;
- il reçoit de l’information de ses voisins ;
- il agrège ces informations ;
- il produit une prédiction.

Dans YelpChi, il prédit si un avis est frauduleux.

Dans le futur fiscal, il pourra prédire si un contribuable est à risque.

Rôle simple :

> Modèle GNN principal qui exploite les relations du graphe.

---

## `src/models/gat.py`

Ce fichier implémente GAT, Graph Attention Network.

Différence avec GraphSAGE :

- GraphSAGE agrège les voisins de manière assez uniforme ;
- GAT apprend quels voisins sont plus importants.

C’est utile si certains voisins sont très informatifs et d’autres sont bruités.

Exemple :

> Si un contribuable est connecté à beaucoup de nœuds, GAT peut apprendre à donner plus de poids aux relations vraiment suspectes.

Rôle simple :

> Tester une architecture GNN avec attention pour mieux gérer les voisins importants ou bruités.

---

## `src/models/multi_rel_gnn.py`

Ce fichier implémente un modèle GNN multi-relationnel.

Pourquoi ?

YelpChi contient plusieurs relations :

- `net_rur` ;
- `net_rtr` ;
- `net_rsr`.

Chaque relation n’a pas la même qualité :

- `net_rur` est très homophile mais peu dense ;
- les autres relations sont plus denses mais plus bruitées.

Le modèle multi-relationnel utilise une convolution par relation, puis combine les résultats.

Rôle simple :

> Combiner plusieurs types de relations au lieu de tout mélanger dans un seul graphe.

---

# 5. Dossier `src/train/`

Ce dossier contient le code d’entraînement.

```text
src/train/
├── train_gnn.py
├── baseline_xgb.py
└── __init__.py
```

---

## `src/train/__init__.py`

Fichier d’initialisation du package `train`.

Rôle simple :

> Rendre le dossier `train` importable.

---

## `src/train/train_gnn.py`

Ce fichier contient la boucle d’entraînement des GNN.

Il gère :

- l’entraînement du modèle ;
- la loss ;
- les poids de classes pour le déséquilibre ;
- l’early stopping ;
- la prédiction des scores.

Fonctions importantes :

### `train_gnn(...)`

Entraîne un modèle GNN.

Il prend :

- le modèle ;
- les données ;
- la configuration ;
- éventuellement les poids de classes.

### `predict_scores(...)`

Retourne un score de probabilité pour la classe positive, donc la fraude.

### `class_weights_from_labels(...)`

Calcule les poids de classes pour gérer le déséquilibre.

Pourquoi c’est nécessaire ?

Parce qu’il y a beaucoup moins de fraudeurs que de non-fraudeurs.

Rôle simple :

> Entraîner GraphSAGE, GAT ou MultiRelGNN proprement.

---

## `src/train/baseline_xgb.py`

Ce fichier contient la baseline XGBoost.

XGBoost est un modèle tabulaire puissant.

Il ne voit pas le graphe directement, seulement les features des nœuds.

Fonctions :

### `train_xgb(...)`

Entraîne XGBoost.

### `predict_xgb(...)`

Retourne les probabilités de fraude.

Rôle simple :

> Fournir un modèle de comparaison sans message-passing graphe.

---

# 6. Dossier `src/eval/`

Ce dossier contient l’évaluation.

```text
src/eval/
├── metrics.py
└── __init__.py
```

---

## `src/eval/__init__.py`

Fichier d’initialisation du package `eval`.

Rôle simple :

> Rendre le dossier `eval` importable.

---

## `src/eval/metrics.py`

Ce fichier calcule les métriques importantes.

Il contient :

### `compute_metrics(...)`

Calcule :

- AUC-ROC ;
- AUC-PR ;
- F1-macro ;
- GMean ;
- Recall@k.

Ces métriques sont adaptées aux datasets déséquilibrés.

On n’utilise pas l’accuracy comme métrique principale, parce qu’elle serait trompeuse.

### `print_comparison(...)`

Affiche une comparaison entre plusieurs modèles.

Rôle simple :

> Évaluer correctement les modèles de fraude.

---

# 7. Dossier `src/experiments/`

Ce dossier contient les scripts d’expériences.

```text
src/experiments/
├── benchmark.py
├── relation_sweep.py
├── multi_rel.py
└── __init__.py
```

---

## `src/experiments/__init__.py`

Fichier d’initialisation du package `experiments`.

Rôle simple :

> Rendre le dossier `experiments` importable.

---

## `src/experiments/benchmark.py`

C’est le script principal de benchmark.

Il compare :

1. GraphSAGE ;
2. XGBoost ;
3. XGBoost+graph.

Il fait les expériences sur plusieurs seeds pour éviter de dépendre d’un seul split chanceux.

Il calcule ensuite moyenne ± écart-type.

Il contient aussi `paired_delta`, pour comparer deux modèles sur les mêmes splits.

Rôle simple :

> Comparer proprement GraphSAGE, XGBoost et XGBoost+graph.

---

## `src/experiments/relation_sweep.py`

Ce script teste GraphSAGE relation par relation.

Par exemple :

- GraphSAGE sur `net_rur` ;
- GraphSAGE sur `net_rtr` ;
- GraphSAGE sur `net_rsr` ;
- GraphSAGE sur `homo`.

Pourquoi ?

Pour savoir quelle relation est la plus utile pour le Graph ML.

Dans les résultats, `net_rur` est la relation la plus homophile.

Rôle simple :

> Identifier quelles relations du graphe sont les plus utiles pour détecter la fraude.

---

## `src/experiments/multi_rel.py`

Ce script lance l’expérience avec le modèle multi-relationnel.

Il combine :

- `net_rur` ;
- `net_rtr` ;
- `net_rsr`.

L’idée est :

- `net_rur` est propre mais peu dense ;
- les autres relations sont plus denses ;
- donc on essaie de combiner qualité + couverture.

Rôle simple :

> Tester un GNN qui utilise plusieurs relations en même temps.

---

# 8. Dossier `tests/`

Ce dossier contient les tests automatiques.

```text
tests/
├── test_metrics.py
├── test_graphsage.py
├── test_gat.py
├── test_multi_rel_gnn.py
├── test_train_gnn.py
├── test_baseline_xgb.py
├── test_cora.py
├── test_yelpchi_import.py
├── test_fiscal_graph_stub.py
├── test_graph_features.py
└── __init__.py
```

Les tests servent à vérifier que chaque partie du code fonctionne.

---

## `tests/__init__.py`

Fichier d’initialisation du package `tests`.

Rôle simple :

> Rendre le dossier `tests` importable.

---

## `tests/test_metrics.py`

Teste les métriques.

Il vérifie par exemple que :

- AUC = 1 si la séparation est parfaite ;
- Recall@k est bien calculé ;
- le `k` par défaut correspond au nombre de positifs.

Rôle simple :

> Vérifier que l’évaluation est correcte.

---

## `tests/test_graphsage.py`

Teste le modèle GraphSAGE.

Il vérifie :

- que la sortie a la bonne taille ;
- qu’il y a bien deux couches `SAGEConv`.

Rôle simple :

> Vérifier que GraphSAGE est correctement construit.

---

## `tests/test_gat.py`

Teste le modèle GAT.

Il vérifie :

- que la sortie a la bonne forme ;
- que le modèle utilise bien des couches `GATConv` ;
- que les paramètres sont cohérents.

Rôle simple :

> Vérifier que le modèle avec attention fonctionne.

---

## `tests/test_multi_rel_gnn.py`

Teste le modèle multi-relationnel.

Il vérifie :

- que le modèle accepte plusieurs relations ;
- que la sortie a la bonne forme.

Rôle simple :\n> Vérifier que le GNN multi-relationnel fonctionne.

---

## `tests/test_train_gnn.py`

Teste l’entraînement GNN.

Il vérifie notamment :

- que GraphSAGE apprend sur Cora ;
- que `predict_scores` retourne bien des probabilités entre 0 et 1.

Rôle simple :

> Vérifier que la boucle d’entraînement GNN est correcte.

---

## `tests/test_baseline_xgb.py`

Teste XGBoost.

Il crée un petit dataset artificiel facilement séparable, entraîne XGBoost, puis vérifie qu’il apprend bien.

Rôle simple :

> Vérifier que la baseline XGBoost fonctionne.

---

## `tests/test_cora.py`

Teste le chargement de Cora.

Il vérifie :

- nombre de nœuds ;
- dimensions des features ;
- labels ;
- masques train/validation/test.

Rôle simple :

> Vérifier que le dataset Cora est chargé correctement.

---

## `tests/test_yelpchi_import.py`

Teste que le module YelpChi peut être importé.

C’est important car on veut éviter que le projet casse juste parce qu’une dépendance externe comme DGL manque.

Aujourd’hui, comme YelpChi est chargé sans DGL, ce test valide que le module reste propre.

Rôle simple :

> Vérifier que le loader YelpChi est importable.

---

## `tests/test_fiscal_graph_stub.py`

Teste le stub fiscal.

Il vérifie que `build_fiscal_graph` lève bien `NotImplementedError`.

Pourquoi tester une fonction non implémentée ?

Parce que c’est le comportement attendu tant que les vraies données fiscales ne sont pas là.

Rôle simple :

> Vérifier que le futur graphe fiscal est prévu mais pas faussement implémenté.

---

## `tests/test_graph_features.py`

Teste les features de graphe.

Il vérifie notamment qu’il n’y a pas de fuite de labels validation/test.

Rôle simple :

> Vérifier que XGBoost+graph utilise des features de graphe calculées proprement.

---

# 9. Dossier `notebooks/`

```text
notebooks/
├── 01_cora_demo.ipynb
└── 02_yelpchi_pipeline.ipynb
```

Les notebooks servent à exécuter les expériences dans un environnement interactif, par exemple Google Colab.

---

## `notebooks/01_cora_demo.ipynb`

Notebook pour Cora.

Il sert à :

- charger Cora ;
- entraîner GraphSAGE ;
- afficher l’accuracy.

Rôle simple :

> Démonstration simple que la pipeline GNN fonctionne.

---

## `notebooks/02_yelpchi_pipeline.ipynb`

Notebook pour YelpChi.

Il sert à :

- charger YelpChi ;
- entraîner GraphSAGE ;
- entraîner XGBoost ;
- comparer les métriques.

Rôle simple :

> Expérience principale de fraude sur dataset public.

---

# 10. Dossier `docs/`

```text
docs/
├── RESULTATS.md
├── rapport_avancement_choix.md
├── rapport_avancement_choix.pdf
├── utilite_fichiers.md
└── superpowers/
```

---

## `docs/RESULTATS.md`

C’est le fichier des résultats expérimentaux.

Il contient :

- résultat Cora ;
- résultats YelpChi ;
- comparaison GraphSAGE vs XGBoost vs XGBoost+graph ;
- analyse de l’homophilie ;
- résultats du sweep de relations ;
- interprétation ;
- limites.

C’est un fichier très important pour le mémoire.

Rôle simple :

> Regrouper les chiffres et leur interprétation.

---

## `docs/rapport_avancement_choix.md`

C’est la version Markdown du rapport d’avancement créé pour résumer :

- l’état du projet ;
- les choix pris ;
- pourquoi ces choix ont été faits ;
- les résultats ;
- les prochaines étapes ;
- les limites.

Rôle simple :

> Avoir un rapport écrit et modifiable sur l’avancement du projet.

---

## `docs/rapport_avancement_choix.pdf`

C’est la version PDF du rapport d’avancement.

Rôle simple :

> Avoir une version propre à partager ou présenter.

---

## `docs/utilite_fichiers.md`

C’est ce document.

Il explique le rôle de chaque fichier du projet.

Rôle simple :

> Comprendre rapidement à quoi sert chaque fichier.

---

# 11. Dossier `docs/superpowers/`

```text
docs/superpowers/
├── specs/
└── plans/
```

Ce dossier contient les documents de conception et de planification.

---

## `docs/superpowers/specs/2026-06-11-pfe-fraude-graph-ml-design.md`

C’est la spécification de conception.

Elle décrit :

- l’objectif ;
- les décisions d’architecture ;
- les modules prévus ;
- les interfaces ;
- les choix techniques.

Rôle simple :

> Décrire comment le projet doit être conçu.

---

## `docs/superpowers/plans/2026-06-11-pfe-fraude-graph-ml.md`

C’est le plan d’implémentation détaillé.

Il décrit les tâches étape par étape :

- créer les fichiers ;
- écrire les tests ;
- implémenter les modules ;
- lancer les tests ;
- faire les commits.

Rôle simple :

> Montrer comment le projet a été construit progressivement.

---

# 12. Résumé global

Le projet peut être vu comme une chaîne :

```text
Données
  ↓
src/data/
  ↓
Modèles
  ↓
src/models/
  ↓
Entraînement
  ↓
src/train/
  ↓
Évaluation
  ↓
src/eval/
  ↓
Expériences complètes
  ↓
src/experiments/
  ↓
Résultats
  ↓
docs/RESULTATS.md
```

En parallèle :

```text
tests/ = vérifie que chaque pièce marche
notebooks/ = lance les expériences facilement sur Colab
docs/ = explique les choix, résultats et avancement
```

Phrase courte à retenir :

> Le projet est organisé de façon modulaire. Le dossier `src` contient le code principal : chargement des données, modèles Graph ML, entraînement, métriques et expériences. Le dossier `tests` vérifie automatiquement chaque composant. Les notebooks servent uniquement de wrappers Colab pour lancer les expériences. Enfin, les fichiers `docs` et `AVANCEMENT.md` documentent les décisions, les résultats et l’évolution du projet.

# Rapport d’avancement — PFE Détection de fraude fiscale par Graph ML

**Date :** 2026-06-11  
**Projet :** `pfe-fraude-graph-ml`  
**Branche de travail :** `impl-pipeline`  

## 1. Objectif du projet

Le projet vise à construire un **outil de triage** qui attribue un **score de risque de fraude** à des contribuables, entreprises ou particuliers. Ce score sert à **prioriser les dossiers à examiner** par un humain : il ne constitue pas une décision automatique de fraude.

La question de recherche centrale est :

> Est-ce qu’exploiter les relations entre contribuables avec une approche graphe, notamment GraphSAGE, améliore la détection de fraude par rapport à une approche classique qui regarde chaque contribuable isolément, comme XGBoost ?

Le périmètre est volontairement limité à la fraude fiscale. Sont exclus : TVA spécifique, blanchiment, banque et cartes bancaires.

## 2. Idée scientifique

Un modèle classique de Machine Learning analyse surtout les caractéristiques individuelles d’un contribuable : chiffre d’affaires, retards, ratios, historique déclaratif, etc.

Le Graph Machine Learning ajoute une dimension relationnelle :

- même dirigeant ;
- même adresse ;
- même comptable ;
- relation client-fournisseur ;
- participation entre sociétés ;
- proximité avec des entités déjà connues comme frauduleuses.

L’hypothèse est qu’un contribuable peut paraître peu suspect individuellement, mais devenir prioritaire si son voisinage dans le graphe ressemble à celui de fraudeurs connus.

## 3. Choix d’architecture

### 3.1 Modules Python comme source de vérité

Le choix principal a été de mettre la logique métier dans des fichiers `.py` sous `src/`, et de garder les notebooks Colab comme wrappers fins.

**Pourquoi :**

- éviter la duplication de code dans les notebooks ;
- rendre les fonctions testables avec `pytest` ;
- faciliter la réutilisation sur Cora, YelpChi puis données fiscales ;
- rendre le projet plus propre pour un mémoire ou une soutenance.

### 3.2 GraphSAGE comme modèle principal

GraphSAGE a été choisi comme modèle GNN principal.

**Pourquoi :**

- simple à expliquer ;
- robuste ;
- standard dans PyTorch Geometric ;
- adapté à un PFE ;
- évite la complexité de modèles comme PC-GNN ou CARE-GNN, qui sont cités mais non reproduits.

### 3.3 XGBoost comme baseline

XGBoost sert de modèle de comparaison sans message-passing graphe.

**Pourquoi :**

- très performant sur données tabulaires ;
- représente une approche classique qui regarde les contribuables isolément ;
- permet de tester si le graphe apporte réellement quelque chose.

Une variante **XGBoost+graph** a aussi été utilisée : elle ajoute quelques features de graphe, comme le degré et le ratio de voisins frauduleux connu uniquement sur le train.

### 3.4 Pas de base graphe

Le projet n’utilise pas Neo4j, ArangoDB ou autre base graphe.

**Pourquoi :**

- inutile pour ce stade du PFE ;
- PyTorch Geometric manipule le graphe en mémoire ;
- cela simplifie l’environnement et réduit les dépendances.

## 4. Avancement réalisé

### Étape 1 — Validation sur Cora

Cora a été utilisé comme dataset “hello world” du Graph ML.

Résultat : GraphSAGE atteint une accuracy test de **0.798**, supérieure au seuil attendu de 0.75.

**Pourquoi cette étape :**

- vérifier que PyTorch Geometric fonctionne ;
- vérifier que GraphSAGE apprend correctement ;
- valider la boucle d’entraînement avant de passer à la fraude.

Cette étape ne répond pas à la question de recherche fiscale ; elle valide seulement la mécanique GNN.

### Étape 2 — Pipeline fraude sur YelpChi

YelpChi est un dataset public de détection de faux avis Yelp.

Caractéristiques utilisées :

- 45 954 nœuds ;
- 32 features ;
- environ 14,5 % de fraude ;
- plusieurs relations graphe : `homo`, `net_rur`, `net_rtr`, `net_rsr`.

La pipeline implémentée comprend :

- chargement du dataset ;
- normalisation des features ;
- entraînement GraphSAGE ;
- entraînement XGBoost ;
- ajout de features de graphe pour XGBoost+graph ;
- calcul des métriques adaptées au déséquilibre ;
- benchmark multi-seed ;
- analyse relation par relation.

### Étape 3 — Stub fiscal

Un fichier `src/data/fiscal_graph.py` prépare la future construction du graphe fiscal réel avec `HeteroData`.

Il prévoit notamment :

- contribuables entreprises ;
- particuliers ;
- relations client-fournisseur ;
- partage de dirigeant ;
- partage d’adresse ;
- partage de comptable ;
- participation.

Le corps n’est pas encore implémenté, car les vraies données fiscales ne sont pas encore disponibles.

## 5. Changement important : abandon de DGL

Le plan initial prévoyait de charger YelpChi avec DGL (`FraudDataset`).

En pratique, DGL a posé des problèmes de compatibilité, notamment avec `graphbolt` et `torchdata.datapipes` sur Colab.

Le choix final a donc été de charger directement le fichier `.mat` YelpChi avec `scipy`.

**Pourquoi ce choix est meilleur :**

- moins fragile ;
- fonctionne en local et sur Colab ;
- supprime une dépendance lourde ;
- garde la même donnée et les mêmes relations ;
- rend le projet plus reproductible.

## 6. Structure actuelle du code

### `src/config.py`

Contient le seed, le device CPU/GPU et les hyperparamètres d’entraînement.

### `src/data/cora.py`

Charge Cora pour valider la mécanique Graph ML.

### `src/data/yelpchi.py`

Télécharge et lit YelpChi depuis le fichier `.mat`, convertit les relations en `edge_index`, crée les splits train/val/test.

### `src/data/fiscal_graph.py`

Prépare le futur graphe fiscal hétérogène. Actuellement volontairement non implémenté.

### `src/data/graph_features.py`

Calcule des features de graphe sans fuite de labels : degré et ratio de voisins frauduleux calculé seulement avec les labels train.

### `src/data/transforms.py`

Normalise les features avec des statistiques calculées uniquement sur le train.

### `src/models/graphsage.py`

Implémente GraphSAGE à deux couches.

### `src/models/gat.py`

Implémente GAT, un modèle avec attention permettant de pondérer différemment les voisins.

### `src/models/multi_rel_gnn.py`

Implémente un GNN multi-relationnel qui utilise plusieurs relations YelpChi en parallèle.

### `src/train/train_gnn.py`

Contient la boucle d’entraînement des GNN, l’early stopping, les class weights et la prédiction des scores.

### `src/train/baseline_xgb.py`

Contient la baseline XGBoost.

### `src/eval/metrics.py`

Calcule AUC-ROC, AUC-PR, F1-macro, GMean et Recall@k.

### `src/experiments/benchmark.py`

Compare GraphSAGE, XGBoost et XGBoost+graph sur plusieurs seeds.

### `src/experiments/relation_sweep.py`

Teste GraphSAGE relation par relation pour identifier les relations les plus utiles.

### `src/experiments/multi_rel.py`

Lance l’expérience avec le GNN multi-relationnel.

## 7. Résultats actuels

### Cora

GraphSAGE atteint une accuracy de **0.798**, ce qui valide la pipeline GNN.

### YelpChi — comparaison principale

Résultats moyens sur 5 seeds :

- **GraphSAGE (homo)** : AUC-ROC 0.894 ± 0.002 ; AUC-PR 0.676 ± 0.006.
- **XGBoost** : AUC-ROC 0.946 ± 0.003 ; AUC-PR 0.819 ± 0.010.
- **XGBoost+graph** : AUC-ROC 0.951 ± 0.002 ; AUC-PR 0.830 ± 0.009.

Conclusion provisoire : sur YelpChi avec la relation `homo`, XGBoost bat GraphSAGE.

Mais cette conclusion doit être nuancée : le graphe `homo` est peu homophile, donc peu favorable au message-passing.

## 8. Analyse des relations

YelpChi contient plusieurs relations :

- `homo` ;
- `net_rur` ;
- `net_rtr` ;
- `net_rsr`.

La relation la plus intéressante est `net_rur`, qui correspond à R-U-R : deux avis reliés parce qu’ils sont écrits par le même utilisateur.

Homophilie observée :

- `net_rur` : **0.996** ;
- `homo` : 0.773 ;
- `net_rsr` : 0.772 ;
- `net_rtr` : 0.759.

`net_rur` est donc presque parfaitement homophile : deux nœuds connectés par cette relation ont presque toujours le même label.

Mais elle a une limite importante : elle est très sparse, avec environ 48,1 % de nœuds isolés. Elle donne donc un signal très propre, mais ne couvre pas suffisamment tout le graphe.

## 9. Pourquoi XGBoost bat GraphSAGE actuellement

GraphSAGE agrège l’information des voisins. Cela fonctionne bien si les voisins sont informatifs.

Sur YelpChi, la relation `homo` est presque non homophile. Cela signifie que les voisins ne partagent pas toujours le même label. Dans ce cas, le message-passing peut lisser ou diluer le signal.

XGBoost, lui, exploite très bien les features tabulaires et n’est pas perturbé par un graphe bruité.

Donc l’explication n’est pas “le Graph ML ne sert à rien”, mais plutôt :

> Le Graph ML dépend fortement de la qualité des relations. Si la relation est mauvaise ou peu homophile, le GNN peut être moins performant qu’un modèle tabulaire.

## 10. Pourquoi continuer en Graph ML

Les expériences montrent que la relation `net_rur` est fortement homophile et améliore GraphSAGE en AUC-ROC.

Cela prouve que la structure du graphe compte.

Les prochaines étapes Graph ML sont donc justifiées :

- **GAT** : pour apprendre quels voisins sont importants ;
- **GNN multi-relationnel** : pour combiner une relation propre mais sparse avec des relations plus denses ;
- **graphe fiscal hétérogène** : pour représenter correctement les entités fiscales réelles.

## 11. Homogène, multi-relationnel et hétérogène

Actuellement, Cora et YelpChi sont utilisés principalement comme graphes homogènes : les nœuds sont du même type.

YelpChi est aussi multi-relationnel, car il contient plusieurs types de relations entre les mêmes types de nœuds.

Pour la vraie IA fiscale, le choix le plus cohérent est un graphe hétérogène, car les données fiscales contiennent naturellement plusieurs types d’entités :

- entreprises ;
- particuliers ;
- dirigeants ;
- adresses ;
- comptables ;
- déclarations ;
- relations commerciales.

Le fichier `fiscal_graph.py` prépare déjà cette direction avec `HeteroData`.

## 12. Métriques choisies

Les métriques utilisées sont :

- **AUC-ROC** : capacité globale à classer positifs au-dessus des négatifs ;
- **AUC-PR** : plus pertinente lorsque la fraude est rare ;
- **F1-macro** : équilibre entre classes ;
- **GMean** : équilibre sensibilité/spécificité ;
- **Recall@k** : utile pour le triage, car il mesure combien de fraudeurs sont retrouvés dans les dossiers les plus prioritaires.

L’accuracy n’est pas utilisée comme métrique principale, car la fraude est déséquilibrée : un modèle qui prédit toujours “non fraudeur” peut avoir une bonne accuracy tout en étant inutile.

## 13. État actuel

Le projet dispose actuellement :

- d’une pipeline GNN fonctionnelle ;
- d’une baseline XGBoost ;
- d’une variante XGBoost+graph ;
- de métriques adaptées ;
- de tests automatisés ;
- de résultats expérimentaux documentés ;
- d’un début de stratégie multi-relationnelle ;
- d’un stub pour le futur graphe fiscal hétérogène.

## 14. Ce qui reste à faire

Les prochaines étapes sont :

1. benchmarker GAT ;
2. terminer le benchmark multi-relationnel ;
3. analyser si le multi-relationnel améliore GraphSAGE ;
4. construire le vrai graphe fiscal lorsque les données arrivent ;
5. passer d’un graphe public YelpChi à un graphe fiscal métier ;
6. documenter les limites et le biais de sélection dans le mémoire.

## 15. Limites à mentionner

### 15.1 YelpChi n’est pas le fiscal

YelpChi est un dataset public de faux avis. Il sert à valider la méthode, mais il ne reproduit pas exactement les données fiscales.

### 15.2 Labels partiels

Dans le fiscal, seuls les contribuables déjà enquêtés ont un label. Les autres sont inconnus, pas forcément non-fraudeurs.

### 15.3 Pas de décision automatique

Le modèle donne un score de risque pour prioriser les contrôles. Il ne doit pas être présenté comme une preuve de fraude.

### 15.4 Pas de comparaison directe avec PC-GNN

PC-GNN et CARE-GNN sont cités comme références, mais non reproduits. Il ne faut pas comparer directement les chiffres obtenus ici avec ceux des papiers si les protocoles ne sont pas identiques.

## 16. Conclusion provisoire

Le projet a atteint un stade solide : la pipeline Graph ML est fonctionnelle, testée et comparée à une baseline forte.

Les résultats actuels montrent que XGBoost bat GraphSAGE sur YelpChi avec la relation `homo`, mais l’analyse relationnelle montre que le graphe reste porteur d’information. La relation `net_rur`, très homophile, confirme que la qualité des relations est déterminante.

La suite logique est de passer vers des modèles plus adaptés aux relations multiples, puis vers un graphe fiscal hétérogène réel. C’est dans ce contexte métier que le Graph ML a le plus de chances d’apporter une valeur ajoutée par rapport au ML tabulaire classique.

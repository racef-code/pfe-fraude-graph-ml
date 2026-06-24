# Claude Code audit — PFE fraude fiscale Graph ML

Commande utilisée : `claude --print --model claude-opus-4-8 --effort high` depuis le repo.

I have enough to produce the audit. Here it is.

---

# Audit — PFE Détection de fraude fiscale (Graph ML)

**Auditeur :** senior AI/research engineer · **Date :** 2026-06-18 · **Branche :** `impl-pipeline`
**Périmètre :** direction scientifique, qualité, risques, fuite/méthodo, architecture, prochaines étapes.
Aucune modification de fichier effectuée (audit en lecture seule).

**Légende sévérité :** 🔴 Bloquant pour une thèse défendable · 🟠 Important · 🟡 Mineur / hygiène · 🟢 Point fort

---

## 0. Verdict résumé

La **direction est juste** et le projet est bien plus solide que « on a lancé un GNN sur YelpChi ». Le récit scientifique (la fraude IS est relationnelle et hétérogène ; il faut prouver que les relations apportent au-delà du tabulaire) est le bon et l'infrastructure pour le défendre existe (baseline isolée, GNN hétérogène, modèle relation-gated, diagnostics, ablations, contrat de données).

**Mais** la preuve actuelle de « graph advantage » repose entièrement sur un **dataset synthétique construit pour que le graphe gagne**, avec une **baseline volontairement affaiblie** et **3 seeds sur 240 entreprises**. Telles quelles, les têtes de chapitre (+0.25 AUC-PR) sont **non défendables devant un jury exigeant**. Le travail est au bon endroit ; il faut maintenant **fermer la baseline manquante (XGBoost+graph-features)** et **discipliner l'évaluation** avant d'ajouter de la complexité modèle.

---

## 1. Est-ce la bonne direction ? 🟢 Oui, avec un recadrage

**Ce qui est juste :**
- Cadrer le PFE sur la **question « les relations apportent-elles au-delà du tabulaire ? »** plutôt que « on a un GNN » est exactement la bonne posture de recherche.
- Choix de l'**hétérogène** (`company` cible + person/address/accountant/sector) : cohérent avec la réalité métier de la fraude IS (dirigeant/adresse/comptable communs, participations).
- La séquence Cora (sanity) → YelpChi (banc public) → synthétique (smoke) → contrat CSV pour les vraies données est **méthodologiquement saine** et honnête sur ce que chaque étape prouve.

**Le recadrage nécessaire :** la contribution scientifique d'un PFE ne peut **pas** être un résultat sur données synthétiques. Le synthétique doit être présenté **uniquement** comme validation d'ingénierie/protocole (ce que les docs disent déjà, 🟢), et l'effort doit basculer au plus vite vers (a) un **dataset public corporate/fiscal réel** et (b) la **rigueur d'évaluation**. La valeur défendable du mémoire sera : *un protocole reproductible + des diagnostics qui décident quand le graphe aide + un résultat sur données réelles/quasi-réelles*, pas le chiffre synthétique.

---

## 2. Ce qui est bon 🟢

1. **Séparation des préoccupations propre** : données (`fiscal_graph`/`real_fiscal`) ↔ modèles ↔ entraînement ↔ métriques ↔ expériences. Le contrat CSV (`contrat_donnees_is.md`) découple l'arrivée des vraies données du code.
2. **Conscience anti-fuite réelle** : normalisation `company` sur le train uniquement (`standardize_hetero_features`), labels `NaN` (non-enquêté) exclus des masks et jamais traités comme négatifs (`_make_masks`, contrat §1) — c'est le piège classique de la fraude fiscale, et il est explicitement géré.
3. **Métriques adaptées au déséquilibre** : AUC-PR, F1-macro, GMean, Recall@k au lieu d'accuracy. Comparaisons **appariées par seed** (paired delta) — bon réflexe.
4. **Diagnostics avant modèle** (`graph_diagnostics.py`) : projeter chaque relation en paires company-company et mesurer couverture/degré/homophilie/lift est la bonne façon de *justifier* qu'une relation porte du signal avant de l'attribuer au GNN.
5. **Ablation leave-one-relation-out** : le bon outil pour montrer que « graph advantage ≠ tout empiler ».
6. **Honnêteté documentée** : limites, biais de sélection, « pas une preuve scientifique » sont écrits noir sur blanc. C'est rare et précieux.
7. **TDD / 33 tests** : bonne hygiène d'ingénierie.

---

## 3. Faiblesses et risques

### 🔴 3.1 Le « graph advantage » est circulaire (synthétique + baseline affaiblie)
Le générateur **injecte** le signal graphe (fraudeurs partagent des pools risqués de personnes/adresses/comptables et transactent entre eux : `fiscal_graph.py:279-314`) **et** rend délibérément le signal tabulaire bruité (`marge`, `taux_is`, etc. avec σ large, commentaire ligne 249-250). Conclusion mécanique : le GNN bat XGBoost **par construction**. Ce n'est pas une découverte, c'est une tautologie. Les +0.25 AUC-PR ne mesurent que « le pipeline sait lire un signal qu'on y a mis ». À ne **jamais** présenter comme résultat, seulement comme « le pipeline fonctionne ».

### 🔴 3.2 La baseline décisive manque (XGBoost + features de graphe)
La comparaison actuelle est **XGBoost (features company seules)** vs **GNN (company + graphe)**. Elle confond deux choses : *« l'information de graphe aide »* et *« le message-passing aide »*. Un relecteur dira immédiatement : *des features de graphe simples (degré par relation, nb de voisins fraudeurs sur labels train, comptes partagés) suffisent peut-être*. Tant que `XGBoost+fiscal-graph-features` n'existe pas (planifié §6.2 du work log mais **non fait**), **aucune** affirmation « il faut un GNN » n'est soutenable. C'est la priorité n°1.

### 🟠 3.3 Puissance statistique insuffisante
240 entreprises, ~15 % de fraude → **~7 fraudeurs dans le test**, **3 seeds**. Les écarts-types le trahissent : AUC-PR ±0.094 à ±0.166 ; Recall@k = 0.5714 ± 0.0000 (figé car k≈4/7 sur un test minuscule). Les deltas « 3/3 positifs » reposent sur n=3. Pour toute affirmation : **≥10 seeds** et **n_companies nettement plus grand** (≥2000), sinon intervalles de confiance ininterprétables.

### 🟠 3.4 La métrique « homophilie » est trompeuse telle que rapportée
Avec un taux de fraude de 0.15, la fraction attendue de paires de même label **au hasard** est p²+(1−p)² ≈ **0.745**. Or `RESULTATS_GRAPH_ADVANTAGE.md` rapporte des homophilies de 0.72–0.78 pour **toutes** les relations — c'est-à-dire **le niveau du hasard**. Présenter ça comme « relations homophiles » serait faux. Le vrai signal est le `fraud_neighbor_lift` (1.6–3.9). **Recommandation :** rapporter l'homophilie **relativement au null** (0.745) ou la retirer, et mettre en avant le lift.

### 🟠 3.5 `fraud_neighbor_lift` : définition fragile / potentiellement optimiste
`relation_diagnostics` (lignes 73-89) calcule le taux de fraude parmi les *voisins de fraudeurs*, sur **tous** les labels connus (train+val+test). Sur données réelles ce diagnostic doit être calculé **sur labels train seulement**, sinon il « voit » le test. Acceptable pour un diagnostic exploratoire, mais à documenter et à restreindre au train dès qu'on touche du réel.

### 🟠 3.6 Évaluation purement transductive, non alignée sur l'usage réel
Le GNN voit le graphe entier (features des entreprises test incluses) à l'entraînement. C'est standard, mais l'usage cible (scorer une **nouvelle** entreprise) est **inductif**. Le synthétique surestime donc la perf opérationnelle. À déclarer explicitement, et prévoir au moins une évaluation inductive (mask des nouveaux nœuds) avant le mémoire.

### 🟡 3.7 `RelationGatedFiscalGNN` = complexité prématurée
Le gain réel vs equal-sum est **AUC-PR +0.0002 ± 0.0509** (bruit pur) ; seuls F1/GMean bougent, sur 3 seeds. Le modèle est élégant (gates softmax par type de destination, interprétables via `relation_weights`) mais **non justifié empiriquement** aujourd'hui. À garder comme piste d'interprétabilité, **pas** comme « upgrade » avant que la baseline graph-features et les vraies données existent. Ne pas surinvestir l'architecture tant que le banc d'essai est synthétique.

### 🟡 3.8 Pas de tuning ni de budget égal documenté
XGBoost tourne en quasi-défaut, GNN avec une config fixe. Pour une comparaison défendable, fixer un protocole de sélection d'hyperparamètres **identique** (même budget de recherche sur val) pour chaque modèle, et le documenter.

---

## 4. Fuite / méthodologie — synthèse

| # | Point | Statut |
|---|---|---|
| 4.1 | Normalisation `company` sur train only | 🟢 Correct (`transforms.py:39-41`) |
| 4.2 | Labels `NaN` exclus des masks, jamais négatifs | 🟢 Correct (`_make_masks`, contrat §1) |
| 4.3 | Mêmes splits pour XGBoost et GNN | 🟢 Correct (masks partagés, `fiscal_synthetic.py:44-45`) |
| 4.4 | Labels jamais utilisés comme features de nœud | 🟢 Correct |
| 4.5 | `fraud_neighbor_lift` calculé sur tous labels (train+test) | 🟠 À restreindre au train sur données réelles (§3.5) |
| 4.6 | Futures features de graphe à base de labels | 🟠 **Risque à venir** : §6.2 du work log prévoit « nb de voisins fraudeurs » — impératif **train-only** (déjà noté par l'auteur, à faire respecter en code + test) |
| 4.7 | Pas de split temporel | 🟠 Sur données réelles, la fraude est datée et différée : features=passé, label=contrôle ultérieur (contrat §4.3 le dit mais aucun code ne l'impose). Un split aléatoire fuiterait le futur. |
| 4.8 | Standardisation appelée **avant** le split sur données réelles ? | 🟡 Non : `build_fiscal_graph` crée les masks puis `standardize` utilise `train_mask` → OK. Mais `real_fiscal.load_fiscal_csv_graph` standardise toujours globalement les nœuds support (sans label) — acceptable car non-cibles, à garder tel quel. |

**Verdict fuite :** le noyau est propre et la conscience du problème est réelle. Les risques sont **futurs** (features de graphe à base de labels, split temporel sur données réelles) et doivent être verrouillés par des tests **avant** de toucher au réel.

---

## 5. Architecture du code

🟢 **Globalement bonne** : modulaire, testée, contrat de données explicite.

Points à corriger :

- 🟡 **Init paresseuse fragile** : `SAGEConv((-1,-1), …)` impose un forward « à blanc » manuel avant de créer l'optimiseur (`fiscal_synthetic.py:60`, `fiscal_ablation.py:43`). Si on l'oublie, l'optimiseur reçoit une liste de params vide et l'entraînement échoue silencieusement. → Encapsuler l'init (méthode `lazy_init(data)` ou `torch_geometric.nn.to_hetero` / appel interne dans `train_hetero_company_gnn`).
- 🟡 **Duplication** `FiscalHeteroGNN` ↔ `RelationGatedFiscalGNN` (deux couches, head, dropout identiques). Factoriser une base commune.
- 🟡 **`data.metadata_obj`** (node_maps, feature_cols) est attaché mais **jamais utilisé** dans les expériences (qui appellent `data.metadata()`). Il sera utile pour `company_explainer` ; soit l'exploiter, soit documenter son usage prévu pour éviter qu'il diverge.
- 🟡 **`class_weight` appliqué aussi à la val_loss** (`train_hetero.py:42`) : l'early-stopping optimise une perte pondérée, pas la métrique cible. Préférer early-stop sur **AUC-PR de validation** (la métrique qui compte), pas sur la cross-entropy pondérée.
- 🟡 **`TrainConfig` par défaut (epochs=200, dropout=0.5, hidden=64) ≠ config réellement utilisée** (epochs=30-40, dropout=0.2, hidden=32 dans les scripts). Désalignement source de confusion ; centraliser les configs d'expérience.
- 🟢 Gestion des arêtes orphelines (skip plutôt que crash) et arêtes inverses auto : bon réflexe robustesse.

---

## 6. Les 5 prochaines étapes, classées

### 1. 🔴 Implémenter `XGBoost + fiscal-graph-features` (la baseline qui décide tout)
Features de graphe **train-label-only** : degré par relation, nb de voisins fraudeurs par relation (labels train uniquement), comptes de comptable/adresse partagés, flags de couverture. Ajouter au banc à 4-5 modèles. **Sans cette baseline, aucune conclusion « il faut un GNN » n'est publiable.** Verrouiller le no-leak par un test dédié (déjà fait pour YelpChi via `graph_features.py` — réutiliser le pattern).

### 2. 🔴 Discipliner l'évaluation : ≥10 seeds, n≥2000, et test de significativité
Augmenter taille et seeds ; rapporter IC bootstrap ou Wilcoxon apparié sur les deltas. Re-générer `RESULTATS_*` avec ces réglages. Reformuler explicitement tout chiffre synthétique comme « validation de pipeline », jamais comme résultat.

### 3. 🟠 Obtenir un dataset corporate/fiscal réaliste (sortir du synthétique)
Prioriser un dataset public **inductif et daté** plus proche du métier que YelpChi (pistes du work log : TED `T20H`/`T15S`, KeHGN-R — vérifier accès/licence). À défaut, durcir le synthétique : rendre le **signal tabulaire compétitif** (pour que la victoire du graphe ne soit plus garantie) et ajouter du **bruit de label** (fraudeurs non détectés). Le but : un banc où le graphe peut *perdre*.

### 4. 🟠 Verrouiller la méthodo « données réelles » par du code+tests avant l'arrivée des données
(a) split **temporel** dans `build_fiscal_graph` (option `time_col`), (b) test garantissant que toute feature à base de label n'utilise que le train, (c) une passe d'évaluation **inductive** (nœuds test masqués au message-passing entrant depuis labels). Corriger l'homophilie rapportée **relativement au null 0.745** (§3.4) et restreindre `fraud_neighbor_lift` au train (§3.5).

### 5. 🟡 Nettoyage architecture + early-stop sur AUC-PR, *puis seulement* interprétabilité
Factoriser les deux GNN, encapsuler l'init paresseuse, aligner `TrainConfig`, passer l'early-stopping sur AUC-PR val. Ensuite seulement, valoriser `RelationGatedFiscalGNN.relation_weights` et le `company_explainer` (§6.3) comme **volet interprétabilité/triage** — argument fort pour un jury (« score = aide à la décision, pas accusation »), mais après que les baselines soient solides.

---

### Un point à ne pas perdre de vue
La meilleure version de ce mémoire ne dira pas « notre GNN fait 0.95 ». Elle dira : *« voici un protocole qui décide, diagnostics à l'appui, quand les relations valent un GNN plutôt qu'un XGBoost+features de graphe ; sur données réelles biaisées et datées, voici le verdict honnête, et voici une couche d'explication exploitable par un contrôleur »*. Tout le code nécessaire à ce récit est presque là — il manque surtout la **baseline graph-features** et la **rigueur d'évaluation**, pas de l'architecture.

Souhaitez-vous que je détaille l'étape 1 (liste exacte des features de graphe + le test anti-fuite associé) ou que je rédige la version corrigée de la section diagnostics/homophilie ?

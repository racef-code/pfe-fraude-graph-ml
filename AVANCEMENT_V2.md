# Avancement V2 — ce qui a été fait après la dernière séance

> Suite de `AVANCEMENT.md` (= phase 1). Couvre la **phase 2** : pivot vers la vraie
> cible (fraude fiscale IS d'entreprise, graphe hétérogène), l'audit, et les corrections.
> Date : 2026-06-18. Branche : `impl-pipeline` (repo public). **41 tests pytest passent.**

## 0. Résumé en une phrase

Le projet est passé d'un banc d'essai homogène (YelpChi) à une **pipeline de fraude
fiscale d'entreprise hétérogène**, a subi un **audit critique**, et la conclusion
honnête actuelle est : *le graphe contient beaucoup de signal, mais une baseline
XGBoost+features-de-graphe bat encore les GNNs* — donc tout reste à prouver sur des
**données réelles** (le synthétique est circulaire).

## 1. Le pivot : vers la fraude fiscale IS (hétérogène)

La vraie cible n'est pas YelpChi ni la fraude générique : c'est le **scoring de risque
de fraude IS au niveau entreprise**, par nature **relationnel et hétérogène**. Cora et
YelpChi ne sont plus que des sanity checks / benchmarks publics.

**Construit** (commits `0823218`, `83ec65e`) :
- `src/data/fiscal_graph.py` — vrai builder `HeteroData` (remplace le stub).
  - Nœud cible : `company`. Support : `person`, `address`, `accountant`, `sector`.
  - Relations : `transaction`, `has_director`, `registered_at`, `uses_accountant`,
    `in_sector`, `participates_in` (+ arêtes inverses).
  - Labels `NaN` (non-enquêtés) → exclus des masks, **jamais traités comme négatifs**.
  - `make_synthetic_is_fiscal_tables(...)` : générateur de données synthétiques IS.
- Modèles :
  - `FiscalHeteroGNN` (`hetero_fiscal_gnn.py`) : 2 couches `HeteroConv`+`SAGEConv`,
    somme **égale** des messages de toutes les relations, head sur les nœuds `company`.
  - `RelationGatedFiscalGNN` (`relation_gated_fiscal_gnn.py`) : au lieu de la somme
    égale, **gates softmax appris par type de relation** (downweighte les relations
    bruitées). Poids inspectables via `relation_weights` → piste d'interprétabilité.
- `src/data/real_fiscal.py` + `docs/contrat_donnees_is.md` : adaptateur CSV + format
  attendu des vraies données (pour brancher le réel sans changer le code).
- `src/analysis/graph_diagnostics.py` : diagnostics par relation (couverture, degré,
  homophilie, **fraud_neighbor_lift**) — pour juger si une relation porte du signal
  AVANT d'entraîner.
- `src/experiments/fiscal_ablation.py` : ablation leave-one-relation-out.

**Diagnostics & ablation (résultats concrets) :**
- Homophilie de toutes les relations ≈ 0.72–0.78 ≈ **niveau du hasard** (null 0.745) →
  le vrai signal est le `fraud_neighbor_lift` (1.6–3.9 : un voisin de fraudeur est
  1.6 à 3.9× plus souvent fraudeur).
- Ablation leave-one-out (chute d'AUC-PR si on retire la relation) : `transaction`
  **+0.043** (utile), `uses_accountant` **+0.041** (utile), `registered_at` +0.012,
  `has_director`/`participates_in`/`in_sector` négatifs (bruités sur ce synthétique).
  → « graph advantage ≠ tout empiler » : certaines relations aident, d'autres nuisent.

## 2. L'audit (Claude Code) — `docs/CLAUDE_CODE_AUDIT.md`

Audit critique en lecture seule. Verdict : **bonne direction**, mais :
- 🔴 le « graph advantage » est **circulaire** (synthétique avec signal injecté +
  baseline tabulaire affaiblie → le GNN gagne par construction).
- 🔴 **baseline décisive manquante** : `XGBoost + features de graphe`.
- 🟠 puissance statistique faible (3 seeds / 240 entreprises).
- 🟠 homophilie rapportée ≈ niveau du hasard (0.745) → trompeuse.
- 🟡 `RelationGatedFiscalGNN` = complexité prématurée ; early-stop sur loss pondérée.

## 3. Corrections appliquées — `docs/AUDIT_CHANGES_APPLIED.md`

- ✅ **Baseline `XGBoost+graph-features`** ajoutée (`src/features/fiscal_graph_features.py`),
  **leak-free** (features à base de labels = labels **train uniquement**), test dédié
  (`test_fiscal_graph_features_do_not_leak_test_labels` : changer les labels test ne
  change pas la matrice). Features par relation (projetées en paires company-company) :
  `log_degree`, `has_edge`, `log_train_neighbor_count`, `train_fraud_neighbor_ratio`
  (+ versions globales `any_relation__*`).
- ✅ Diagnostics corrigés : homophilie **relative au null** + `fraud_neighbor_lift` mis en avant.
- ✅ Diagnostics de label restreignables au **train only** (`label_company_ids`).
- ✅ Warning si < 10 seeds ; run sérieux 10 seeds / 2000 entreprises lancé.
- ✅ `src/experiments/fiscal_relation_selection.py` (sélection de relations).

## 4. Le résultat honnête (run renforcé 10 seeds / 2000 entreprises)

`docs/RESULTATS_SYNTHETIC_STRONGER_AUDIT.md` :

| Modèle | AUC-PR |
|---|---:|
| XGBoost company-only | 0.465 |
| **XGBoost + graph-features** | **0.915** |
| FiscalHeteroGNN | 0.797 |
| RelationGatedFiscalGNN | 0.836 |

- Le graphe apporte énormément (XGBoost+graph ≫ company-only, 10/10 seeds).
- **Mais les GNNs ne battent pas la baseline graph-features en AUC-PR.**
- Question de recherche reformulée : *« quand le message-passing dépasse-t-il une
  baseline graph-features forte ? »* — plus défendable qu'une victoire vs company-only.
- ⚠️ Synthétique = **validation d'ingénierie**, pas un résultat scientifique (circulaire).

## 5. Dataset TED (cible fiscale réelle) — `docs/RESULTATS_TED_TAX_DATASET.md`

- `src/data/ted_tax.py` + `ted_tax_benchmark.py` : adaptateur pour **TED** (détection
  d'évasion fiscale sur graphe hétérogène, DMKD 2025).
- Seul un **mini-échantillon** (31 nœuds) est dispo localement → smoke test, pas un
  benchmark. Le full dataset doit être téléchargé (lien Baidu du repo TED).

## 6. Dataset synthétique sauvé en local (cette séance)

- `src/data/generate_synthetic_is.py` : génère et **sauve** les tables synthétiques
  → `data/synthetic_is/nodes.csv` + `edges.csv` (gitignoré, local).
  Exemple généré : 2974 nœuds (2000 entreprises), 14654 arêtes, 15 % fraude.
- Données = **rule-based**, pas aléatoires, pas réelles (`make_synthetic_is_fiscal_tables`,
  `fiscal_graph.py:221-315`). Règles exactes :

### 6.1 Features tabulaires (signal réel mais volontairement bruité)
Pour chaque entreprise, distributions différentes fraude vs non-fraude :

| Feature | Fraude | Non-fraude | bruit (σ / loi) |
|---|---|---|---|
| `ca` (chiffre d'affaires) | — | — | lognormal(μ=12.0, σ=0.8) |
| `marge` | 0.09 | 0.13 | N(·, σ=0.07) |
| `taux_is_effectif` | 0.16 | 0.22 | N(·, σ=0.08), ≥0 |
| `retards_declaration` | λ=1.4 | λ=0.7 | Poisson |
| `rectifications` | λ=0.8 | λ=0.35 | Poisson |
| `variation_ca` | 0.20 | 0.10 | N(·, σ=0.30) |
| `ratio_charges` | 0.82 | 0.74 | N(·, σ=0.16), clip[0,1.5] |

`resultat = ca × marge`, `impot_is = resultat × taux_is`. **σ larges → distributions
qui se chevauchent → signal tabulaire faible** (pour laisser le graphe apporter).

### 6.2 Structure du graphe (signal injecté chez les fraudeurs)
Nœuds support : 8 secteurs, `n/20` comptables, `n/10` adresses, `n/3` personnes.
**Pools « risqués »** : 1/6 des personnes, 1/5 des adresses, 1/4 des comptables.

Câblage des fraudeurs (proba de piocher dans le pool risqué plutôt qu'au hasard) :
- `has_director` (personne) : **0.75** · `registered_at` (adresse) : **0.70** ·
  `uses_accountant` (comptable) : **0.65**
- `transaction` : `Poisson(3 si fraude, sinon 2) + 1` arêtes ; une transaction de
  fraudeur vise un autre fraudeur avec proba **0.55**, sinon une entreprise au hasard.
- `participates_in` (participation/holding, sparse) : proba **0.35** (fraude) / **0.12**
  (sinon) ; cible un autre fraudeur avec proba **0.65**.
- `in_sector` : secteur au hasard (pas de signal).

→ Conséquence : fraudeurs **regroupés** (partagent personnes/adresses/comptables) et
**inter-connectés** (transactions/participations entre eux). C'est ce signal de
structure que le graphe/GNN peut lire — d'où la **circularité** notée par l'audit.

## 7. État actuel

- **41 tests pytest passent** (progression : 18 phase 1 → 26 → 33 → 41).
- Repo public, branche `impl-pipeline`, PR #1.
- Pipeline complète : données (homogène + hétérogène + contrat réel + générateur local),
  5 modèles (XGBoost, XGBoost+graph, GraphSAGE/GAT, FiscalHeteroGNN, RelationGated),
  diagnostics, ablations, sélection de relations, métriques, benchmarks multi-seed.
- Commandes clés :
  - `python -m src.experiments.fiscal_synthetic 10 2000 20` (benchmark renforcé)
  - `python -m src.experiments.fiscal_ablation 3 240 40` (ablation relations)
  - `python -m src.data.generate_synthetic_is 2000 0.15 42` (génère le dataset local)

## 8. Ce qui reste (priorités, depuis l'audit)

1. 🔴 **Vrai dataset** : full TED (ou fiscal réel) — la seule preuve qui compte.
2. 🔴 **Rigueur d'éval** : ≥10 seeds (fait), n≥2000 (fait), + test de significativité.
3. 🟠 **Méthodo données réelles** : split **temporel** + évaluation **inductive** (scorer
   une entreprise nouvelle, pas vue à l'entraînement).
4. 🟡 Nettoyage : early-stop sur AUC-PR val, factoriser les 2 GNN.
5. 🟡 `company_explainer` : couche d'explication / triage (argument jury fort).

## 9. Note de cadrage

La meilleure version du mémoire ne dira pas « notre GNN fait 0.95 ». Elle dira :
*« voici un protocole + des diagnostics qui décident quand les relations valent un GNN
plutôt qu'un XGBoost+features-de-graphe ; sur données réelles biaisées et datées, voici
le verdict honnête, et voici une couche d'explication exploitable par un contrôleur. »*

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
- Modèles : `hetero_fiscal_gnn.py` (`FiscalHeteroGNN`, HeteroConv+SAGEConv),
  `relation_gated_fiscal_gnn.py` (`RelationGatedFiscalGNN`, poids appris par relation).
- `src/data/real_fiscal.py` + `docs/contrat_donnees_is.md` : adaptateur CSV + format
  attendu des vraies données (pour brancher le réel sans changer le code).
- `src/analysis/graph_diagnostics.py` : diagnostics par relation (couverture, degré,
  homophilie, **fraud_neighbor_lift**) — pour juger si une relation porte du signal
  AVANT d'entraîner.
- `src/experiments/fiscal_ablation.py` : ablation leave-one-relation-out.

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
  **leak-free** (features à base de labels = labels **train uniquement**), test dédié.
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
  selon les règles → `data/synthetic_is/nodes.csv` + `edges.csv` (gitignoré, local).
- Données = **rule-based** (fraudeurs câblés pour partager des pools risqués et
  transactionner entre eux ; signal tabulaire volontairement bruité). Pas aléatoire,
  pas réel.

## 7. État actuel

- **41 tests pytest passent.** Repo public, branche `impl-pipeline`, PR #1.
- Pipeline complète : données (homogène + hétérogène + contrat réel), 5 modèles
  (XGBoost, XGBoost+graph, GraphSAGE/GAT, FiscalHeteroGNN, RelationGated), diagnostics,
  ablations, sélection de relations, métriques, benchmarks multi-seed.

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

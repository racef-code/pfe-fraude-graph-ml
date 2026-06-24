# Hermes work log and plan — PFE fraude fiscale Graph ML

Date: 2026-06-17  
Repository: `pfe-fraude-graph-ml`  
Branch: `impl-pipeline`

## 1. Context and main objective

The project is a PFE about fraud detection with Graph Machine Learning. The real target is not generic fraud and not banking/card fraud: it is **IS/company-oriented fiscal fraud risk scoring**. The desired final model should detect whether a company/taxpayer is suspicious from its own behavior and from its contacts/relations.

The key research concern is:

> Are graph relations useful beyond tabular company features, and can a heterogeneous GNN exploit fiscal/company relations better than a classical isolated model such as XGBoost?

The true real-world data is expected later. Until then, the project uses:

1. Cora only as a PyG/GNN sanity check.
2. YelpChi as a public graph-fraud benchmark, but not as fiscal/company evidence.
3. A synthetic IS/company heterogeneous graph as an engineering and methodology smoke test.
4. A documented CSV contract so real/anonymized fiscal data can be plugged in later.

## 2. What was already in the project before my recent work

The repository already had a solid baseline pipeline:

- `src/data/cora.py`: Cora loader.
- `src/data/yelpchi.py`: YelpChi `.mat` loader without DGL, avoiding DGL/GraphBolt fragility.
- `src/models/graphsage.py`: homogeneous GraphSAGE.
- `src/models/gat.py`: GAT model.
- `src/models/multi_rel_gnn.py`: simple multi-relation YelpChi model.
- `src/train/train_gnn.py`: GNN training loop.
- `src/train/baseline_xgb.py`: XGBoost baseline.
- `src/data/graph_features.py`: leak-free graph features for XGBoost+graph.
- `src/eval/metrics.py`: AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k.
- `src/experiments/benchmark.py`: GraphSAGE vs XGBoost vs XGBoost+graph.
- `src/experiments/relation_sweep.py`: relation sweep on YelpChi.
- `src/experiments/multi_rel.py`: multi-relation GNN experiment.
- `docs/RESULTATS.md`: results showing that on YelpChi, XGBoost beats GraphSAGE on `homo`, but graph features help XGBoost and relation quality matters.

The important existing insight was: **YelpChi is useful but not enough**, because the real fiscal problem is naturally heterogeneous and company/entity based.

## 3. What I changed recently

### 3.1 Heterogeneous fiscal IS pipeline

Commit:

```text
0823218 feat: add heterogeneous fiscal IS graph pipeline
```

Main additions:

- `src/data/fiscal_graph.py`
  - Replaced the old `NotImplementedError` stub with a real `HeteroData` builder.
  - Target node type: `company`.
  - Support node types: `person`, `address`, `accountant`, `sector`.
  - Relations: `transaction`, `has_director`, `registered_at`, `uses_accountant`, `in_sector`, `participates_in`.
  - Adds reverse edges so `company` can receive messages from support nodes.
  - Handles labels only on `company` nodes.
  - Keeps unknown/non-investigated labels as `NaN` -> encoded as `-1` and excluded from train/val/test masks.
  - Includes `make_synthetic_is_fiscal_tables(...)` to generate synthetic IS data for pipeline validation.

- `src/models/hetero_fiscal_gnn.py`
  - Added `FiscalHeteroGNN` using `HeteroConv` + `SAGEConv`.
  - Predicts only company nodes.

- `src/train/train_hetero.py`
  - Added training and prediction functions for heterogeneous company-node classification.

- `src/data/real_fiscal.py`
  - CSV adapter for future real/anonymized fiscal data.
  - Loads `nodes.csv` + `edges.csv`, builds `HeteroData`, standardizes features.

- `src/experiments/fiscal_synthetic.py`
  - Synthetic benchmark comparing XGBoost-company-features and heterogeneous GNN.

- `src/data/transforms.py`
  - Added anti-leak `standardize_hetero_features(...)`.
  - `company` stats use train companies only; support nodes can be normalized globally.

- Docs:
  - `docs/contrat_donnees_is.md`: real-data format expected.
  - `docs/recherche_datasets_is.md`: dataset/paper research notes.
  - `docs/RESULTATS_SYNTHETIC_IS.md`: synthetic benchmark results.

- Tests:
  - `tests/test_fiscal_graph.py`
  - `tests/test_real_fiscal.py`
  - `tests/test_transforms.py`

Verified at that point:

```text
26 passed
```

### 3.2 Graph advantage diagnostics and ablations

Commit:

```text
83ec65e feat: add graph advantage diagnostics and ablations
```

Main additions:

- `src/analysis/graph_diagnostics.py`
  - Computes relation-level diagnostics:
    - raw edge count;
    - projected company-company pairs;
    - coverage;
    - isolated company rate;
    - average degree;
    - homophily;
    - fraud-neighbor lift.
  - Purpose: determine whether graph relations carry useful signal before training a model.

- `src/experiments/fiscal_ablation.py`
  - Runs leave-one-relation-out ablations:
    - `all_relations`
    - `without_transaction`
    - `without_has_director`
    - `without_registered_at`
    - `without_uses_accountant`
    - `without_participates_in`
    - `without_in_sector`
  - Purpose: identify which relations actually help the GNN.

- `src/models/relation_gated_fiscal_gnn.py`
  - Added `RelationGatedFiscalGNN`.
  - Instead of equal-summing all relation messages, it learns scalar gates per edge type.
  - Purpose: handle noisy/sparse relations and provide an architecture better aligned with heterogeneous fraud graphs.

- Synthetic generator update:
  - Added `participates_in` relation to better mimic company/holding/control structures.

- Docs:
  - `docs/GRAPH_ADVANTAGE_PLAN.md`
  - `docs/RESULTATS_GRAPH_ADVANTAGE.md`

- Tests:
  - `tests/test_graph_diagnostics.py`
  - `tests/test_fiscal_ablation.py`
  - `tests/test_relation_gated_fiscal_gnn.py`

Verified after that work:

```text
33 passed in 9.39s
```

## 4. Latest verified outputs

### 4.1 Synthetic model benchmark

Command:

```bash
python -m src.experiments.fiscal_synthetic 3 240 40
```

Output summary:

| Model | AUC-ROC | AUC-PR | F1-macro | GMean | Recall@k |
|---|---:|---:|---:|---:|---:|
| XGBoost-company-features | 0.8246 ± 0.0146 | 0.5193 ± 0.0255 | 0.7260 ± 0.0362 | 0.6905 ± 0.0795 | 0.5714 ± 0.0000 |
| FiscalHeteroGNN | 0.9477 ± 0.0103 | 0.7724 ± 0.0940 | 0.7097 ± 0.0826 | 0.8396 ± 0.0595 | 0.6667 ± 0.0673 |
| RelationGatedFiscalGNN | 0.9570 ± 0.0214 | 0.7726 ± 0.1447 | 0.8248 ± 0.0482 | 0.9136 ± 0.0389 | 0.6667 ± 0.0673 |

Interpretation:

- Synthetic results show the pipeline can exploit graph signal when the graph contains signal.
- Relation-gated model improves F1/GMean and slightly AUC-ROC over equal-sum HeteroGNN.
- AUC-PR is similar between equal-sum and gated models on the synthetic setting.
- This is **not** evidence that the model works on real fiscal fraud yet; it is an engineering/protocol validation.

### 4.2 Relation ablation benchmark

Command:

```bash
python -m src.experiments.fiscal_ablation 3 240 40
```

Key relation-importance output by AUC-PR drop vs all relations:

| Relation | AUC-PR drop | Interpretation |
|---|---:|---|
| transaction | +0.0431 | useful |
| uses_accountant | +0.0411 | useful for AUC-PR/AUC-ROC, threshold metrics mixed |
| registered_at | +0.0117 | small useful signal |
| has_director | -0.0331 | noisy in this synthetic run |
| participates_in | -0.0367 | high lift but sparse/noisy; needs better weighting/selection |
| in_sector | -0.0431 | broad/noisy context |

Interpretation:

- The graph advantage is not simply “use all relations.”
- Some relations are useful, some are noisy, and relation selection or gated/attention models may matter.
- This supports the PFE narrative: graph design and relation quality matter as much as the GNN architecture.

## 5. Why I think this is the right direction

The project’s strongest scientific direction is not “we used a GNN.” It is:

> Fiscal/company fraud is naturally relational and heterogeneous. A useful model should evaluate company behavior plus the relational context around companies: transactions, shared accountants, shared addresses, directors, sectors, and ownership/participation links.

The current code now supports that story because it includes:

1. A classical isolated baseline: XGBoost on company features.
2. A heterogeneous message-passing model: FiscalHeteroGNN.
3. A relation-aware model: RelationGatedFiscalGNN.
4. Relation diagnostics before training.
5. Relation ablations after training.
6. A real-data contract for future IS company data.

This is much stronger than only running GraphSAGE on YelpChi.

## 6. What I was planning / thinking to do next

### 6.1 Relation selection experiment

The next concrete experiment should be relation selection, not just leave-one-out ablation.

Planned variants:

```text
all_relations
transaction_only
accountant_only
address_only
transaction_accountant
transaction_accountant_address
no_sector
no_participation
diagnostic_selected
```

Purpose:

- Test whether a curated graph beats the full graph.
- Show that graph quality matters, not only graph quantity.
- Prepare a clear recommendation for real fiscal data collection: which relations matter most.

### 6.2 XGBoost + fiscal graph features baseline

The current synthetic benchmark has XGBoost-company-features vs GNNs. We still need a stronger baseline:

```text
XGBoost-company-features
XGBoost+fiscal-graph-features
FiscalHeteroGNN
RelationGatedFiscalGNN
Relation-selected GNN
```

Purpose:

- Separate “graph information helps” from “message passing helps.”
- A reviewer could argue that simple graph features may be enough. This baseline addresses that.

Potential graph features:

- degree by relation;
- shared-accountant count;
- shared-address count;
- transaction degree;
- train-only fraud-neighbor ratio by relation;
- relation coverage flags;
- number of fraud neighbors by relation, using train labels only.

Critical rule:

- Any feature using labels must use **train labels only** to avoid leakage.

### 6.3 Company risk explanation layer

Planned module:

```text
src/analysis/company_explainer.py
```

Purpose:

- Produce human-readable context for high-risk companies.
- Explain model output as triage, not automatic accusation.

Example output:

```text
Company c_042 has high risk because:
- shares an accountant with 4 known fraud companies;
- has transactions with 2 known fraud companies;
- is registered at an address shared by suspicious companies;
- has late declarations and low effective IS rate.
```

### 6.4 Try a closer public dataset

Investigate and possibly adapt:

- TED `T20H` / `T15S` if accessible.
- KeHGN-R if accessible and legally usable.

Purpose:

- Reduce reliance on synthetic data before the real fiscal dataset arrives.
- Move from YelpChi/general fraud toward corporate/tax graph data.

### 6.5 Improve model only after baselines are strong

Possible upgrades later:

- relation-gated model with sparsity/regularization;
- relation dropout;
- HGT/HAN-style attention;
- temporal splits if dates are available;
- label-noise-aware training because real fiscal fraud labels are biased and delayed.

But I would not jump immediately to a complex architecture before implementing relation selection and XGBoost+graph-features.

## 7. Risks / concerns to audit

Please audit these points carefully:

1. Is the current direction scientifically appropriate for a PFE on fiscal/company fraud and Graph ML?
2. Is the synthetic dataset too artificial or misleading, even with warnings?
3. Are the graph diagnostics and ablations the right way to prove graph advantage?
4. Is `RelationGatedFiscalGNN` a useful upgrade or premature complexity?
5. Should the next step be relation selection, XGBoost+graph features, TED/KeHGN-R dataset integration, or something else?
6. Are there hidden leakage risks in normalization, graph features, train/val/test masks, or label usage?
7. Are there code architecture issues that should be fixed before more experiments?
8. What would you suggest next as a senior AI engineer/research engineer?

## 8. Current verification state

Latest known full test command:

```bash
python -m pytest -q
```

Latest known result:

```text
33 passed in 9.39s
```

Latest local commits:

```text
83ec65e feat: add graph advantage diagnostics and ablations
0823218 feat: add heterogeneous fiscal IS graph pipeline
```

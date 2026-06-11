# PFE Fraude Graph ML — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Graph-ML fraud-triage pipeline (GraphSAGE + XGBoost baseline) validated on Cora and YelpChi, with a fiscal-graph stub for real data.

**Architecture:** `.py` modules under `src/` are the single source of truth; Colab notebooks are thin wrappers that import them. GraphSAGE (PyG, homogeneous) and XGBoost consume the same features and splits, evaluated by the same 5 KPIs, to answer the research question.

**Tech Stack:** Python 3.12, PyTorch 2.9 (CPU local / CUDA Colab), PyTorch Geometric, XGBoost, scikit-learn, pandas, DGL (Colab-only for YelpChi), pytest.

**Design choice (refinement of spec §4):** `train_gnn` uses `CrossEntropyLoss(weight=class_weight)` with `out_dim = num_classes`. This is uniform across multiclass Cora and binary YelpChi; for the binary case the positive-class weight `= n_neg / n_pos` reproduces the `pos_weight` effect required by piège n°2. `predict_scores` returns `softmax(logits)[:, 1]` (positive-class probability) for binary tasks.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned deps for local CPU dev |
| `src/__init__.py`, `src/*/__init__.py` | Package markers |
| `src/config.py` | seed, device, `TrainConfig` |
| `src/eval/metrics.py` | 5 KPIs + comparison table |
| `src/models/graphsage.py` | 2-layer GraphSAGE |
| `src/data/cora.py` | Cora loader → PyG `Data` |
| `src/train/train_gnn.py` | GNN train loop + predict |
| `src/train/baseline_xgb.py` | XGBoost baseline |
| `src/data/yelpchi.py` | YelpChi loader (Colab-only) |
| `src/data/fiscal_graph.py` | fiscal HeteroData STUB |
| `notebooks/01_cora_demo.ipynb` | Étape 1 wrapper |
| `notebooks/02_yelpchi_pipeline.ipynb` | Étape 2 wrapper |
| `README.md` | usage |
| `tests/` | pytest suite (CPU-local) |

---

## Task 0: Environment & scaffold

**Files:**
- Create: `requirements.txt`, `README.md`
- Create: `src/__init__.py`, `src/config.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/train/__init__.py`, `src/eval/__init__.py`, `tests/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: Write `requirements.txt`**

```
torch==2.9.1
torch_geometric>=2.6
xgboost>=2.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
pytest>=8.0
```

- [ ] **Step 2: Install local dev deps**

Run: `pip install torch_geometric xgboost scikit-learn pandas pytest`
Expected: installs succeed (torch 2.9.1+cpu already present). DGL intentionally NOT installed locally.

- [ ] **Step 3: Create package markers**

All `__init__.py` files empty. Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -v
```

- [ ] **Step 4: Write `src/config.py`**

```python
import random
from dataclasses import dataclass
import numpy as np
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    lr: float = 0.01
    epochs: int = 200
    weight_decay: float = 5e-4
    dropout: float = 0.5
    patience: int = 30  # early-stop on val loss
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini src/ tests/__init__.py
git commit -m "chore: scaffold package, config, deps"
```

---

## Task 1: Metrics (`src/eval/metrics.py`)

**Files:**
- Create: `src/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.eval.metrics import compute_metrics, print_comparison


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, scores, k=2)
    assert m["auc_roc"] == 1.0
    assert m["auc_pr"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["gmean"] == 1.0
    assert m["recall_at_k"] == 1.0


def test_recall_at_k_partial():
    y_true = np.array([0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1])  # top-2 = idx 0,1 -> 1 of 3 positives
    m = compute_metrics(y_true, scores, k=2)
    assert abs(m["recall_at_k"] - 1 / 3) < 1e-9


def test_default_k_is_positive_count():
    y_true = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    m = compute_metrics(y_true, scores)  # k defaults to n_pos = 1
    assert m["recall_at_k"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with "No module named 'src.eval.metrics'" / ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
"""Évaluation : KPIs adaptés au déséquilibre (jamais l'accuracy)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_metrics(y_true, scores, k: int | None = None, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    n_pos = int((y_true == 1).sum())
    if k is None:
        k = n_pos
    topk = np.argsort(scores)[::-1][:k]
    recall_at_k = int((y_true[topk] == 1).sum()) / n_pos if n_pos else 0.0

    return {
        "auc_roc": roc_auc_score(y_true, scores),
        "auc_pr": average_precision_score(y_true, scores),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "gmean": float(np.sqrt(sensitivity * specificity)),
        "recall_at_k": recall_at_k,
    }


def print_comparison(results: dict[str, dict]) -> None:
    """results = {'GraphSAGE': {...}, 'XGBoost': {...}}. Gabarit PC-GNN."""
    keys = ["auc_roc", "auc_pr", "f1_macro", "gmean", "recall_at_k"]
    header = "Model".ljust(12) + "".join(k.ljust(13) for k in keys)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = name.ljust(12) + "".join(f"{m[k]:.4f}".ljust(13) for k in keys)
        print(row)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: imbalance-aware metrics (AUC-ROC/PR, F1-macro, GMean, Recall@k)"
```

---

## Task 2: GraphSAGE model (`src/models/graphsage.py`)

**Files:**
- Create: `src/models/graphsage.py`
- Test: `tests/test_graphsage.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.models.graphsage import GraphSAGE


def test_forward_shape():
    model = GraphSAGE(in_dim=8, hidden_dim=16, out_dim=3, dropout=0.5)
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (10, 3)


def test_two_sageconv_layers():
    from torch_geometric.nn import SAGEConv
    model = GraphSAGE(in_dim=4, hidden_dim=8, out_dim=2, dropout=0.0)
    convs = [m for m in model.modules() if isinstance(m, SAGEConv)]
    assert len(convs) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graphsage.py -v`
Expected: FAIL ImportError "No module named 'src.models.graphsage'".

- [ ] **Step 3: Write minimal implementation**

```python
"""GraphSAGE 2 couches (piège n°6 : 2-3 couches max, sinon over-smoothing)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graphsage.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/graphsage.py tests/test_graphsage.py
git commit -m "feat: 2-layer GraphSAGE model"
```

---

## Task 3: Cora loader (`src/data/cora.py`)

**Files:**
- Create: `src/data/cora.py`
- Test: `tests/test_cora.py`

- [ ] **Step 1: Write the failing test**

```python
from src.data.cora import load_cora


def test_load_cora_structure():
    data = load_cora()
    assert data.num_nodes == 2708
    assert data.x.shape[1] == 1433
    assert int(data.y.max()) == 6  # 7 classes (0..6)
    for mask in ("train_mask", "val_mask", "test_mask"):
        assert hasattr(data, mask)
        assert data[mask].sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cora.py -v`
Expected: FAIL ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
"""Étape 1 : Cora (graphe de citations) pour valider la mécanique GNN."""
from torch_geometric.datasets import Planetoid

_CACHE = "data/cora"


def load_cora():
    dataset = Planetoid(root=_CACHE, name="Cora")
    return dataset[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cora.py -v`
Expected: 1 passed (downloads Cora to `data/cora` on first run).

- [ ] **Step 5: Commit**

```bash
git add src/data/cora.py tests/test_cora.py
git commit -m "feat: Cora loader (Etape 1)"
```

Add `data/` to `.gitignore`:

```bash
echo "data/" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore data cache"
```

---

## Task 4: GNN training (`src/train/train_gnn.py`)

**Files:**
- Create: `src/train/train_gnn.py`
- Test: `tests/test_train_gnn.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.config import TrainConfig, set_seed
from src.data.cora import load_cora
from src.models.graphsage import GraphSAGE
from src.train.train_gnn import train_gnn, predict_scores


def test_train_gnn_learns_on_cora():
    set_seed(42)
    data = load_cora()
    model = GraphSAGE(data.x.shape[1], 64, int(data.y.max()) + 1, dropout=0.5)
    cfg = TrainConfig(epochs=60, patience=60)
    model = train_gnn(model, data, cfg)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    assert acc > 0.75  # critère succès Étape 1


def test_predict_scores_binary_shape():
    set_seed(0)
    import torch_geometric
    from torch_geometric.data import Data
    x = torch.randn(20, 5)
    edge_index = torch.randint(0, 20, (2, 40))
    y = torch.tensor([0, 1] * 10)
    mask = torch.ones(20, dtype=torch.bool)
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=mask, val_mask=mask, test_mask=mask)
    model = GraphSAGE(5, 8, 2, dropout=0.0)
    scores = predict_scores(model, data)
    assert scores.shape == (20,)
    assert ((scores >= 0) & (scores <= 1)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_gnn.py -v`
Expected: FAIL ImportError "src.train.train_gnn".

- [ ] **Step 3: Write minimal implementation**

```python
"""Boucle d'entraînement GNN full-batch + early-stop. CrossEntropy avec
class_weight gère le déséquilibre (piège n°2)."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from src.config import DEVICE


def train_gnn(model, data, cfg, class_weight=None):
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    if class_weight is not None:
        class_weight = class_weight.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                               weight=class_weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask],
                                       weight=class_weight).item()
        if val_loss < best_val:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_scores(model, data):
    """Probabilité de la classe positive (index 1)."""
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index), dim=1)
    return probs[:, 1].cpu().numpy()


def class_weights_from_labels(y) -> torch.Tensor:
    """w_c = N / (n_classes * count_c). Pour le binaire -> pondère le positif."""
    y = np.asarray(y.cpu() if hasattr(y, "cpu") else y)
    classes, counts = np.unique(y, return_counts=True)
    w = len(y) / (len(classes) * counts)
    return torch.tensor(w, dtype=torch.float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_gnn.py -v`
Expected: 2 passed (Cora training ~30-60s CPU).

- [ ] **Step 5: Commit**

```bash
git add src/train/train_gnn.py tests/test_train_gnn.py
git commit -m "feat: GNN training loop with early-stop and class weighting"
```

---

## Task 5: XGBoost baseline (`src/train/baseline_xgb.py`)

**Files:**
- Create: `src/train/baseline_xgb.py`
- Test: `tests/test_baseline_xgb.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.train.baseline_xgb import train_xgb, predict_xgb


def test_xgb_learns_separable_data():
    rng = np.random.default_rng(0)
    X_pos = rng.normal(2, 0.5, (50, 4))
    X_neg = rng.normal(-2, 0.5, (50, 4))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)
    model = train_xgb(X, y, scale_pos_weight=1.0)
    scores = predict_xgb(model, X)
    assert scores.shape == (100,)
    assert ((scores >= 0) & (scores <= 1)).all()
    # separable -> AUC perfect
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, scores) > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_baseline_xgb.py -v`
Expected: FAIL ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
"""Baseline XGBoost : mêmes features (data.x), AUCUNE info de graphe.
Point de comparaison 'sans graphe' pour la question de recherche."""
import numpy as np
from xgboost import XGBClassifier


def train_xgb(X_train, y_train, scale_pos_weight: float = 1.0):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # déséquilibre (piège n°2)
        eval_metric="aucpr",
        n_jobs=-1,
    )
    model.fit(np.asarray(X_train), np.asarray(y_train))
    return model


def predict_xgb(model, X):
    return model.predict_proba(np.asarray(X))[:, 1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_baseline_xgb.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/train/baseline_xgb.py tests/test_baseline_xgb.py
git commit -m "feat: XGBoost baseline (no graph)"
```

---

## Task 6: YelpChi loader (`src/data/yelpchi.py`) — Colab-only

**Files:**
- Create: `src/data/yelpchi.py`
- Test: `tests/test_yelpchi_import.py` (import-only; DGL not installed locally)

- [ ] **Step 1: Write the import-only test**

```python
import importlib


def test_yelpchi_module_imports_without_dgl():
    """Module must import even when DGL absent (lazy import inside function)."""
    mod = importlib.import_module("src.data.yelpchi")
    assert hasattr(mod, "load_yelpchi")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yelpchi_import.py -v`
Expected: FAIL ImportError "src.data.yelpchi".

- [ ] **Step 3: Write minimal implementation**

```python
"""Étape 2 : YelpChi via DGL FraudDataset -> PyG Data homogène.
Colab-only (DGL pénible sur Windows). DGL importé en lazy pour que le
module reste importable en local sans DGL installé."""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def load_yelpchi(seed: int = 42) -> Data:
    import dgl  # lazy: absent en local
    from dgl.data import FraudDataset

    dataset = FraudDataset("yelp")
    g = dataset[0]
    g = dgl.to_homogeneous(g, ndata=["feature", "label", "train_mask",
                                      "val_mask", "test_mask"])

    x = g.ndata["feature"].float()
    y = g.ndata["label"].long()
    src, dst = g.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)

    n = x.shape[0]
    idx = np.arange(n)
    train_idx, tmp = train_test_split(idx, test_size=0.4, random_state=seed,
                                      stratify=y.numpy())
    val_idx, test_idx = train_test_split(tmp, test_size=0.5, random_state=seed,
                                         stratify=y.numpy()[tmp])

    def _mask(indices):
        m = torch.zeros(n, dtype=torch.bool)
        m[indices] = True
        return m

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=_mask(train_idx),
                val_mask=_mask(val_idx),
                test_mask=_mask(test_idx))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_yelpchi_import.py -v`
Expected: 1 passed (import works; `load_yelpchi` body only runs on Colab).

- [ ] **Step 5: Commit**

```bash
git add src/data/yelpchi.py tests/test_yelpchi_import.py
git commit -m "feat: YelpChi loader (Colab-only, lazy DGL import)"
```

---

## Task 7: Fiscal graph STUB (`src/data/fiscal_graph.py`)

**Files:**
- Create: `src/data/fiscal_graph.py`
- Test: `tests/test_fiscal_graph_stub.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import pandas as pd
from src.data.fiscal_graph import build_fiscal_graph


def test_stub_raises_not_implemented():
    nodes = pd.DataFrame({"id": [1, 2]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "type": ["client_fournisseur"]})
    with pytest.raises(NotImplementedError):
        build_fiscal_graph(nodes, edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fiscal_graph_stub.py -v`
Expected: FAIL ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
"""Étape 3 (STUB) : construction du graphe fiscal hétérogène depuis 3 tables.
Corps à implémenter quand les vraies données arrivent (Étape 4)."""
import pandas as pd
from torch_geometric.data import HeteroData

# Relations attendues (cahier des charges §4.1)
RELATION_TYPES = [
    "client_fournisseur",
    "partage_dirigeant",
    "partage_adresse",
    "partage_comptable",
    "participation",
]
# Types de nœuds
NODE_TYPES = ["entreprise", "particulier"]


def build_fiscal_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> HeteroData:
    """Construit un HeteroData depuis les tables Contribuables + Relations.

    nodes_df: id, type (entreprise/particulier), features (CA/revenu, secteur,
        ancienneté, effectif, impôt déclaré, ratios, retards, rectifications, écarts),
        label (fraudeur/non-fraudeur, enquêtés seulement -> NaN sinon).
    edges_df: id_source, id_cible, type_relation (cf. RELATION_TYPES).

    TODO (Étape 4, données réelles) :
      1. Séparer les nœuds par type ; encoder/normaliser les features par type.
      2. Indexer les ids -> indices contigus par type de nœud.
      3. Pour chaque type de relation -> HeteroData[src, rel, dst].edge_index.
      4. Masque de labels = uniquement les contribuables enquêtés
         (biais de sélection : non-enquêtés != négatifs).
      5. Splits train/val/test stratifiés sur les seuls nœuds labellisés.
    """
    raise NotImplementedError(
        "Stub fiscal : à compléter en Étape 4 avec les vraies données."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fiscal_graph_stub.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data/fiscal_graph.py tests/test_fiscal_graph_stub.py
git commit -m "feat: fiscal graph HeteroData stub (Etape 3)"
```

---

## Task 8: Full suite + notebooks

**Files:**
- Create: `notebooks/01_cora_demo.ipynb`, `notebooks/02_yelpchi_pipeline.ipynb`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (Cora downloads on first run).

- [ ] **Step 2: Create `notebooks/01_cora_demo.ipynb`**

Cells (use jupytext or write JSON; content per cell):

```python
# Cell 1 — setup (Colab)
!pip -q install torch_geometric
!git clone https://github.com/<user>/pfe-fraude-graph-ml.git || true
%cd pfe-fraude-graph-ml
```

```python
# Cell 2 — Étape 1 : GraphSAGE sur Cora
from src.config import TrainConfig, set_seed
from src.data.cora import load_cora
from src.models.graphsage import GraphSAGE
from src.train.train_gnn import train_gnn
import torch

set_seed(42)
data = load_cora()
model = GraphSAGE(data.x.shape[1], 64, int(data.y.max()) + 1, dropout=0.5)
model = train_gnn(model, data, TrainConfig(epochs=200))
model.eval()
with torch.no_grad():
    pred = model(data.x.to('cpu'), data.edge_index.to('cpu')).argmax(1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print("Cora test accuracy:", acc.item())
```

- [ ] **Step 3: Create `notebooks/02_yelpchi_pipeline.ipynb`**

```python
# Cell 1 — setup (Colab, DGL ici)
!pip -q install torch_geometric dgl xgboost scikit-learn
!git clone https://github.com/<user>/pfe-fraude-graph-ml.git || true
%cd pfe-fraude-graph-ml
```

```python
# Cell 2 — Étape 2 : GraphSAGE vs XGBoost sur YelpChi
from src.config import TrainConfig, set_seed
from src.data.yelpchi import load_yelpchi
from src.models.graphsage import GraphSAGE
from src.train.train_gnn import train_gnn, predict_scores, class_weights_from_labels
from src.train.baseline_xgb import train_xgb, predict_xgb
from src.eval.metrics import compute_metrics, print_comparison

set_seed(42)
data = load_yelpchi()
y = data.y.numpy()
test = data.test_mask.numpy()

# GraphSAGE
cw = class_weights_from_labels(data.y[data.train_mask])
gnn = GraphSAGE(data.x.shape[1], 64, 2, dropout=0.5)
gnn = train_gnn(gnn, data, TrainConfig(epochs=200), class_weight=cw)
gnn_scores = predict_scores(gnn, data)

# XGBoost (mêmes features, sans graphe)
import numpy as np
X = data.x.numpy()
n_neg, n_pos = (y[data.train_mask.numpy()] == 0).sum(), (y[data.train_mask.numpy()] == 1).sum()
xgb = train_xgb(X[data.train_mask.numpy()], y[data.train_mask.numpy()],
                scale_pos_weight=n_neg / n_pos)
xgb_scores = predict_xgb(xgb, X)

# Comparaison
results = {
    "GraphSAGE": compute_metrics(y[test], gnn_scores[test]),
    "XGBoost":   compute_metrics(y[test], xgb_scores[test]),
}
print_comparison(results)
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/
git commit -m "feat: Colab notebooks (Cora demo, YelpChi pipeline)"
```

---

## Task 9: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# PFE — Détection de fraude fiscale par Graph ML

Outil de triage : score de risque de fraude (GraphSAGE) vs baseline sans graphe (XGBoost).

## Structure
- `src/` — modules (source de vérité)
- `notebooks/` — wrappers Colab (Étape 1 Cora, Étape 2 YelpChi)
- `docs/superpowers/` — spec + plan
- `tests/` — suite pytest (CPU-local)

## Dev local (CPU)
```
pip install -r requirements.txt
pytest -v
```
DGL/YelpChi tournent seulement sur Colab (`notebooks/02_yelpchi_pipeline.ipynb`).

## Question de recherche
GraphSAGE (relations) vs XGBoost (isolé) sur AUC-ROC, AUC-PR, F1-macro, GMean, Recall@k.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README"
```

---

## Self-Review

**Spec coverage:**
- D1 modules/notebooks → Task 0–9 ✓ | D2 YelpChi Colab-only → Task 6 lazy import ✓
- D3 fiscal stub → Task 7 ✓ | D4 homogeneous GraphSAGE → Task 2 ✓
- D5 GraphSAGE → Task 2 ✓ | D6 XGBoost same features → Task 5 + notebook Task 8 ✓
- D7 device auto → Task 0 config ✓
- §5 flux → notebook 02 (Task 8) ✓ | §6 imbalance → train_gnn class_weight + xgb scale_pos_weight ✓
- §8 critères : Cora acc>0.75 → Task 4 test ✓ ; YelpChi 5-KPI table → Task 8 ✓

**Placeholder scan:** notebook git-clone URL `<user>` is a real user-fill (repo not yet on GitHub) — acceptable, flagged in notebook. No other placeholders.

**Type consistency:** `GraphSAGE(in_dim, hidden_dim, out_dim, dropout)`, `train_gnn(model, data, cfg, class_weight)`, `predict_scores(model, data)`, `compute_metrics(y_true, scores, k, threshold)`, `train_xgb(X, y, scale_pos_weight)`, `predict_xgb(model, X)` — consistent across tasks and notebook. ✓

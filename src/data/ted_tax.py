"""Adapter for the TED tax-evasion heterogeneous graph dataset.

TED source: https://github.com/yimingxu24/TED
Paper/repo describe T20H/T15S for related-party-transaction guided tax evasion
detection. This adapter does not vendor the dataset; it reads an existing TED
clone/download and maps the data to this project's fiscal HeteroData contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.fiscal_graph import build_fiscal_graph
from src.data.transforms import standardize_hetero_features

NODE_TYPE_MAP_T20H = {
    0: "company",
    1: "event",
    2: "item",
    3: "person",
}

EDGE_TYPE_MAP_T20H = {
    0: "transaction",
    1: "info_change",
    2: "company_sell_item",
    3: "company_buy_item",
    4: "person_company",
    5: "belong_to",
}


@dataclass(frozen=True)
class TEDLabelSplit:
    train_label_file: str = "label_TaxPayer1v9.dat"
    test_label_file: str = "label_TaxPayer1v9.dat.test"
    val_fraction: float = 0.2


def _read_node_dat(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 4:
                continue
            node_id = int(parts[0])
            node_type = NODE_TYPE_MAP_T20H.get(int(parts[2]), f"type_{parts[2]}")
            attrs = [float(x) for x in parts[3].split(",") if x != ""]
            row = {"id": node_id, "type": node_type, "label": np.nan}
            for i, val in enumerate(attrs):
                row[f"feat_{i}"] = val
            rows.append(row)
    return pd.DataFrame(rows)


def _read_link_dat(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 4:
                continue
            rows.append({
                "id_source": int(parts[0]),
                "id_cible": int(parts[1]),
                "type_relation": EDGE_TYPE_MAP_T20H.get(int(parts[2]), f"rel_{parts[2]}"),
                "weight": float(parts[3]),
            })
    return pd.DataFrame(rows)


def _read_labels(path: Path) -> dict[int, int]:
    labels = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 4:
                continue
            labels[int(parts[0])] = int(parts[3])
    return labels


def load_ted_tables(
    dataset_dir: str | Path,
    *,
    split: TEDLabelSplit = TEDLabelSplit(),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load TED node/link/label files as fiscal-style tables.

    TED provides train and test label files. We preserve the official test file
    and carve a stratified validation set from the train labels.
    """
    dataset_dir = Path(dataset_dir)
    nodes = _read_node_dat(dataset_dir / "node.dat")
    edges = _read_link_dat(dataset_dir / "link.dat")
    train_labels = _read_labels(dataset_dir / split.train_label_file)
    test_labels = _read_labels(dataset_dir / split.test_label_file)

    nodes = nodes.copy()
    all_labels = {**train_labels, **test_labels}
    nodes["label"] = nodes["id"].map(all_labels)

    # Split official train labels into train/val masks. build_fiscal_graph creates
    # random masks, so we override them after graph construction.
    rng = np.random.default_rng(seed)
    train_ids = np.asarray(list(train_labels.keys()))
    y_train = np.asarray([train_labels[int(i)] for i in train_ids])
    val_ids = []
    keep_train_ids = []
    for cls in sorted(set(y_train.tolist())):
        cls_ids = train_ids[y_train == cls]
        rng.shuffle(cls_ids)
        n_val = max(1, int(round(len(cls_ids) * split.val_fraction))) if len(cls_ids) > 3 else 0
        val_ids.extend(cls_ids[:n_val].tolist())
        keep_train_ids.extend(cls_ids[n_val:].tolist())

    split_ids = {
        "train_ids": set(map(int, keep_train_ids)),
        "val_ids": set(map(int, val_ids)),
        "test_ids": set(map(int, test_labels.keys())),
    }
    stats = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_train": int(len(split_ids["train_ids"])),
        "n_val": int(len(split_ids["val_ids"])),
        "n_test": int(len(split_ids["test_ids"])),
        "n_features": int(sum(c.startswith("feat_") for c in nodes.columns)),
    }
    return nodes, edges, {"stats": stats, "splits": split_ids}


def load_ted_heterodata(dataset_dir: str | Path, *, split: TEDLabelSplit = TEDLabelSplit(), seed: int = 42):
    nodes, edges, meta = load_ted_tables(dataset_dir, split=split, seed=seed)
    data = build_fiscal_graph(nodes, edges, seed=seed)
    company_ids = list(data["company"].node_id)
    train_ids = meta["splits"]["train_ids"]
    val_ids = meta["splits"]["val_ids"]
    test_ids = meta["splits"]["test_ids"]
    import torch

    data["company"].train_mask = torch.tensor([cid in train_ids for cid in company_ids], dtype=torch.bool)
    data["company"].val_mask = torch.tensor([cid in val_ids for cid in company_ids], dtype=torch.bool)
    data["company"].test_mask = torch.tensor([cid in test_ids for cid in company_ids], dtype=torch.bool)
    data["company"].labeled_mask = data["company"].train_mask | data["company"].val_mask | data["company"].test_mask
    data = standardize_hetero_features(data, target_node_type="company")
    return data, nodes, edges, meta

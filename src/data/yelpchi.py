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

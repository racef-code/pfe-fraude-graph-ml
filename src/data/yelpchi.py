"""Étape 2 : YelpChi (fraude faux avis Yelp).

Chargé directement depuis le fichier .mat hébergé par DGL (data.dgl.ai),
SANS la librairie DGL. Évite l'enfer de dépendances DGL/graphbolt/torchdata
sur Colab et tourne aussi en local (scipy + torch + PyG seulement).

Graphe homogène (relation 'homo' par défaut) pour GraphSAGE.
Le .mat contient : homo, net_rur, net_rtr, net_rsr (adjacences),
features (45954x32), label (binaire, déséquilibré).
"""
import io
import os
import urllib.request
import zipfile

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

_URL = "https://data.dgl.ai/dataset/FraudYelp.zip"
_CACHE_DIR = "data"
_MAT = os.path.join(_CACHE_DIR, "YelpChi.mat")

# Relations disponibles dans le .mat
RELATIONS = ["homo", "net_rur", "net_rtr", "net_rsr"]


def _download() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if os.path.exists(_MAT):
        return
    raw = urllib.request.urlopen(_URL, timeout=300).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(_CACHE_DIR)


def load_yelpchi(seed: int = 42, relation: str = "homo") -> Data:
    """Charge YelpChi en graphe homogène PyG.

    relation : quelle adjacence utiliser comme arêtes (défaut 'homo' = union).
    Renvoie un objet Data avec x, edge_index, y, train/val/test_mask
    (splits stratifiés 60/20/20).
    """
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}, got {relation!r}")
    _download()
    mat = sio.loadmat(_MAT)

    x = torch.tensor(np.asarray(mat["features"].todense()), dtype=torch.float)
    y = torch.tensor(np.asarray(mat["label"]).flatten(), dtype=torch.long)

    adj = sp.coo_matrix(mat[relation])
    edge_index = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)

    n = x.shape[0]
    idx = np.arange(n)
    ynp = y.numpy()
    train_idx, tmp = train_test_split(
        idx, test_size=0.4, random_state=seed, stratify=ynp
    )
    val_idx, test_idx = train_test_split(
        tmp, test_size=0.5, random_state=seed, stratify=ynp[tmp]
    )

    def _mask(indices):
        m = torch.zeros(n, dtype=torch.bool)
        m[indices] = True
        return m

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=_mask(train_idx),
        val_mask=_mask(val_idx),
        test_mask=_mask(test_idx),
    )


def load_yelpchi_multi(seed: int = 42, relations=("net_rur", "net_rtr", "net_rsr")):
    """Charge YelpChi avec PLUSIEURS relations (pour GNN multi-relationnel).

    Renvoie (data, edge_index_list) :
      - data : Data standard (x, y, masks) ; data.edge_index = 1re relation (compat).
      - edge_index_list : un edge_index par relation, dans l'ordre `relations`.
    Splits stratifiés 60/20/20 identiques à load_yelpchi (même seed).
    """
    _download()
    mat = sio.loadmat(_MAT)
    x = torch.tensor(np.asarray(mat["features"].todense()), dtype=torch.float)
    y = torch.tensor(np.asarray(mat["label"]).flatten(), dtype=torch.long)

    edge_index_list = []
    for rel in relations:
        if rel not in RELATIONS:
            raise ValueError(f"relation inconnue: {rel!r}")
        adj = sp.coo_matrix(mat[rel])
        edge_index_list.append(
            torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
        )

    n = x.shape[0]
    idx = np.arange(n)
    ynp = y.numpy()
    train_idx, tmp = train_test_split(
        idx, test_size=0.4, random_state=seed, stratify=ynp
    )
    val_idx, test_idx = train_test_split(
        tmp, test_size=0.5, random_state=seed, stratify=ynp[tmp]
    )

    def _mask(indices):
        m = torch.zeros(n, dtype=torch.bool)
        m[indices] = True
        return m

    data = Data(
        x=x,
        edge_index=edge_index_list[0],
        y=y,
        train_mask=_mask(train_idx),
        val_mask=_mask(val_idx),
        test_mask=_mask(test_idx),
    )
    return data, edge_index_list

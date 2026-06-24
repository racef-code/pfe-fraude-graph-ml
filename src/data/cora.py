"""Étape 1 : Cora (graphe de citations) pour valider la mécanique GNN."""
from torch_geometric.datasets import Planetoid

_CACHE = "data/cora"


def load_cora():
    dataset = Planetoid(root=_CACHE, name="Cora")
    return dataset[0]

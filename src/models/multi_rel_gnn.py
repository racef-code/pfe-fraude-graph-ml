"""GNN multi-relationnel (style R-GCN allégé) pour YelpChi.

YelpChi a 3 relations (net_rur, net_rtr, net_rsr) de qualités différentes :
net_rur très homophile mais sparse (48 % de nœuds isolés), les autres denses
mais bruitées. Un GNN par relation puis agrégation laisse le modèle exploiter
le signal propre LÀ où il existe + la couverture des relations denses ailleurs.

Implémentation : une SAGEConv par relation et par couche, moyenne sur les
relations. Les edge_index sont stockés en buffers (suivent le device du modèle),
donc le modèle reste compatible avec `train_gnn` (forward(x, edge_index) ignore
le 2e argument et utilise les relations internes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class MultiRelGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index_list, dropout=0.5):
        super().__init__()
        self.num_rel = len(edge_index_list)
        for i, ei in enumerate(edge_index_list):
            self.register_buffer(f"ei_{i}", ei)
        self.convs1 = nn.ModuleList(
            [SAGEConv(in_dim, hidden_dim) for _ in range(self.num_rel)]
        )
        self.convs2 = nn.ModuleList(
            [SAGEConv(hidden_dim, out_dim) for _ in range(self.num_rel)]
        )
        self.dropout = dropout

    def _edges(self):
        return [getattr(self, f"ei_{i}") for i in range(self.num_rel)]

    def forward(self, x, edge_index=None):  # edge_index ignoré (relations internes)
        eis = self._edges()
        h = sum(F.relu(c(x, ei)) for c, ei in zip(self.convs1, eis)) / self.num_rel
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = sum(c(h, ei) for c, ei in zip(self.convs2, eis)) / self.num_rel
        return out  # logits

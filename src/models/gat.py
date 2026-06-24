"""GAT 2 couches (Graph Attention Network).

Contrairement à GraphSAGE qui moyenne uniformément les voisins, GAT apprend
des poids d'attention par arête → peut downweighter les voisins camouflés
(non-fraudeurs autour d'un fraudeur). Cible directe de la faiblesse de
GraphSAGE sur les relations à faible homophilie. Standard PyG (GATConv).
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        # couche 1 : multi-têtes concaténées -> hidden_dim * heads
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        # couche 2 : 1 tête, moyenne -> out_dim
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False,
                             dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits

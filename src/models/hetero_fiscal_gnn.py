"""GNN hétérogène pour le futur graphe fiscal IS.

Le modèle prédit uniquement les nœuds ``company`` mais agrège les messages des
relations hétérogènes (dirigeant, adresse, comptable, secteur, transactions).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class FiscalHeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_dim: int = 32, out_dim: int = 2, dropout: float = 0.4):
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout

        # Lazy input dimensions (-1, -1) allow different feature dimensions per node type.
        self.conv1 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in edge_types},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in edge_types},
            aggr="sum",
        )
        self.company_head = nn.Linear(hidden_dim, out_dim)
        self.node_types = node_types

    def forward(self, x_dict, edge_index_dict):
        h = self.conv1(x_dict, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h_company = h.get("company")
        if h_company is None:
            raise ValueError("Hetero graph produced no 'company' embeddings; check reverse/company-dst edges")
        return self.company_head(h_company)

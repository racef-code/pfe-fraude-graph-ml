"""Relation-gated heterogeneous GNN for fiscal graphs.

Upgrade over `FiscalHeteroGNN`: instead of summing all relation messages equally,
this model learns one scalar gate per edge type. That matters for fraud graphs
where some relations are informative (transactions, accountant, address) while
others can be noisy or sparse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def _etype_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


class RelationGatedFiscalGNN(nn.Module):
    def __init__(self, metadata, hidden_dim: int = 32, out_dim: int = 2, dropout: float = 0.4):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.dropout = dropout
        self.edge_type_by_key = {_etype_key(et): et for et in self.edge_types}

        self.conv1 = nn.ModuleDict({
            _etype_key(et): SAGEConv((-1, -1), hidden_dim) for et in self.edge_types
        })
        self.conv2 = nn.ModuleDict({
            _etype_key(et): SAGEConv((-1, -1), hidden_dim) for et in self.edge_types
        })
        # One learnable scalar per relation per layer. Softmaxed per destination type.
        self.gate1 = nn.ParameterDict({_etype_key(et): nn.Parameter(torch.zeros(())) for et in self.edge_types})
        self.gate2 = nn.ParameterDict({_etype_key(et): nn.Parameter(torch.zeros(())) for et in self.edge_types})
        self.company_head = nn.Linear(hidden_dim, out_dim)

    def _layer(self, x_dict, edge_index_dict, convs: nn.ModuleDict, gates: nn.ParameterDict):
        incoming: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for et, edge_index in edge_index_dict.items():
            key = _etype_key(et)
            if key not in convs:
                continue
            src, _, dst = et
            msg = convs[key]((x_dict[src], x_dict[dst]), edge_index)
            incoming.setdefault(dst, []).append((key, msg))

        out = {}
        for dst, parts in incoming.items():
            logits = torch.stack([gates[key] for key, _ in parts])
            weights = torch.softmax(logits, dim=0)
            h = sum(w * msg for w, (_, msg) in zip(weights, parts))
            out[dst] = h
        return out

    def forward(self, x_dict, edge_index_dict):
        h = self._layer(x_dict, edge_index_dict, self.conv1, self.gate1)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self._layer(h, edge_index_dict, self.conv2, self.gate2)
        h_company = h.get("company")
        if h_company is None:
            raise ValueError("RelationGatedFiscalGNN produced no company embeddings")
        return self.company_head(h_company)

    def relation_weights(self, layer: int = 2) -> dict[str, float]:
        """Return learned gate weights grouped by destination node type.

        Call after at least one forward pass. Useful for model interpretation.
        """
        gates = self.gate2 if layer == 2 else self.gate1
        by_dst: dict[str, list[str]] = {}
        for key, et in self.edge_type_by_key.items():
            by_dst.setdefault(et[2], []).append(key)
        weights = {}
        for dst, keys in by_dst.items():
            logits = torch.stack([gates[k].detach().cpu() for k in keys])
            probs = torch.softmax(logits, dim=0).tolist()
            for k, p in zip(keys, probs):
                weights[k.replace("__", "->")] = float(p)
        return weights

"""Construction d'un graphe fiscal hétérogène pour le cas IS / entreprises.

Le vrai dataset fiscal n'est pas encore disponible, mais ce module fixe le
contrat propre : tables tabulaires -> ``torch_geometric.data.HeteroData``.

Nœud cible du modèle : ``company`` (entreprise soumise à l'IS). Les autres
nœuds (person, address, accountant, sector, ...) servent de contexte relationnel.
Les labels/masks sont portés uniquement par ``company``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

# Relations métier attendues pour une fraude IS orientée entreprises.
RELATION_TYPES = [
    "transaction",          # company -> company : client/fournisseur/facturation
    "has_director",         # company -> person : dirigeant / gérant
    "registered_at",        # company -> address : adresse déclarée
    "uses_accountant",      # company -> accountant : fiduciaire / comptable
    "in_sector",            # company -> sector : secteur d'activité
    "participates_in",      # company -> company : participation / contrôle
]

# Types de nœuds utiles pour le PFE IS. On garde des noms anglais parce que PyG
# les affiche dans les clés HeteroData et c'est plus standard dans le code.
NODE_TYPES = ["company", "person", "address", "accountant", "sector"]

# Colonnes numériques IS plausibles. Les vraies données pourront en ajouter ;
# build_fiscal_graph détecte automatiquement les colonnes numériques si
# feature_cols=None.
DEFAULT_COMPANY_FEATURES = [
    "ca",                   # chiffre d'affaires
    "resultat",             # résultat fiscal/comptable
    "impot_is",             # IS déclaré/payé
    "marge",                # résultat / CA
    "taux_is_effectif",     # IS / résultat imposable approx.
    "age_entreprise",       # ancienneté
    "retards_declaration",  # retards déclaratifs
    "rectifications",       # historique de rectifications
    "variation_ca",         # variation annuelle du CA
    "ratio_charges",        # charges / CA
]

ID_COL = "id"
TYPE_COL = "type"
LABEL_COL = "label"


@dataclass(frozen=True)
class FiscalGraphMetadata:
    """Métadonnées utiles pour relier indices PyG et identifiants originaux."""

    node_maps: dict[str, dict[object, int]]
    feature_cols: dict[str, list[str]]
    target_node_type: str = "company"


def _numeric_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    exclude = set(exclude)
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _frame_to_x(df: pd.DataFrame, cols: list[str]) -> torch.Tensor:
    if not cols:
        # PyG HeteroConv needs every node type to have at least one feature.
        return torch.ones((len(df), 1), dtype=torch.float)
    arr = df[cols].copy()
    arr = arr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float)


def _make_masks(y: pd.Series, seed: int, train_size: float, val_size: float):
    """Crée des masks train/val/test sur les entreprises labellisées uniquement.

    Les labels NaN restent hors masks : dans un vrai contexte fiscal, non-enquêté
    signifie inconnu, pas non-fraudeur.
    """
    known = y.notna().to_numpy()
    known_idx = np.flatnonzero(known)
    y_known = y.iloc[known_idx].astype(int).to_numpy()

    n = len(y)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    if len(known_idx) == 0:
        return train_mask, val_mask, test_mask
    if len(np.unique(y_known)) < 2 or len(known_idx) < 5:
        # Trop petit pour stratifier proprement : tout en train, utile pour tests/unitaires.
        train_mask[known_idx] = True
        return train_mask, val_mask, test_mask

    tmp_size = 1.0 - train_size
    train_idx, tmp_idx, y_train, y_tmp = train_test_split(
        known_idx,
        y_known,
        test_size=tmp_size,
        random_state=seed,
        stratify=y_known,
    )
    # val_size est exprimé sur le total ; on le convertit dans le sous-ensemble tmp.
    val_fraction_of_tmp = val_size / tmp_size
    val_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=1.0 - val_fraction_of_tmp,
        random_state=seed,
        stratify=y_tmp,
    )
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def build_fiscal_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    feature_cols: dict[str, list[str]] | None = None,
    seed: int = 42,
    train_size: float = 0.6,
    val_size: float = 0.2,
    add_reverse_edges: bool = True,
) -> HeteroData:
    """Construit un ``HeteroData`` fiscal depuis deux tables.

    Parameters
    ----------
    nodes_df:
        Table de nœuds avec au minimum ``id`` et ``type``. Les entreprises
        (``type == 'company'``) peuvent porter ``label`` : 1 fraudeur, 0 non
        fraudeur, NaN inconnu/non-enquêté.
    edges_df:
        Table d'arêtes avec ``id_source``, ``id_cible``, ``type_relation``.
        Les types source/destination sont inférés depuis ``nodes_df``.
    feature_cols:
        Optionnel, mapping type_nœud -> colonnes numériques à utiliser. Si absent,
        toutes les colonnes numériques hors id/label sont utilisées par type.
    seed/train_size/val_size:
        Paramètres des splits stratifiés sur les entreprises labellisées.
    add_reverse_edges:
        Ajoute ``rev_<relation>`` pour permettre au nœud cible ``company`` de
        recevoir les messages depuis person/address/accountant/sector.
    """
    required_nodes = {ID_COL, TYPE_COL}
    required_edges = {"id_source", "id_cible", "type_relation"}
    missing_nodes = required_nodes - set(nodes_df.columns)
    missing_edges = required_edges - set(edges_df.columns)
    if missing_nodes:
        raise ValueError(f"nodes_df missing columns: {sorted(missing_nodes)}")
    if missing_edges:
        raise ValueError(f"edges_df missing columns: {sorted(missing_edges)}")

    data = HeteroData()
    node_maps: dict[str, dict[object, int]] = {}
    used_features: dict[str, list[str]] = {}

    for ntype, group in nodes_df.groupby(TYPE_COL, sort=False):
        group = group.reset_index(drop=True)
        ids = group[ID_COL].tolist()
        node_maps[str(ntype)] = {node_id: i for i, node_id in enumerate(ids)}

        cols = (feature_cols or {}).get(str(ntype))
        if cols is None:
            cols = _numeric_feature_columns(group, exclude=[ID_COL, LABEL_COL])
        used_features[str(ntype)] = list(cols)
        data[str(ntype)].x = _frame_to_x(group, list(cols))
        data[str(ntype)].node_id = ids

        if str(ntype) == "company":
            labels = group[LABEL_COL] if LABEL_COL in group.columns else pd.Series([np.nan] * len(group))
            y = labels.fillna(-1).astype(int).to_numpy()
            data["company"].y = torch.tensor(y, dtype=torch.long)
            train_mask, val_mask, test_mask = _make_masks(labels, seed, train_size, val_size)
            data["company"].train_mask = train_mask
            data["company"].val_mask = val_mask
            data["company"].test_mask = test_mask
            data["company"].labeled_mask = train_mask | val_mask | test_mask

    id_to_type = dict(zip(nodes_df[ID_COL], nodes_df[TYPE_COL]))
    edge_buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for row in edges_df.itertuples(index=False):
        src_id = getattr(row, "id_source")
        dst_id = getattr(row, "id_cible")
        rel = str(getattr(row, "type_relation"))
        if src_id not in id_to_type or dst_id not in id_to_type:
            # Ignore les arêtes orphelines plutôt que casser toute la construction.
            continue
        src_t = str(id_to_type[src_id])
        dst_t = str(id_to_type[dst_id])
        if src_t not in node_maps or dst_t not in node_maps:
            continue
        key = (src_t, rel, dst_t)
        edge_buckets.setdefault(key, []).append((node_maps[src_t][src_id], node_maps[dst_t][dst_id]))
        if add_reverse_edges:
            rev_key = (dst_t, f"rev_{rel}", src_t)
            edge_buckets.setdefault(rev_key, []).append((node_maps[dst_t][dst_id], node_maps[src_t][src_id]))

    for key, pairs in edge_buckets.items():
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[key].edge_index = edge_index

    data.metadata_obj = FiscalGraphMetadata(node_maps=node_maps, feature_cols=used_features)
    return data


def make_synthetic_is_fiscal_tables(
    n_companies: int = 240,
    fraud_rate: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Génère un banc d'essai synthétique IS orienté entreprises.

    Ce n'est PAS un substitut scientifique aux données fiscales réelles. Son rôle
    est de tester le pipeline hétérogène avant réception des données : labels
    déséquilibrés, features comportementales, et relations dirigeant/adresse/
    comptable/transactions qui créent des communautés à risque.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(n_companies * fraud_rate))
    labels = np.zeros(n_companies, dtype=int)
    fraud_idx = rng.choice(n_companies, size=n_fraud, replace=False)
    labels[fraud_idx] = 1

    sectors = [f"sector_{i}" for i in range(8)]
    accountants = [f"accountant_{i}" for i in range(max(6, n_companies // 20))]
    addresses = [f"address_{i}" for i in range(max(12, n_companies // 10))]
    persons = [f"person_{i}" for i in range(max(30, n_companies // 3))]

    rows = []
    for i in range(n_companies):
        fraud = labels[i]
        ca = rng.lognormal(mean=12.0, sigma=0.8)
        # Fraude IS stylisée mais volontairement NON triviale : le signal tabulaire
        # existe (retards, marge, taux IS, variations), mais il est bruité pour
        # que le graphe puisse réellement apporter de l'information.
        marge = rng.normal(0.09 if fraud else 0.13, 0.07)
        resultat = ca * marge
        taux_is = max(0.0, rng.normal(0.16 if fraud else 0.22, 0.08))
        rows.append({
            "id": f"company_{i}",
            "type": "company",
            "ca": ca,
            "resultat": resultat,
            "impot_is": max(0.0, resultat * taux_is),
            "marge": marge,
            "taux_is_effectif": taux_is,
            "age_entreprise": rng.integers(1, 25),
            "retards_declaration": rng.poisson(1.4 if fraud else 0.7),
            "rectifications": rng.poisson(0.8 if fraud else 0.35),
            "variation_ca": rng.normal(0.20 if fraud else 0.10, 0.30),
            "ratio_charges": np.clip(rng.normal(0.82 if fraud else 0.74, 0.16), 0, 1.5),
            "label": fraud,
        })
    for p in persons:
        rows.append({"id": p, "type": "person", "risk_prior": rng.random()})
    for a in addresses:
        rows.append({"id": a, "type": "address", "risk_prior": rng.random()})
    for a in accountants:
        rows.append({"id": a, "type": "accountant", "risk_prior": rng.random()})
    for s in sectors:
        rows.append({"id": s, "type": "sector", "risk_prior": rng.random()})
    nodes = pd.DataFrame(rows)

    # Pools partagés pour donner aux fraudeurs des relations plus homophiles.
    risky_persons = persons[: max(4, len(persons) // 6)]
    risky_addresses = addresses[: max(3, len(addresses) // 5)]
    risky_accountants = accountants[: max(2, len(accountants) // 4)]

    edges = []
    for i in range(n_companies):
        c = f"company_{i}"
        fraud = labels[i] == 1
        person_pool = risky_persons if fraud and rng.random() < 0.75 else persons
        address_pool = risky_addresses if fraud and rng.random() < 0.70 else addresses
        accountant_pool = risky_accountants if fraud and rng.random() < 0.65 else accountants
        edges += [
            {"id_source": c, "id_cible": rng.choice(person_pool), "type_relation": "has_director"},
            {"id_source": c, "id_cible": rng.choice(address_pool), "type_relation": "registered_at"},
            {"id_source": c, "id_cible": rng.choice(accountant_pool), "type_relation": "uses_accountant"},
            {"id_source": c, "id_cible": rng.choice(sectors), "type_relation": "in_sector"},
        ]
        # Transactions : les fraudeurs échangent plus souvent entre eux, mais pas parfaitement.
        for _ in range(rng.poisson(3 if fraud else 2) + 1):
            if fraud and rng.random() < 0.55:
                dst = int(rng.choice(fraud_idx))
            else:
                dst = int(rng.integers(0, n_companies))
            if dst != i:
                edges.append({"id_source": c, "id_cible": f"company_{dst}", "type_relation": "transaction"})
    return nodes, pd.DataFrame(edges)

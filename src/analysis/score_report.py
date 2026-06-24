"""Produit un classement lisible des entreprises par score de risque de fraude.

Relie chaque score (probabilité de fraude) à l'id de l'entreprise via
``data.metadata_obj.node_maps['company']`` et trie du plus risqué au moins risqué.
C'est la sortie de TRIAGE : un humain enquête le haut de la liste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.train.train_hetero import predict_hetero_company_scores


def build_company_score_report(model, data) -> pd.DataFrame:
    """Renvoie un DataFrame trié : rang, company_id, risk_score, risk_pct, split, label_connu."""
    scores = predict_hetero_company_scores(model, data)  # ordre = index des nœuds company
    node_map = data.metadata_obj.node_maps["company"]      # {company_id: index}
    idx_to_id = {i: cid for cid, i in node_map.items()}

    company = data["company"]
    n = len(scores)
    y = company.y.cpu().numpy() if hasattr(company, "y") else np.full(n, -1)
    train = company.train_mask.cpu().numpy()
    val = company.val_mask.cpu().numpy()
    test = company.test_mask.cpu().numpy()

    def split(i: int) -> str:
        if train[i]:
            return "train"
        if val[i]:
            return "val"
        if test[i]:
            return "test"
        return "non_enquete"  # pas de label connu

    def label(i: int):
        v = int(y[i])
        return v if v in (0, 1) else None  # -1 / NaN => inconnu

    rows = [
        {
            "company_id": idx_to_id[i],
            "risk_score": round(float(scores[i]), 4),
            "risk_pct": f"{100 * scores[i]:.1f}%",
            "split": split(i),
            "label_connu": label(i),
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rang", df.index + 1)
    return df

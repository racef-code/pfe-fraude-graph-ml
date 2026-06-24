"""Baseline XGBoost : mêmes features (data.x), AUCUNE info de graphe.
Point de comparaison 'sans graphe' pour la question de recherche."""
import numpy as np
from xgboost import XGBClassifier


def train_xgb(X_train, y_train, scale_pos_weight: float = 1.0):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # déséquilibre (piège n°2)
        eval_metric="aucpr",
        n_jobs=-1,
    )
    model.fit(np.asarray(X_train), np.asarray(y_train))
    return model


def predict_xgb(model, X):
    return model.predict_proba(np.asarray(X))[:, 1]

import numpy as np
from src.train.baseline_xgb import train_xgb, predict_xgb


def test_xgb_learns_separable_data():
    rng = np.random.default_rng(0)
    X_pos = rng.normal(2, 0.5, (50, 4))
    X_neg = rng.normal(-2, 0.5, (50, 4))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)
    model = train_xgb(X, y, scale_pos_weight=1.0)
    scores = predict_xgb(model, X)
    assert scores.shape == (100,)
    assert ((scores >= 0) & (scores <= 1)).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, scores) > 0.99

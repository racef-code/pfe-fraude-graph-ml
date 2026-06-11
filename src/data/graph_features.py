"""Features de graphe pour enrichir un modèle tabulaire (XGBoost+graph).

But : tester si l'info de structure aide un modèle SANS message-passing.
Deux features par nœud :
  - log(1 + degré)
  - ratio de fraude des voisins, calculé UNIQUEMENT à partir des labels
    d'entraînement (anti-fuite : les labels val/test n'influencent jamais
    la feature, sinon résultats gonflés et invalides).
"""
import numpy as np
import scipy.sparse as sp
import torch


def compute_graph_features(edge_index, y, train_mask, num_nodes: int) -> torch.Tensor:
    """Renvoie un tenseur [num_nodes, 2] : [log1p(degré), ratio_fraude_voisins_train].

    Le ratio de fraude des voisins n'utilise que les labels des nœuds
    d'entraînement. Pour un nœud sans voisin d'entraînement, on retombe sur
    le taux de fraude global du train (pas de fuite val/test).
    """
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    A = sp.coo_matrix(
        (np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes)
    ).tocsr()
    A.data[:] = 1.0  # binariser (au cas où des arêtes seraient dupliquées)

    deg = np.asarray(A.sum(axis=1)).ravel()

    ynp = y.cpu().numpy().astype(float)
    tm = train_mask.cpu().numpy().astype(float)
    train_pos = ynp * tm          # label de fraude seulement sur le train
    train_known = tm              # indicateur d'appartenance au train

    nb_pos = np.asarray(A @ train_pos).ravel()      # somme labels voisins-train
    nb_known = np.asarray(A @ train_known).ravel()  # nb de voisins-train
    global_rate = train_pos.sum() / max(train_known.sum(), 1.0)
    ratio = np.where(nb_known > 0, nb_pos / np.maximum(nb_known, 1.0), global_rate)

    feats = np.stack([np.log1p(deg), ratio], axis=1)
    return torch.tensor(feats, dtype=torch.float)

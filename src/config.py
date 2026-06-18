import random
from dataclasses import dataclass
import numpy as np
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    lr: float = 0.01
    epochs: int = 200
    weight_decay: float = 5e-4
    dropout: float = 0.5
    patience: int = 30  # early-stop on validation metric/loss
    early_stop_metric: str = "loss"  # "loss" or "auc_pr"

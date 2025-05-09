import os
import numpy as np

from .game import FreckersGame
from .eval_func import *

class XGModel():
    def __init__(self, game: FreckersGame):
        self.model = None

    def predict(self, board: np.ndarray, player: int = 1) -> float:
        player_features = compute_features(board, player)
        opp_features = compute_features(board, -player)
        features = np.concatenate([player_features, opp_features]).reshape(1, -1)
        return self.model.predict_proba(features)[0][1]

import os
import numpy as np
import joblib

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

    def load_model(self, folder='checkpoint', filename='checkpoint.pkl'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

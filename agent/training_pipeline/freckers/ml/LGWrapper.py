import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')

from utils import *
from NeuralNet import NeuralNet

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.exceptions import NotFittedError

from freckers.FreckersGame import FreckersGame
from freckers.ml.eval_func import *
import joblib

class LGWrapper():
    def __init__(self, game: FreckersGame):
        self.model = LogisticRegression(max_iter=5000)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.is_trained = False

    def train(self, examples):
        """
        examples: list of examples, each is (board, pi, v)
        board: np.ndarray (8x8), pi: policy (unused here), v: value (-1, 0, 1)
        Only v=1 (win) is mapped to 1, and all else (draw/loss) is mapped to 0.
        """
        X_all = []
        y_all = []

        print("Extracting features from examples...")
        for board, _, v in examples:
            player_features = compute_features(board, player_color=1)  # assuming Red perspective
            opp_features = compute_features(board, player_color=-1)
            features = np.concatenate([player_features, opp_features])
            X_all.append(features)
            y_all.append(1 if v == 1 else 0)  # binary: 1 = win, 0 = not win

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        print(f"Training model on full dataset of length {len(y_all)}...")
        self.model.fit(X_all, y_all)
        pred = self.model.predict_proba(X_all)
        target = np.zeros((len(y_all), 2))
        target[np.arange(len(y_all)), y_all] = 1
        loss = log_loss(target, pred)
        print(f"Final log loss: {loss:.4f}")

        self.is_trained = True

    def predict(self, board: np.ndarray, player: int=1) -> float:
        player_features = compute_features(board, player)
        opp_features = compute_features(board, -player)

        if not self.is_trained: # load_model = False
            return None # sigmoid_func(eval_func(player_features) - eval_func(opp_features))
        try:
            features = np.concatenate([player_features, opp_features]).reshape(1, -1)
            return self.model.predict_proba(features)[0][1]  # prob curr player win 
        except NotFittedError: # load_model = True but can't find folder 
            return None # sigmoid_eval_func(features.flatten()) 

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pkl'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pkl'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

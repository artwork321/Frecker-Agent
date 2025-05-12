import os
import sys
import csv

import numpy as np

sys.path.append('../../')

from utils import *

from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from freckers.FreckersGame import FreckersGame
from freckers.ml.eval_func import *
import joblib

from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

class XGWrapper():
    def __init__(self, game: FreckersGame):
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            reg_alpha=1.0
        )

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.is_trained = False
        self.train_metrics = []
        self.test_metrics = []
        self.train_size = 0

    def train(self, examples):
        """
        examples: list of examples, each is (board, pi, v)
        board: np.ndarray (8x8), pi: policy (unused here), v: value (-1, 0, 1)
        Only v=1 (win) is mapped to 1, and all else (draw/loss) is mapped to 0.
        """
        X_all = []
        y_all = []

        print("Extracting features from examples...")
        for board, v in examples:
            features = compute_features(board, player_color=1)
            X_all.append(features)
            y_all.append(1 if v == 1 else 0)

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
        )

        self.train_size = len(y_train)
        print(f"Training XGBoost model on {self.train_size } training examples...")
        self.model.fit(X_train, y_train)

        print("\nEvaluation:")
        self.evaluate_model(X_train, y_train, "Train")
        self.evaluate_model(X_test, y_test, "Test")

        self.save_model_analysis(X_train, plot_shap=True)

        self.is_trained = True

    def evaluate_model(self, X, y, label):
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        target = np.zeros((len(y), 2))
        target[np.arange(len(y)), y] = 1
        logloss = log_loss(target, probs)
        accuracy = accuracy_score(y, preds)
        brier = brier_score_loss(y, probs[:, 1])

        print(f"{label} log loss:       {logloss:.4f}")
        print(f"{label} accuracy:       {accuracy:.4f}")
        print(f"{label} Brier score:     {brier:.4f}")

        if label == "Train":
            self.train_metrics = [logloss, accuracy, brier]
        else:
            self.test_metrics = [logloss, accuracy, brier]

    def save_model_analysis(self, X_train, plot_shap=False):
        os.makedirs('temp_xg9', exist_ok=True)

        # Generate unique filename
        i = 0
        while True:
            importance_path = f'temp_xg9/feature_importances_{i}.csv'
            if not os.path.exists(importance_path):
                break
            i += 1

        # Zip and sort the feature importances
        feat_imp = list(zip(FEATURE_NAMES, self.model.feature_importances_))
        feat_imp.sort(key=lambda x: x[1], reverse=True)

        # Write to CSV
        with open(importance_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature_name', 'importance'])  # header
            writer.writerows(feat_imp)

        print(f"Feature importances saved to {importance_path}")

        if plot_shap:
            # Save SHAP summary plot with unique filename
            i = 0
            while True:
                shap_path = f'temp_xg5/shap_summary_{i}.png'
                if not os.path.exists(shap_path):
                    break
                i += 1
            explainer = shap.Explainer(self.model, X_train)
            shap_values = explainer(X_train)
            plt.figure()
            shap.summary_plot(shap_values, X_train, feature_names=FEATURE_NAMES, show=False)
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to {shap_path}")

    def predict(self, board: np.ndarray, player: int = 1) -> float:
        if not self.is_trained:
            return None
        try:
            features = compute_features(board, player).reshape(1, -1)
            return self.model.predict_proba(features)[0][1]
        except NotFittedError:
            return None

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

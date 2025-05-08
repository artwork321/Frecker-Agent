# import os
# import sys

# import numpy as np

# # sys.path.append('../../')

# from agent.ml.utils import *
# from agent.ml.eval_func import *

# import joblib

# from xgboost import XGBClassifier
# import shap
# import matplotlib.pyplot as plt

# FEATURE_NAMES = [
#     "avg row dist to goal",            # 0
#     "jumpable ratio",                  # 1
#     "spread variance",                     # 2
#     "inverse spread dist",             # 3
#     "interaction score",              # 4
#     "blocked dir ratio",             # 5
#     "edge position ratio",                 # 6
#     "col centrality score",              # 7
#     "assistable ratio",                # 8
#     "near goal ratio",                 # 9
#     "upper half pad coverage",       # 10
#     "avg reachable pads",              # 11
#     "grow needed ratio",             # 12
#     "target jump ratio",                   # 13
#     "target blocked ratio",                # 14
#     "sideway move ratio",                  # 15
#     "sideway jump ratio",                  # 16
#     "#frogs at goal row",         # 17
#     "goal pad ratio",                      # 18
#     "avg min euclid dist to goal"    # 19
# ]

# class XGWrapper():
#     def __init__(self):
#         self.model = XGBClassifier(
#             use_label_encoder=False,
#             eval_metric='logloss',
#             n_estimators=200,
#             learning_rate=0.1,
#             max_depth=8
#         )

#     def predict(self, board: np.ndarray, player: int = 1) -> float:
#         player_features = compute_features(board, player)
#         opp_features = compute_features(board, -player)

#         if not self.is_trained:
#             return None
#         try:
#             features = np.concatenate([player_features, opp_features]).reshape(1, -1)
#             return self.model.predict_proba(features)[0][1]
#         except NotFittedError:
#             return None

#     def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pkl'):
#         filepath = os.path.join(folder, filename)
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"No model in path {filepath}")
#         self.model = joblib.load(filepath)
#         self.is_trained = True
#         print(f"Model loaded from {filepath}")

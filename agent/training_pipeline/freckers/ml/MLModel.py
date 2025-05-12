import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple

class MLModel():
    def __init__(self, max_iter=1000):
        self.model = LogisticRegression()
        self.max_iter = max_iter

    def train_model(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        model = LogisticRegression(max_iter=self.max_iter)
        model.fit(X, y)
        return model
    

    
    
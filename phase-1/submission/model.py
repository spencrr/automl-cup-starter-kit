# This is a sample code submission.
# It is a simple machine learning classifier.

import numpy as np
import torch
from datasets import Dataset
from sklearn.tree import DecisionTreeClassifier


class Model:
    def __init__(self):
        """<ADD DOCUMENTATION HERE>"""
        self.classifier = DecisionTreeClassifier()

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Train the model.

        Args:
            X: Training data matrix of shape (num-samples, num-features), type np.ndarray.
            y: Training label vector of shape (num-samples), type np.ndarray.
        """
        print(f"FROM MODEL.PY: TRAIN {train_dataset}, VAL {val_dataset}")
        # print(f"X has shape {X.shape} and is:\n{X}")
        # print(f"y has shape {y.shape} and is:\n{y}")
        # self.classifier.fit(X, y)

    def predict(self, dataset: Dataset):
        """Predict labels.

        Args:
          X: Data matrix of shape (num-samples, num-features) to pass to the model for inference, type np.ndarray.
        """
        print(f"FROM MODEL.PY: TEST {dataset}")
        # y = self.classifier.predict(X)
        print(len(dataset), np.zeros(len(dataset)).shape)
        return np.zeros(len(dataset))

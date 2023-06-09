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

    def train(self, train_dataset: Dataset):
        """Train the model.

        Args:
            train_dataset: Training data, type datasets.Dataset.
        """
        print(f"FROM MODEL.PY: TRAIN {train_dataset}")

        try:
            X = np.array(train_dataset["input"])
        except ValueError:
            max_len = max(map(len, train_dataset["input"]))
            X = np.zeros((len(train_dataset), max_len), dtype=np.float32)
            for idx, row in enumerate(train_dataset["input"]):
                X[idx, : len(row)] = row

        y = np.array(train_dataset["label"])

        print(f"FROM MODEL.PY: X SHAPE {X.shape}")
        print(f"FROM MODEL.PY: Y SHAPE {y.shape}")

        # self.classifier.fit(X, y)

    def predict(self, dataset: Dataset):
        """Predict labels.

        Args:
            test_dataset: Testing data, type datasets.Dataset.
        """
        print(f"FROM MODEL.PY: TEST {dataset}")
        # y = self.classifier.predict(X)
        print(len(dataset), np.zeros(len(dataset)).shape)
        return np.zeros(len(dataset))

import numpy as np
from abc import ABC, abstractmethod

from models import *
from utils import find_k_nearest_neighbors
from numpy_array_evaluator import evaluate_all_elements


class Model(ABC):
    @abstractmethod
    def fit(self, train_x, train_y):
        pass

    @abstractmethod
    def predict(self, new_instance):
        pass


class KNearestNeighbours(Model):
    """K-Nearest Neighbors classifier implementation."""

    variants = ['regular', 'test']  # Class attribute for different variants

    def __init__(self):
        """
        Initialize KNearestNeighbours model.
        """
        self.root = None
        self.d = None

    def fit(self, train_x, train_y):
        """
        Fit the KNearestNeighbours model to the training data.

        Parameters:
        train_x (numpy.ndarray): The training data.
        train_y (numpy.ndarray): The target labels.
        """
        if len(train_x) != len(train_y):
            raise ValueError("Amount of elements in train_x and train_y must be the same.")

        if isinstance(train_x, list):
            train_x = np.array(train_x)

        if isinstance(train_y, list):
            train_y = np.array(train_y)

        elements = [Element(Point(x), y) for x, y in zip(train_x, train_y)]

        self.d = len(train_x[0])
        self.root = Node(elements, 0, self.d)

    def predict(self, test_x, k=3, variant="regular"):
        """
        Predict the class labels for test data using KNearestNeighbours.

        Parameters:
        test_x (numpy.ndarray): The test data.
        k (int): The number of neighbors to consider (default=3).
        variant (str): The type of prediction to perform (default='regular').

        Returns:
        int or numpy.ndarray: Predicted class labels.
        """
        if self.root is None:
            raise ValueError("Model must be trained before.")

        if not isinstance(test_x, np.ndarray):
            raise ValueError("test_x must be a numpy array.")

        if variant not in KNearestNeighbours.variants:
            raise ValueError("Unknown variant.")

        return evaluate_all_elements(self._predict_single_instance_wrapper, test_x, (k, variant))

    def _predict_single_instance_wrapper(self, instance, args):
        """
        Wrapper function for predicting a single instance.

        Parameters:
        instance (numpy.ndarray): Single instance to predict.
        args (tuple): Arguments required for prediction.
        """
        return self._predict_single_instance(instance, *args)

    def _predict_single_instance(self, instance, k, variant):
        """
        Predict the class label for a single instance.

        Parameters:
        instance (numpy.ndarray): Single instance to predict.
        k (int): The number of neighbors to consider.
        variant (str): The type of prediction to perform.

        Returns:
        int: Predicted class label.
        """
        if len(instance) != self.d:
            raise ValueError("Dimensionality mismatch.")

        k_nearest = find_k_nearest_neighbors(self.root, Point(instance), k, self.d)

        if variant == "regular":

            counter = {}
            maximum = 0
            predicted_class = None

            for _, elem_class in k_nearest:
                if elem_class in counter:
                    counter[elem_class] += 1
                else:
                    counter[elem_class] = 1

            for elem_class, count in counter.items():
                if count > maximum:
                    maximum = count
                    predicted_class = elem_class
            return predicted_class

        elif variant == "test":
            return 0

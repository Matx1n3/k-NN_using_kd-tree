# K-Nearest Neighbors (KNN) Classifier using a KD-Tree

This repository contains a Python implementation of the K-Nearest Neighbors (KNN) algorithm. The KNearestNeighbours class provides a flexible KNN classifier equipped with a k-d tree for efficient nearest neighbor search.

## Overview

The `KNearestNeighbours` class implements a KNN classifier, leveraging a k-d tree for faster nearest neighbor searches. It offers methods for training (`fit`) and making predictions (`predict`) on datasets.

## Features

- **KNearestNeighbours class**: Implements KNN classification.
- **K-d tree**: Utilizes a k-d tree for efficient nearest neighbor search.
- **Training and prediction**: Provides methods for model training and prediction.
- **Hyperparameter tunning**: Allows searching for the most optimal k

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Matx1n3/k-NN_using_kd-tree.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```python
# Example usage of the KNearestNeighbours class and accuracy_score function
from knn_classifier import KNearestNeighbours
from metrics import accuracy_score

# Instantiate the KNN classifier
model = KNearestNeighbours()

# Training
model.fit(train_x, train_y)

# Prediction
predictions = model.predict(test_x, k=5)

# Accuracy
print("Accuracy score:", accuracy_score(prediction, test_y))

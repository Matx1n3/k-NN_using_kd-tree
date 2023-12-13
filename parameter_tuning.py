import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from numpy_array_evaluator import evaluate_all_elements
from metrics import accuracy_score


def choose_k(knn_model, validate_x, validate_y, k_array, args):
    results = evaluate_all_elements(predict_wrapper, k_array, (validate_x, knn_model, ) + args)
    accuracies = evaluate_all_elements(accuracy_score_wrapper, results, (validate_y, ))
    return int(k_array[np.argmax(accuracies)])


def accuracy_score_wrapper(y_pred, args):
    return accuracy_score(y_pred, *args)


def predict_wrapper(k, args):
    val_x = args[0]
    model = args[1]
    variant = args[2]
    distance_type = args[3]
    return model.predict(val_x, k=int(k), variant=variant, distance_type=distance_type)

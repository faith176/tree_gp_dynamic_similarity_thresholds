import numpy as np
from sklearn.metrics import accuracy_score

def get_predicted_labels(similarity_arr, threshold_arr):
    return np.where(similarity_arr >= threshold_arr, 1, 0)

def predicted_threshold_accuracy(similarity_arr, threshold_arr, y_test):
    predicted_labels =  get_predicted_labels(similarity_arr, threshold_arr)
    TP = np.sum((predicted_labels == 1) & (y_test == 1))
    TN = np.sum((predicted_labels == 0) & (y_test == 0))
    FP = np.sum((predicted_labels == 1) & (y_test == 0))
    FN = np.sum((predicted_labels == 0) & (y_test == 1))
    accuracy = (TP + TN) / len(y_test)
    return accuracy, TP, TN, FP, FN


def specific_threshold_accuracy(similarity_arr, threshold, y_test):
    threshold_predictions = (np.array(similarity_arr) >= threshold).astype(int)
    return accuracy_score(y_test, threshold_predictions)

def best_possible_threshold(similarity_arr, y_test, increment=0.01):
    similarity_arr = np.array(similarity_arr)
    y_test = np.array(y_test)
    
    thresholds = np.arange(0, 1.01, increment)
    accuracies = [
        specific_threshold_accuracy(similarity_arr, threshold, y_test)
        for threshold in thresholds
    ]
    best_index = np.argmax(accuracies)
    return thresholds[best_index], accuracies[best_index]


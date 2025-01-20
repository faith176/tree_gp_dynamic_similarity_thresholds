

import numpy as np
from sklearn.model_selection import StratifiedKFold
from Evaluation_Modules.Threshold_Eval import  predicted_threshold_accuracy
from deap import gp
from joblib import Parallel, delayed

def evolved_threshold(adjustment):
    return np.clip(np.round(adjustment,2), 0, 1)

def evaluate_fold(get_action, X_val, Y_val, SIMILARITY_val):
    predicted_thresholds = evolved_threshold(np.apply_along_axis(lambda row: get_action(*row), 1, X_val))
    accuracy, _, _, _, _ = predicted_threshold_accuracy(SIMILARITY_val, predicted_thresholds, Y_val)
    return accuracy

def evaluate_individual(individual, arg_names, X, Y, SIMILARITY, pset):
    k = 5
    individual_str = str(individual)
    get_action = gp.compile(expr=individual, pset=pset)

    # Penalize solutions that don't use any features
    if not any(arg in individual_str for arg in arg_names):
        return (0, 1, 1)

    kf = StratifiedKFold(n_splits=k, shuffle=True)
    fold_fitness = []

    for _, val_index in kf.split(X, Y):
        fitness = evaluate_fold(
            get_action,
            X[val_index],
            Y[val_index],
            SIMILARITY[val_index],
        )
        fold_fitness.append(fitness)


    # Calculate mean metrics and variances
    mean_accuracy = np.mean(fold_fitness)
    
    # Calculate maximum deviation from the mean 
    deviation = np.clip((np.max(np.abs(np.array(fold_fitness)*100 - mean_accuracy*100)))*0.1,0, 1)
    
    # Normalize tree size to same scale as accuracy and consistency, ideally want expression with minimum 2 nodes or maximum 100 nodes with smaller tree being better
    max_tree_size = 100
    min_tree_size = 2
    normalized_tree_size = (len(individual) - min_tree_size) / (max_tree_size - min_tree_size)
    normalized_tree_size = max(0.0, min(1.0, normalized_tree_size))

    # Final fitness values
    final_fitness = (
        mean_accuracy,
        deviation,
        normalized_tree_size
    )

    return final_fitness
import datetime
import os
import numpy as np
from Evaluation_Modules.Threshold_Eval import predicted_threshold_accuracy
from GP_Modules.Fitness_Function import evolved_threshold
from deap import gp
import dill as pickle
        
def evaluate_solution(individual, X_test, Y_test, SIMILARITY_test, pset):
    best_func = gp.compile(expr=individual, pset=pset)
    test_threshold_predictions = [evolved_threshold(best_func(*x)) for x in X_test]
    
    # Metrics
    test_acc, TP, TN, FP, FN = predicted_threshold_accuracy(SIMILARITY_test, test_threshold_predictions, Y_test)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    evaluation_results = {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "length": len(individual),
        "expression": str(individual),
        "max_threshold": np.max(test_threshold_predictions),
        "min_threshold": np.min(test_threshold_predictions),
        "mean_threshold": np.mean(test_threshold_predictions),
        "std_threshold": np.std(test_threshold_predictions),
    }

    return evaluation_results

def print_evaluation_results(evaluation_results):
    print("-" * 50)
    print(f"Length: {evaluation_results['length']} | Expression: {evaluation_results['expression']}")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Precision: {evaluation_results['precision']:.4f} | Recall: {evaluation_results['recall']:.4f} | F1-Score: {evaluation_results['f1']:.4f}")
    print(f"Max Threshold: {evaluation_results['max_threshold']:.4f} | Min Threshold: {evaluation_results['min_threshold']:.4f}")
    print(f"Mean Threshold: {evaluation_results['mean_threshold']:.4f} | Std Threshold: {evaluation_results['std_threshold']:.4f}")
    print("-" * 50)
    
    
def top_solutions_with_accuracy_and_length(hof, X_test, Y_test, SIMILARITY_test, pset, amount = 5):
    print(f"Top {amount} Solutions from the Hall of Fame (HOF):")
    print("-" * 50)
    
    for idx, individual in enumerate(hof[:amount]):
        func = gp.compile(expr=individual, pset=pset)
        correct = 0
        total = len(X_test)
        
        for x, y_true, sim in zip(X_test, Y_test, SIMILARITY_test):
            test_threshold_predictions = evolved_threshold(func(*x))
            predicted_labels = 1 if sim >= test_threshold_predictions else 0
            if predicted_labels == y_true:
                correct += 1
        
        accuracy = correct / total
    
        length = len(individual)
        
        print(f"Solution #{idx + 1}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Length: {length}")
        print(f"  Expression: {individual}")
        print("-" * 50)
        
    
        
def save_gp(parameters, pop, log, hof, stats, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    parameters_file = os.path.join(output_dir, "parameters.pkl")
    with open(parameters_file, "wb") as f:
        pickle.dump(parameters, f)

    hof_file = os.path.join(output_dir, "hof_solutions.pkl")
    with open(hof_file, "wb") as f:
        pickle.dump(hof, f)

    all_solutions_file = os.path.join(output_dir, "all_solutions.pkl")
    with open(all_solutions_file, "wb") as f:
        pickle.dump(pop, f)

    log_file = os.path.join(output_dir, "logbook.pkl")
    stats_file = os.path.join(output_dir, "stats.pkl")
    with open(log_file, "wb") as f:
        pickle.dump(log, f)
    with open(stats_file, "wb") as f:
        pickle.dump(stats, f)

    print(f"All outputs saved to {output_dir}")
    return output_dir
    
    
def load_gp(directory):
    files = {
        "parameters": os.path.join(directory, "parameters.pkl"),
        "hof": os.path.join(directory, "hof_solutions.pkl"),
        "pop": os.path.join(directory, "all_solutions.pkl"),
        "log": os.path.join(directory, "logbook.pkl"),
        "stats": os.path.join(directory, "stats.pkl")
    }

    # Load all files into a dictionary
    results = {}
    for key, file_path in files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found in the directory.")
        with open(file_path, "rb") as f:
            results[key] = pickle.load(f)
            
    parameters = results["parameters"]
    pop = results["pop"]
    log = results["log"]
    hof = results["hof"]
    stats = results["stats"]
    
    print("Loaded GP results successfully.")

    return parameters, pop, log, hof, stats
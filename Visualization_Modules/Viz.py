

from matplotlib import pyplot as plt
from deap import algorithms, base, creator, tools, gp
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from deap import algorithms, base, creator, tools, gp
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from GP_Modules.Fitness_Function import evolved_threshold
from Evaluation_Modules.Threshold_Eval import best_possible_threshold, get_predicted_labels, predicted_threshold_accuracy, specific_threshold_accuracy
from mpl_toolkits.mplot3d import Axes3D


def plot_class_distribution(df):
    duplicate_counts = df['is_duplicate'].value_counts()

    # Calculate the percentage of duplicates
    duplicate_percentage = (duplicate_counts[1] / len(df)) * 100
    duplicate_non_percentage = ((len(df) - duplicate_counts[1]) / len(df)) * 100
    
    # Bar plot for duplicate counts
    plt.figure(figsize=(8, 5))
    sns.barplot(x=duplicate_counts.index, y=duplicate_counts.values)
    plt.xticks([0, 1], ['Non-Duplicate', 'Duplicate'])
    plt.title('Distribution of Duplicate vs Non-Duplicate Questions')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()
  
def plot_accuracy_and_variance_dual_axis(logbook):
    gen = logbook.select("gen")
    
    def extract_objective_stats(logbook, stat_key, objective_index):
        stats = logbook.select(stat_key)
        return [stat[objective_index] for stat in stats]

    max_accuracy = extract_objective_stats(logbook, "max", 0)
    avg_accuracy = extract_objective_stats(logbook, "avg", 0)

    max_variance = extract_objective_stats(logbook, "max", 1)
    avg_variance = extract_objective_stats(logbook, "avg", 1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # accuracy
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Accuracy")
    ax1.plot(gen, max_accuracy, label="Max Accuracy", linestyle='-', color="#6495ED")
    ax1.plot(gen, avg_accuracy, label="Avg Accuracy", linestyle='--', color="#F08080")
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.grid(True)

    # variance
    
    ax2 = ax1.twinx()
    ax2.set_yscale('log') 
    ax2.set_ylabel("Accuracy Variance")
    ax2.plot(gen, max_variance, label="Max Variance", linestyle='-', color="#3CB371")
    ax2.plot(gen, avg_variance, label="Avg Variance", linestyle='--', color="#DDA0DD")
    ax2.tick_params(axis="y")
    ax2.legend(loc="upper right", fontsize="small")

    plt.title("Accuracy and Variance of Accuracy Over Generations")
    plt.show()
    
    
def plot_pareto_front(pop):
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    feasible_solutions = [ind for ind in pop if ind not in pareto_front]
    
    # fitness values
    pareto_obj1 = [ind.fitness.values[0] for ind in pareto_front]
    pareto_obj2 = [ind.fitness.values[1] for ind in pareto_front]

    feasible_obj1 = [ind.fitness.values[0] for ind in feasible_solutions]
    feasible_obj2 = [ind.fitness.values[1] for ind in feasible_solutions]
    
    
    plt.figure(figsize=(8, 6))
    plt.scatter(feasible_obj2, feasible_obj1, c='blue', alpha=0.5, label="Solutions")
    plt.scatter(pareto_obj2, pareto_obj1, c='red', label="Pareto Front", edgecolors='black', s=80)
    plt.xlabel("Variance")
    plt.ylabel("Accuracy")
    plt.title("Pareto Front and Solutions")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def visualize_solution(individual):
    nodes, edges, labels = gp.graph(individual)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # Use spring layout with a higher k value for increased spacing
    pos = nx.spring_layout(g, seed=42, k=0.7) 

    nx.draw_networkx_nodes(
        g, 
        pos, 
        node_size=1500, 
        node_color="lightblue", 
        alpha=0.9
    )

    nx.draw_networkx_edges(
        g, 
        pos, 
        edge_color="gray", 
        width=1.5, 
        alpha=0.8
    )

    nx.draw_networkx_labels(
        g, 
        pos, 
        labels=labels, 
        font_size=10,
        font_color="black", 
        font_weight="regular"
    )

    plt.axis("off")
    plt.show()   
    
    
def visualize_confusion_matrix(individual, X_test, Y_test, SIMILARITY_test, pset):
    best_func = gp.compile(expr=individual, pset=pset)
    pred_thresholds = np.array([evolved_threshold(best_func(*x)) for x in X_test])
    y_pred = get_predicted_labels(SIMILARITY_test, pred_thresholds)
    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Duplicate', 'Predicted Non-Duplicate'],
                yticklabels=['True Duplicate', 'True Non-Duplicate'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    
def visualize_threshold_statistics(individual, X_test, pset):
    best_func = gp.compile(expr=individual, pset=pset)
    
    thresholds = np.array([evolved_threshold(best_func(*x)) for x in X_test])
    
    aggregate_min = np.min(thresholds)
    aggregate_max = np.max(thresholds)
    aggregate_mean = np.mean(thresholds)
    aggregate_std = np.std(thresholds)

    plt.figure(figsize=(10, 6))
    plt.hist(thresholds, bins=20, alpha=0.7, label='Thresholds', color='blue')

    plt.axvline(aggregate_min, color='red', linestyle='--', label=f'Min: {aggregate_min:.2f}')
    plt.axvline(aggregate_max, color='green', linestyle='--', label=f'Max: {aggregate_max:.2f}')
    plt.axvline(aggregate_mean, color='purple', linestyle='--', label=f'Mean: {aggregate_mean:.2f}')

    plt.errorbar(
        x=aggregate_mean,
        y=0.8 * plt.gca().get_ylim()[1],
        xerr=aggregate_std,
        fmt='o',
        color='black',
        label=f'Mean ± Std: {aggregate_mean:.2f} ± {aggregate_std:.2f}'
    )
    # Increase font sizes
    plt.xlabel('Threshold Value', fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    plt.title('Threshold Statistics for Solution', fontsize=16, pad=15)
    plt.legend(loc='upper right', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.show()
    


def plot_3D_objectives(pop):
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    pop = pareto_front

    # Extract data from population
    accuracy = []
    deviation = []
    tree_size = []

    for ind in pop:
        acc, var, _ = ind.fitness.values
        accuracy.append(acc)
        deviation.append(var)    
        tree_size.append(len(ind))
    
    # Increase figure size to reduce squishing
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with tree size as the color
    scatter = ax.scatter(
        deviation, 
        accuracy, 
        tree_size, 
        c=tree_size, 
        cmap='viridis', 
        s=50, 
        alpha=0.8,
        label="Full Population"
    )
    
    # Increase font sizes for axis labels, title, and color bar
    ax.set_xlabel("Deviation", fontsize=16, labelpad=10)
    ax.set_ylabel("Accuracy", fontsize=16, labelpad=10)
    ax.set_zlabel("Tree Size", fontsize=16, labelpad=10)
    ax.set_title("Pareto Front", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Color bar settings
    cbar = plt.colorbar(scatter)
    cbar.set_label("Tree Size", fontsize=14)
    ax.grid(True)
    plt.show()
    
    
def clf_comparison_gp_vs_static(ind, X_test, Y_test, SIMILARITY_test, pset):
    # Compile the best GP solution
    best_func = gp.compile(expr=ind, pset=pset)
    
    # Predict thresholds using GP
    test_threshold_predictions = [evolved_threshold(best_func(*x)) for x in X_test]
    gp_accuracy, _, _, _, _ = predicted_threshold_accuracy(SIMILARITY_test, test_threshold_predictions, Y_test)

    # Generate static thresholds
    static_thresholds = np.linspace(0, 1, 100)  # Thresholds from 0 to 1
    static_accuracies = [specific_threshold_accuracy(SIMILARITY_test, t, Y_test) for t in static_thresholds]
    
    # Identify the best static threshold
    best_threshold, best_static_accuracy = best_possible_threshold(SIMILARITY_test, Y_test, increment=0.01)

    # Visualization
    plt.figure(figsize=(10, 7))
    plt.plot(static_thresholds, static_accuracies, label='Static Threshold Accuracies', color='blue')
    plt.scatter(best_threshold, best_static_accuracy, color='red', label=f'Best Static Threshold', zorder=5)
    plt.axvline(best_threshold, color='red', linestyle='--', linewidth=1)
    plt.text(best_threshold, best_static_accuracy + 0.005, f'{best_static_accuracy:.3f}', ha='center', color='red', fontsize=11)

    # Add GP-derived accuracy
    plt.axhline(gp_accuracy, color='green', linestyle='--', label='GP Threshold Accuracy', zorder=5)
    plt.text(0.5, gp_accuracy + 0.005, f'GP: {gp_accuracy:.3f}', ha='center', color='green', fontsize=11)

    # Adjust plot limits
    min_y = min([gp_accuracy, min(static_accuracies)]) - 0.05
    max_y = max([gp_accuracy, max(static_accuracies)]) + 0.05
    plt.ylim(max(0, min_y), min(1, max_y))
    
    # Labels, title, and legend
    plt.xlabel('Threshold', fontsize=14, labelpad=10)
    plt.ylabel('Accuracy', fontsize=14, labelpad=10)
    plt.title('GP vs Static Threshold Accuracy', fontsize=16, pad=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.show()
    
    
def plot_objective_progress(logbook, objective_names =  ["Accuracy", "Deviation", "Normalized Tree Size"], generations=None):
    if generations is None:
        generations = len(logbook)

    gen = logbook.select("gen")

    base_colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]  # RGB for red, blue, green

    plt.figure(figsize=(12, 8))

    for i, name in enumerate(objective_names):
        color = base_colors[i % len(base_colors)]  # Cycle through the colors

        # Extract average and max values for the current objective
        avg_values = np.array([record["avg"][i] for record in logbook[:generations]])
        max_values = np.array([record["max"][i] for record in logbook[:generations]])

        # Plot average and maximum values with lighter/darker shades
        plt.plot(gen[:generations], avg_values, label=f"Avg {name}", linestyle="--", color=color, alpha=0.6)
        plt.plot(gen[:generations], max_values, label=f"Max {name}", color=color, alpha=1.0)

    # Increase font sizes
    plt.xlabel("Generations", fontsize=14, labelpad=10)
    plt.ylabel("Objective Values", fontsize=14, labelpad=10)
    plt.title("Progress of Objectives Over Generations", fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.show()







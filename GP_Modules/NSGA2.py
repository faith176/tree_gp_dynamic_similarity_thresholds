# PSET    
import random
import operator
import numpy as np
from GP_Modules.Create_PSET import create_PSET
from deap import base, creator, tools, gp
from joblib import Parallel, delayed
from deap.tools import ParetoFront

# Evaluation Function
from GP_Modules.Fitness_Function import evaluate_individual

def parallel_evaluation(individual, toolbox):
    return toolbox.evaluate(individual)

def configure_NSGA2_GP(feature_names, X_train, Y_train, SIMILARITY_train):
    pset = create_PSET(feature_names, SIMILARITY_train)
    
    # Define fitness function
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # Toolbox configuration
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=2, max_=8)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evaluate_individual, arg_names=feature_names, X=X_train, Y=Y_train, SIMILARITY=SIMILARITY_train, pset=pset)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
    toolbox.register("mutateUniform", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutateNodeReplacement", gp.mutNodeReplacement, pset=pset)
    def combined_mutation(individual, toolbox):
        if random.random() < 0.5:
            return toolbox.mutateNodeReplacement(individual)
        else:
            return toolbox.mutateUniform(individual)
    toolbox.register("mutate", combined_mutation, toolbox=toolbox)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8)) #12
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.register("map", map)
    return toolbox, pset


def NSGA2(toolbox, population_size=72, num_generations=100, prob_xover=0.8, 
                           prob_mutate=0.25, random_seed=42, verbose=True):
    random.seed(random_seed)
    pop = toolbox.population(n=population_size)
    hof = ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg"

    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = Parallel(n_jobs=-1)(delayed(parallel_evaluation)(ind, toolbox) for ind in invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the Hall of Fame
    hof.update(pop)

    # Initial sorting and crowding distance assignment
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, num_generations + 1):        
        # Generate offspring using tournament selection
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < prob_xover:
                toolbox.mate(ind1, ind2)
            if random.random() < prob_mutate:
                toolbox.mutate(ind1)
            if random.random() < prob_mutate:
                toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = Parallel(n_jobs=-1)(delayed(parallel_evaluation)(ind, toolbox) for ind in invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Perform non-dominated sorting and crowding distance assignment, Combine parents and offspring
        pop =  toolbox.select(pop + offspring, population_size)

        # Update the Hall of Fame
        hof.update(pop)

        # Log statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return pop, logbook, hof, stats


# PSET    
import random
import operator

import numpy as np
from GP_Modules.Create_PSET import create_PSET
from deap import algorithms, base, creator, tools, gp

# Evaluation Function
from GP_Modules.Fitness_Function import evaluate_individual


def configure_Multi_Obj_GP(feature_names, X_train, Y_train, SIMILARITY_train):
    # PSET
    pset = create_PSET(feature_names, SIMILARITY_train)
    
    # Define fitness function
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -0.5))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # Toolbox configuration
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genGrow, pset=pset, min_=2, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Specify evaluation on X_train and Y_train
    toolbox.register("evaluate", evaluate_individual, arg_names=feature_names, X=X_train, Y=Y_train, SIMILARITY=SIMILARITY_train, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
    toolbox.register("map", map)
    return toolbox, pset


def Multi_Obj(toolbox, population_size=72, num_generations=100, prob_xover=0.8, 
              prob_mutate=0.25, random_seed=42, verbose=True):
    random.seed(random_seed)
    pop = toolbox.population(n=population_size)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("std", np.std, axis=0)

    pop, logbook = algorithms.eaMuCommaLambda( 
        population=pop,
        toolbox=toolbox,
        mu=population_size,
        lambda_=population_size * 2,
        cxpb=prob_xover,
        mutpb=prob_mutate,
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )
    return pop, logbook, hof, stats
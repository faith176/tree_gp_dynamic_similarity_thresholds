import numpy as np
from GP_Modules.PSET_Functions import s_div, w_avg, if_else, gt, lt, s_average, s_pow, s_square, sigmoid, s_log, s_sqrt, s_round, safe_exp, safe_tanh, reciprocal, safe_cos, safe_sin
import operator
from deap import algorithms, base, creator, tools, gp
    
    
def create_PSET(feature_names, SIMILARITY_train, constants = [0.1, 0.5, 1]):
    if len(SIMILARITY_train) == 0:
        raise ValueError("SIMILARITY_train must not be empty.")

    # Primitives
    num_features = len(feature_names)
    pset = gp.PrimitiveSetTyped("main", [float]*num_features, float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(operator.neg, [float], float)
    pset.addPrimitive(s_div, [float, float], float)
    pset.addPrimitive(w_avg, [float, float, float], float)
    pset.addPrimitive(abs, [float], float)
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(max, [float, float], float)
    pset.addPrimitive(if_else, [float, float, float], float)
    pset.addPrimitive(gt, [float, float], float)
    pset.addPrimitive(lt, [float, float], float)
    pset.addPrimitive(s_average, [float, float], float)
    
    
    pset.addPrimitive(s_pow, [float, float], float)
    pset.addPrimitive(s_sqrt, [float], float)
    pset.addPrimitive(s_square, [float], float)
    pset.addPrimitive(sigmoid, [float], float)
    pset.addPrimitive(s_log, [float], float)
    pset.addPrimitive(s_round, [float], float)
    pset.addPrimitive(reciprocal, [float], float)
    pset.addPrimitive(safe_tanh, [float], float)
    pset.addPrimitive(safe_exp, [float], float)

    # Terminals
    for const in constants:
        pset.addTerminal(const, float)

    for i in range(num_features):
        pset.renameArguments(**{f'ARG{i}': f'{feature_names[i]}'})
    
    return pset
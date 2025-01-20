import math

import numpy as np

def s_div(x, y, epsilon=1e-10):
    try:
        if y == 0:
            return 0
        result = x / (y if abs(y) > epsilon else epsilon)
        return result
    except OverflowError:
        return 0

def w_avg(a, b, w):
    return w * a + (1 - w) * b

def s_log(x):
    try:
        if x > 0:
            return math.log(x)
        else:
            return 0
    except ValueError:
        return 0

def s_exp(x, cap=1e100):
    try:
        if x > math.log(cap):
            return cap
        return math.exp(x)
    except OverflowError:
        return 0
    
def s_pow(base, exponent):
    if base == 0 and exponent <= 0:
        return 1
    if base < 0 and not float(exponent).is_integer():
        return 0
    try:
        return base ** exponent
    except Exception as e:
        return 0


def s_sqrt(x):
    return math.sqrt(x) if x >= 0 else 0

def s_square(x):
    try:
        result = x * x
        return result
    except (OverflowError, ValueError, TypeError):
        return 0

def s_average(x, y):
    return (x + y) / 2

def s_variance(x, y):
    return (x - y) ** 2

def gt(x, y):
    return 1 if x > y else 0

def lt(x, y):
    return 1 if x < y else 0

def sigmoid(x):
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)
    except OverflowError:
        return 0 if x < 0 else 1

def limit(input, minimum, maximum):
    return min(max(input, minimum), maximum)

def if_else(condition, true_value, false_value):
    return true_value if condition > 0 else false_value

def threshold(value, threshold=0.5):
    return 1.0 if value > threshold else 0.0

def s_round(x):
    if math.isnan(x) or math.isinf(x):
        return 0
    return round(x, 2)

def s_trunc(x):
    if math.isnan(x) or math.isinf(x):
        return 0
    return math.trunc(x)


def reciprocal(x):
    return 1.0 / x if x != 0 else 0.0


def safe_exp(x, cap=700):  # Cap based on maximum float exponent
    if x > cap:
        return math.exp(cap)  # Use maximum allowed value
    elif x < -cap:
        return math.exp(-cap)  # Avoid underflow for large negative values
    else:
        return math.exp(x)


def safe_tanh(x):
    try:
        return np.tanh(x)
    except OverflowError:
        return 1.0 if x > 0 else -1.0
    
def safe_cos(x, bound=1000):
    if not isinstance(x, (int, float)):  # Handle non-numerical inputs
        return 1.0
    if abs(x) > bound:  # Normalize large inputs
        x = x % (2 * math.pi)
    return math.cos(x)

import math

def safe_sin(x, bound=1000):
    if not isinstance(x, (int, float)):  # Handle non-numerical inputs
        return 0.0
    if abs(x) > bound:  # Normalize large inputs
        x = x % (2 * math.pi)
    return math.sin(x)


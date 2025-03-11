import numpy as np

def init_weights_zero(shape):
    return np.zeros(shape)

def init_weights_uniform(shape, lower_bound, upper_bound, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(lower_bound, upper_bound, size=shape)

def init_weights_normal(shape, mean, variance, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, np.sqrt(variance), size=shape)
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

def init_weights_xavier_uniform(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def init_weights_xavier_normal(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, std, size=shape)

def init_weights_he(shape, lower_bound, upper_bound, seed=None):
    if seed is not None:
        np.random.seed(seed)
    fan_in, _ = shape
    std = np.sqrt(2 / fan_in)
    return np.random.normal(lower_bound, std, size=shape)
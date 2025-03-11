import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def d_mse_loss(y_true, y_pred):
    N = y_true.size
    return 2 * (y_pred - y_true) / N

def binary_cross_entropy_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def d_binary_cross_entropy_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    size = y_true.size
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / size

def categorical_cross_entropy_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def d_categorical_cross_entropy_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    size = y_true.shape[0]
    return - (y_true / y_pred) / size
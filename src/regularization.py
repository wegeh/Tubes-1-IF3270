import numpy as np

def l1_regularization(weights, lambda_reg):
    """
    L1 regularization (Lasso)
    """
    reg_cost = 0
    reg_grads = []
    
    for w in weights:
        reg_cost += lambda_reg * np.sum(np.abs(w))
        reg_grads.append(lambda_reg * np.sign(w))
    
    return reg_cost, reg_grads

def l2_regularization(weights, lambda_reg):
    """
    L2 regularization (Ridge)
    """
    reg_cost = 0
    reg_grads = []
    
    for w in weights:
        reg_cost += 0.5 * lambda_reg * np.sum(np.square(w))
        reg_grads.append(lambda_reg * w)
    
    return reg_cost, reg_grads

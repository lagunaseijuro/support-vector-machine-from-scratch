"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    std = np.std(x, axis=0)
    std = np.where(std == 0, 1, std)
    return (x - np.mean(x, axis=0)) / std

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    weights = np.zeros(n_features) 
    bias = 0.0
    
    return {
        'w': weights,
        'b': bias
    }

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    return x @ params['w'] + params['b']

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    return np.max([0, 1 - score * y])

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    w = params['w']
    b = params['b']

    margins = x @ w + b

    hinge_loss = np.maximum(0, 1 - y * margins)

    return np.mean(hinge_loss) + reg_lambda * np.sum(w * w)

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    n = x.shape[0]
    raw_scores = compute_scores(x, params)
    margins = 1 - y * raw_scores

    bool_mask = (margins > 0).astype(float)

    dW = -(x.T @ (bool_mask * y)) / n + 2 * reg_lambda * params['w']
    db = -np.mean(y * bool_mask)

    return {
        'dw': dW,
        'db': db
    }

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    return {
        'w' : params['w'] - learning_rate * grads['dw'],
        'b' : params['b'] - learning_rate * grads['db']
    }

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    n_samples, n_features = x.shape

    params = initialize_parameters(n_features)
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)

        params = apply_update(params, grads, learning_rate)
    
    return params

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement


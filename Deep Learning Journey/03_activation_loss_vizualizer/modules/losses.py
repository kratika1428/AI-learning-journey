import numpy as np

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)     #accepts array or numbers
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)      #np.mean() averages the error across all examples

def binary_cross_entropy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)    #keeps bce predictions away from exactly 0 and 1
    loss = -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))   #np.log() calculates the natural logarithm used
    return np.mean(loss)
import numpy as np

def mean_squared_error(actual, prediction):
    error = actual - prediction
    squared_error = error ** 2
    mse = np.mean(squared_error)

    return mse
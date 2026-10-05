import numpy as np

def calculate_gradient(x, y, predictions):
    n = len(x)
    error = predictions - y

    weight_gradient = (2/n) * np.sum(x * error)  #2/n * x * (y' - y)
    bias_gradient = (2/n) * np.sum(error)   #2/n * (y' - y)

    return weight_gradient, bias_gradient
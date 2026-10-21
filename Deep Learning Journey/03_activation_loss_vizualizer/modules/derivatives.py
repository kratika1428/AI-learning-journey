import numpy as np

from modules.activation import sigmoid

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def reLU_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t**2

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)
import numpy as np

# Squash values into the range (0, 1)
def sigmoid(x):
    return 1 / (1+np.exp(-np.clip(x,-500,500)))

# Return 0 for negative value, otherwise x 
def reLU(x):
    return np.maximum(0,x)

# Squash values into range (-1,1)
def tanh(x):
    return np.tanh(x)

# Allow a small negative output for negative values
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha*x)
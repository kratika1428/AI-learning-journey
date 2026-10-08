import numpy as np
import matplotlib.pyplot as plt

from models.neural_network import NeuralNetwork

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

model = NeuralNetwork(
    learning_rate=0.5,
    epochs=10000
)
model.fit(X, y)

# Plot loss
plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Neural Network Training Loss")
plt.show()
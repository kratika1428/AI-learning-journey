import numpy as np

from models.perceptron import perceptron
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
y_perceptron = y.flatten()

#single neural network: perceptron
perceptron = perceptron(
    learning_rate=0.1,
    epochs=20
)
perceptron.fit(X, y_perceptron)
perceptron_predictions = perceptron.predict(X)
perceptron_accuracy = np.mean(
    perceptron_predictions == y_perceptron
)

#multi neural network
neural_network = NeuralNetwork(
    learning_rate=0.5,
    epochs=10000
)
neural_network.fit(X, y)
nn_predictions = neural_network.predict(X)
nn_accuracy = np.mean(
    nn_predictions == y
)

print("\n Model Comparison")
print(f"Perceptron Accuracy: {perceptron_accuracy * 100:.2f}%")
print(f"Neural Network Accuracy: {nn_accuracy * 100:.2f}%")

print("\nPerceptron Predictions:")
print(perceptron_predictions)

print("\nNeural Network Predictions:")
print(nn_predictions.flatten())

print("\nActual:")
print(y.flatten())
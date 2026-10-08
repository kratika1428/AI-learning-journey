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

x1 = np.linspace(-0.5, 1.5, 200)
x2 = np.linspace(-0.5, 1.5, 200)
xx, yy = np.meshgrid(x1, x2)
grid = np.c_[xx.ravel(), yy.ravel()]

predictions = model.predict(grid)
predictions = predictions.reshape(xx.shape)

plt.contourf(
    xx,
    yy,
    predictions,
    alpha=0.3
)
for i in range(len(X)):
    if y[i][0] == 0:
        marker = 'o'
    else:
        marker = 'x'
    plt.scatter(
        X[i, 0],
        X[i, 1],
        marker=marker,
        s=150
    )
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("XOR Decision Boundary")

plt.show()
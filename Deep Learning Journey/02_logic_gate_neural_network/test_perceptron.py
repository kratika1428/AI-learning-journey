import numpy as np

from models.perceptron import perceptron

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([0,0,0,1])

model = perceptron(
    learning_rate=0.1,
    epochs=10
)
model.fit(X,y)
predictions = model.predict(X)

print("Weight:",model.weight)
print("Bias:",model.bias)
print("Predictions:", predictions)
print("Actual:",y)
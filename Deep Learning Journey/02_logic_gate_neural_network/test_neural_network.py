import numpy as np

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
model.fit(X, y) #train
output = model.forward(X)   #raw sigmoid outputs
predictions = model.predict(X)  #final prediction
accuracy = np.mean(predictions == y)    #accuracy

print("\n Output:", output)
print("\n Prediction:", predictions)
print("\n Actual:",y)
print("\n Accuracy:", accuracy * 100, "%")
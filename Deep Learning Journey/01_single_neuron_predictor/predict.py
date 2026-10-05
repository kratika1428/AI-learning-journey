import numpy as np

from neuron import neuron

model = np.load("trained_model.npz")
weight = model["weight"]
bias = model["bias"]

neuron = neuron(
    weight = weight,
    bias = bias
)

hours = float(input("Enter hours studied: "))
if hours < 0:
    print("invalid input")
    exit()
prediction = neuron.forward(hours)

print(f"\nHours Studied: {hours}")
print(f"Predicted Exam Score: {prediction:.2f}")
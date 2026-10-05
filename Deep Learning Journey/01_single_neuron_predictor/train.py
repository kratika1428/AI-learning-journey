import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuron import neuron
from gradient import calculate_gradient
from loss import mean_squared_error

#data reading and information
data = pd.read_csv("data/student_data.csv")

#getting input and target data
x = data["Hours_Studied"].values
y = data["Exam_Score"].values

#creation of neuron and determining parameters
neuron = neuron(weight=5, bias=30)
learning_rate = 0.01
epochs = 1000
loss_history = []

for epoch in range(epochs):

    #making prediction
    prediction = neuron.forward(x)

    #calculating loss
    loss = mean_squared_error(y, prediction)
    loss_history.append(loss)

    #calculating gradient of weight and bias
    weight_gradient, bias_gradient = calculate_gradient(x, y, prediction)

    #update weight and bias
    neuron.update(
        weight_gradient,
        bias_gradient,
        learning_rate
    )

    # Display progress
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch}, "
            f"Loss: {loss:.4f}, "
            f"Weight: {neuron.weight:.4f}, "
            f"Bias: {neuron.bias:.4f}"
        )

print("\n TRAINING COMPLETE")
print("Final weight:", neuron.weight)
print("Final bias:", neuron.bias)
print("Final loss:",loss_history[-1])

#visualization of loss history
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss : MSE")
plt.title("Training Loss")

plt.show()

#saving trained parameters
np.savez(
    "trained_model.npz",
    weight = neuron.weight,
    bias = neuron.bias
)
print("\n Model Saved Successfully")
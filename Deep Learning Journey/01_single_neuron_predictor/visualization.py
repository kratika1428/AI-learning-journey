import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuron import neuron

data = pd.read_csv("data/student_data.csv")
x = data["Hours_Studied"].values
y = data["Exam_Score"].values

model = np.load("trained_model.npz")
weight = model["weight"]
bias = model["bias"]

neuron = neuron(
    weight = weight,
    bias = bias
)
predictions = neuron.forward(x)

print("Actual vs Predicted")
for hours, actual, predicted in zip(x, y, predictions):
    print(
        f"Hours: {hours:2d} | "
        f"Actual: {actual:3d} | "
        f"Predicted: {predicted:.2f}"
    )

#visualization of actual vs predicted
plt.scatter(x, y, label="Actual Score")
plt.plot(x, predictions, label="Predicted Score")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Actual vs Predicted Score")

plt.legend()
plt.show()

#visualization of prediction errors
error = y - predictions
plt.bar(x,error)
plt.axhline(0,linestyle="--")

plt.xlabel("Hours studied")
plt.ylabel("Prediction error")
plt.title("Prediction Error")

plt.show()
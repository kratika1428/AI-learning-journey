import pandas as pd
import matplotlib.pyplot as plt

from neuron import neuron
from loss import mean_squared_error
from gradient import calculate_gradient

data = pd.read_csv("data/student_data.csv")
x = data["Hours_Studied"].values
y = data["Exam_Score"].values

neuron = neuron(weight=5, bias=30)
learning_rate = 0.01

#making prediction
prediction = neuron.forward(x)

#calculating loss
loss_before = mean_squared_error(y, prediction)

#calculating gradient of weight and bias
weight_gradient, bias_gradient = calculate_gradient(x, y, prediction)

#before updating
print("BEFORE UPDATE")
print("\n Weight:", neuron.weight)
print("\n Bias:", neuron.bias)
print("\n Loss Before:", loss_before)

neuron.update(
    weight_gradient,
    bias_gradient,
    learning_rate
)

prediction = neuron.forward(x)
loss_after = mean_squared_error(y, prediction)
#after updating
print("AFTER UPDATE")
print("\n Weight:", neuron.weight)
print("\n Bias:", neuron.bias)
print("\n Loss After:", loss_after)


# print("Actual Score:", y)
# print("Predicted score:", prediction)
# print("Mean squared error:", loss)

# print("\n Weight Gradient:", weight_gradient)
# print("\n Bias Gradient:", bias_gradient)

# #visualization
# plt.scatter(x, y, label="Actual data")
# plt.plot(x, prediction, label="Neuron Prediction")
# plt.xlabel("Hours Studied")
# plt.ylabel("Exam Score")
# plt.title("Neuron Prediction vs Actual data")

# plt.legend()
# plt.show()
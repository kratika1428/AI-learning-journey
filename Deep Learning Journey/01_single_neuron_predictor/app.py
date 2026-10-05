import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuron import neuron
from loss import mean_squared_error
from gradient import calculate_gradient

st.set_page_config(
    page_title="Single Neuron Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Single Neuron Predictor")
st.write(
    "Predict an exam score based on the number of hours studied "
    "using a neuron implemented from scratch with NumPy."
)

#dataset loading and reading
data = pd.read_csv("data/student_data.csv")
X = data["Hours_Studied"].values
y = data["Exam_Score"].values

#model training and parameter setting
neuron = neuron(weight=5, bias=30)
learning_rate = 0.01
epochs = 1000
loss_history = []

for epoch in range(epochs):
    predictions = neuron.forward(X)
    loss = mean_squared_error(y, predictions)
    loss_history.append(loss)

    weight_gradient, bias_gradient = calculate_gradient(
        X,
        y,
        predictions
    )
    neuron.update(
        weight_gradient,
        bias_gradient,
        learning_rate
    )

st.subheader("Model Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Learned Weight",
        f"{neuron.weight:.4f}"
    )
with col2:
    st.metric(
        "Learned Bias",
        f"{neuron.bias:.4f}"
    )
with col3:
    st.metric(
        "Final MSE",
        f"{loss_history[-1]:.4f}"
    )

st.subheader("Predict Exam Score")
hours = st.number_input(
    "Enter Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)
if st.button("Predict Score"):
    prediction = neuron.forward(hours)
    st.success(
        f"Predicted Exam Score: {prediction:.2f}"
    )

#visualization of training loss
st.subheader("Training Loss")
fig1, ax1 = plt.subplots()
ax1.plot(loss_history)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.set_title("Loss vs Epoch")

st.pyplot(fig1)

#visualization of actual data vs learned prediction data
st.subheader("Learned Prediction Line")
line_x = np.linspace(
    X.min(),
    X.max(),
    100
)
line_y = neuron.forward(line_x)

fig2, ax2 = plt.subplots()
ax2.scatter(
    X,
    y,
    label="Actual Data"
)
ax2.plot(
    line_x,
    line_y,
    label="Neuron Prediction"
)

ax2.set_xlabel("Hours Studied")
ax2.set_ylabel("Exam Score")
ax2.set_title(
    "Hours Studied vs Exam Score"
)

ax2.legend()
st.pyplot(fig2)

#table of actual vs predicted
st.subheader("Actual vs Predicted")
predictions = neuron.forward(X)
results = pd.DataFrame({
    "Hours Studied": X,
    "Actual Score": y,
    "Predicted Score": np.round(predictions, 2)
})
st.dataframe(
    results,
    use_container_width=True
)
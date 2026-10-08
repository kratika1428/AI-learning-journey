import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from models.perceptron import perceptron
from models.neural_network import NeuralNetwork
from data.logic_gate import *

st.set_page_config(
    page_title="Logic Gate Neural Network",
    layout="centered"
)
st.title("Logic Gate Neural Network")
st.write("Compare a single perceptron with a multi layer neural network")

gates = {
    "AND": (AND_X, AND_Y),
    "OR": (OR_X, OR_Y),
    "NOT": (NOT_X, NOT_Y),
    "XOR": (XOR_X, XOR_Y)
}

st.sidebar.header("Model Configuration")
gate_name = st.sidebar.selectbox(
    "Select Logic Gate",
    ["AND","OR","NOT","XOR"]
)
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Perceptron","Neural Network"]
)
learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value = 0.01,
    max_value = 1.0,
    value = 0.5
)
epochs = st.sidebar.slider(
    "Epochs",
    min_value=100,
    max_value=20000,
    value=10000,
    step=100
)

X, y = gates[gate_name]
if st.button("Train Model"):
    if model_name == "Perceptron":
        model = perceptron(
            learning_rate = learning_rate,
            epochs = epochs
        )
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)

        st.session_state.model = model
        st.session_state.predictions = predictions
        st.session_state.accuracy = accuracy
    else:
        y_nn = y.reshape(-1, 1)
        model = NeuralNetwork(
            learning_rate = learning_rate,
            epochs = epochs
        )
        model.fit(X, y_nn)
        predictions = model.predict(X)
        accuracy = np.mean(predictions.flatten() == y)

        st.session_state.model = model
        st.session_state.predictions = predictions.flatten()
        st.session_state.accuracy = accuracy

if "accuracy" in st.session_state:
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Accuracy",
            f"{st.session_state.accuracy * 100:.2f}%"
        )
    with col2:
        st.metric("Gate",gate_name)

    st.subheader("Predictions")
    predictions = st.session_state.predictions
    for inputs, prediction, actual in zip(X, predictions, y):
        st.write(
            f"Input: `{inputs}` → "
            f"Prediction: `{prediction}` → "
            f"Actual: `{actual}`"
        )
    if model_name == "Neural Network":
        st.subheader("Training Loss")

        fig, ax = plt.subplots()
        ax.plot(
            st.session_state.model.loss_history
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")

        st.pyplot(fig)
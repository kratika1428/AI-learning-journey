import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

from modules.activation import *
from modules.derivatives import *
from modules.losses import *

st.set_page_config(
    page_title="Vizualizer",
    layout="centered"
)
st.title("Activation & Loss Function Lab")
st.write("An interactive playground for understanding activation function, its derivatives and losses")
st.divider()

activation_tab, loss_tab = st.tabs([
    "Activation Explorer",
    "Loss Explorer"
])
with activation_tab:
    activation_function = {
        "Sigmoid" : (sigmoid, sigmoid_derivative),
        "ReLU" : (reLU, reLU_derivative),
        "Tanh" : (tanh, tanh_derivative),
        "Leaky ReLU" : (leaky_relu, leaky_relu_derivative)
    }

    selected = st.selectbox(
        "Select an activation function",
        list(activation_function.keys())
    )
    x_min, x_max = st.slider(
        "Input range",
        min_value=-10.0,
        max_value=10.0,
        value=(-5.0, 5.0),
        step=0.5,
    )

    x = np.linspace(x_min, x_max, 500)  #used to generate evenly spaced number between range
    activation_fn, derivative_fn = activation_function[selected]

    y = activation_fn(x)
    dy = derivative_fn(x)
    
    st.subheader("Explore a specific input")
    input_value = st.slider(
        "Input value (x)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        key="activation_input",
    )
    function_value = float(activation_fn(input_value))
    derivative_value = float(derivative_fn(input_value))

    col1, col2 = st.columns(2)
    col1.metric(
        "Activation output f(x)",
        f"{function_value:.5f}",
    )
    col2.metric(
        "Derivative f'(x)",
        f"{derivative_value:.5f}",
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{selected}: f(x)",
            f"{selected}: f'(x)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Activation",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=dy,
            mode="lines",
            name="Derivative",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Input (x)")
    fig.update_yaxes(title_text="Output f(x)", row=1, col=1)
    fig.update_yaxes(title_text="Derivative f'(x)", row=1, col=2)

    fig.update_layout(
        height=450,
        showlegend=False,
        title_text="Activation and Derivative",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Understanding the result")
    if selected == "Sigmoid":
        st.write(
            "Sigmoid maps inputs to values between 0 and 1. Its derivative is largest near x = 0 and small at extreme inputs."
        )
    elif selected == "ReLU":
        st.write(
            "ReLU outputs zero for negative inputs and x for positive inputs. Its derivative is zero on the negative side and one on the positive side."
        )
    elif selected == "Tanh":
        st.write(
            "Tanh maps inputs between -1 and 1. Its derivative is largest near zero and approaches zero at extreme inputs."
        )
    else:
        st.write(
            "Leaky ReLU allows a small negative output for negative inputs, with a small derivative of 0.01."
        )

with loss_tab:        
    loss_choice = st.selectbox(
        "Select a loss function",
        ["Mean Squared Error (MSE)", "Binary Cross-Entropy (BCE)"],
    )
    y_true = st.selectbox(
        "True value (y)",
        [0.0, 1.0],
        index=1,
    )
    prediction = st.slider(
        "Predicted value (y_hat)",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
    )

    if loss_choice == "Mean Squared Error (MSE)":
        loss = mean_squared_error(y_true, prediction)
        loss_function = mean_squared_error
    else:
        loss = binary_cross_entropy(y_true, prediction)
        loss_function = binary_cross_entropy

    st.metric("Current loss", f"{loss:.6f}")
    prediction_range = np.linspace(0.01, 0.99, 300)
    
    st.subheader("Interpret the prediction")

    if abs(prediction - y_true) < 1e-9:
        st.success("The prediction matches the true value.")
    elif prediction > y_true:
        st.info("The prediction is above the true value.")
    else:
        st.info("The prediction is below the true value.")

    if loss_choice == "Mean Squared Error (MSE)":
        st.write(
            "MSE measures squared prediction error. "
            "For a single example, the loss is the squared "
            "difference between the true value and prediction."
        )
    else:
        st.write(
            "Binary Cross-Entropy evaluates the predicted "
            "probability assigned to the true binary class. "
            "Confident, incorrect predictions receive a large penalty."
        )

    loss_values = [
        loss_function(y_true, p)
        for p in prediction_range
    ]
    loss_fig = go.Figure()
    loss_fig.add_trace(
        go.Scatter(
            x=prediction_range,
            y=loss_values,
            mode="lines",
            name="Loss curve",
        )
    )
    loss_fig.add_trace(
        go.Scatter(
            x=[prediction],
            y=[loss],
            mode="markers",
            marker=dict(size=12),
            name="Current prediction",
        )
    )
    loss_fig.update_layout(
        title=f"{loss_choice} vs. Prediction",
        xaxis_title="Predicted value",
        yaxis_title="Loss",
        height=450,
    )
    st.plotly_chart(loss_fig, use_container_width=True)

    st.caption(
        f"True value: {y_true:.0f} | "
        f"Prediction: {prediction:.2f} | "
        f"Loss: {loss:.6f}"
    )
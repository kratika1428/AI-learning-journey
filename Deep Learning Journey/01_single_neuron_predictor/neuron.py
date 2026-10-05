import numpy as np

#actual neuron
class neuron:
    def __init__(self,weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return self.weight * x + self.bias       #implements y = wx + b

    def update(self, weight_gradient, bias_gradient, learning_rate):
        self.weight = self.weight - learning_rate * weight_gradient
        self.bias = self.bias - learning_rate * bias_gradient
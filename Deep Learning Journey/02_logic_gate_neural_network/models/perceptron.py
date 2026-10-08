import numpy as np

class perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weight = None
        self.bias = 0

    #step activation function
    def activation(self, z):
        if z >= 0:
            return 1
        else:
            return 0

    def predict(self, x):
        predictions = []
        for x in x:
            z = np.dot(x, self.weight) + self.bias  #z = w1x1 + w2x2 + b
            prediction = self.activation(z)
            predictions.append(prediction)
        return np.array(predictions)

    def fit(self, x, y):
        n_features = x.shape[1]
        self.weight = np.zeros(n_features)
        self.bias = 0
        for epoch in range(self.epochs):
            for sample, target in zip(x, y):
                z = np.dot(sample, self.weight) + self.bias
                prediction = self.activation(z)
                error = target - prediction
                self.weight += (self.learning_rate * error * sample) # w = w + n(error)x
                self.bias += (self.learning_rate * error)   # b = b + n(error)
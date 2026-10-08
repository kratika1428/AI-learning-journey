import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate=0.5, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        np.random.seed(1)   #produces the same random number every iteration

        #input -> hidden
        self.w1 = np.random.randn(2,2)
        self.b1 = np.zeros((1,2))
        #hidden -> output
        self.w2 = np.random.randn(2,1)
        self.b2 = np.zeros((1,1))

        #loss history
        self.loss_history = []

    def sigmoid(self,z):
        return 1/ (1+np.exp(-z))

    def sigmoid_derivative(self, output):
        return output * (1-output)

    #X -> X @ w1 + b1 -> Z1 -> Sigmoid -> A1 -> A1 @ w2 + b2 -> Z2 -> Sigmoid -> A2
    def forward(self, X):

        # Input → Hidden
        self.Z1 = X @ self.w1 + self.b1
        self.A1 = self.sigmoid(self.Z1)
        # Hidden → Output
        self.Z2 = self.A1 @ self.w2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def loss(self, y, output):
        return np.mean((y-output) ** 2)

    def backward(self, x, y):
        #derivative of MSE with respect to output
        dA2 = 2 * (self.A2 - y)

        #derivative through sigmoid
        dZ2 = (dA2 * self.sigmoid_derivative(self.A2))

        #gradients for w2 and b2
        dw2 = self.A1.T @dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        #hidden layer
        dA1 = dZ2 @self.w2.T
        dZ1 = (dA1 * self.sigmoid_derivative(self.A1))

        #gradient for w1 and b1
        dw1 = x.T @dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        #update weights and biases
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

    #training of the model
    def fit(self, x, y):
        self.loss_history = []
        for epoch in range(self.epochs):
            output = self.forward(x)
            current_loss = self.loss(y, output)
            self.loss_history.append(current_loss)
            self.backward(x, y)
            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch} | Loss: {current_loss:.6f}"
                )

    #prediction
    def predict(self, x):
        output = self.forward(x)
        return (output >= 0.5).astype(int)
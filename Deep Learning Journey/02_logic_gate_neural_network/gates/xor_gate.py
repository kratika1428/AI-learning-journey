from models.perceptron import perceptron
from data.logic_gate import XOR_X, XOR_Y

model = perceptron(
    learning_rate=0.1,
    epochs=10
)
model.fit(XOR_X, XOR_Y)
predictions = model.predict(XOR_X)

print("XOR Gate")
print("Predictions:", predictions)
print("Actual:     ", XOR_Y)

accuracy = (predictions == XOR_Y).mean()
print("Accuracy:", accuracy * 100, "%")
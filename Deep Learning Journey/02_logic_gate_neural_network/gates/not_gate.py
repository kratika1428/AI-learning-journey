from models.perceptron import perceptron
from data.logic_gate import NOT_X, NOT_Y

model = perceptron(
    learning_rate=0.1,
    epochs=10
)
model.fit(NOT_X, NOT_Y)
predictions = model.predict(NOT_X)

print("NOT Gate")
print("Predictions:", predictions)
print("Actual:     ", NOT_Y)
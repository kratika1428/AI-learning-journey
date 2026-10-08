from models.perceptron import perceptron
from data.logic_gate import AND_X, AND_Y

model = perceptron(
    learning_rate=0.1,
    epochs=10
)
model.fit(AND_X, AND_Y)
predictions = model.predict(AND_X)

print("AND Gate")
print("Predictions:", predictions)
print("Actual:     ", AND_Y)
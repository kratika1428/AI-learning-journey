from models.perceptron import perceptron
from data.logic_gate import OR_X, OR_Y

model = perceptron(
    learning_rate=0.1,
    epochs=10
)
model.fit(OR_X, OR_Y)
predictions = model.predict(OR_X)

print("OR Gate")
print("Predictions:", predictions)
print("Actual:     ", OR_Y)
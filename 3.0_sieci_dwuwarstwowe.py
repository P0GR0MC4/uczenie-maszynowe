import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        # Inicjalizacja wag
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        # Zmienne dla Momentum
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.errors = []

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            # Forward
            layer1 = sigmoid(np.dot(X, self.w1))
            output = sigmoid(np.dot(layer1, self.w2))
            
            # Error
            error = y - output
            self.errors.append(np.mean(np.abs(error)))
            
            # Backpropagation
            d_output = error * sigmoid_derivative(output)
            d_layer1 = d_output.dot(self.w2.T) * sigmoid_derivative(layer1)
            
            # Update z Momentum
            self.v2 = self.momentum * self.v2 + self.lr * layer1.T.dot(d_output)
            self.v1 = self.momentum * self.v1 + self.lr * X.T.dot(d_layer1)
            
            self.w2 += self.v2
            self.w1 += self.v1

# TEST NA XOR
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([[0], [1], [1], [0]])

nn = TwoLayerNN(2, 4, 1)
nn.train(X_xor, y_xor)

plt.plot(nn.errors)
plt.title("Wykres błędu uczenia (XOR)")
plt.xlabel("Epoka")
plt.ylabel("Błąd")
plt.show()

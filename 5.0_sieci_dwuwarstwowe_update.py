import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

class AdvancedNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.initial_lr = lr # Do adaptacyjnego uczenia
        
        # Wagi
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        
        # Momentum
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.errors = []

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            # 1. Adaptacyjny współczynnik uczenia (Wymóg na 5.0)
            # Zmniejszamy LR co 2000 epok, aby precyzyjniej trafić w minimum
            if epoch > 0 and epoch % 2000 == 0:
                self.lr *= 0.5
            
            # Forward
            layer1 = sigmoid(np.dot(X, self.w1))
            output = sigmoid(np.dot(layer1, self.w2))
            
            # Error
            error = y - output
            self.errors.append(np.mean(np.abs(error)))
            
            # Backpropagation
            d_output = error * sigmoid_derivative(output)
            d_layer1 = d_output.dot(self.w2.T) * sigmoid_derivative(layer1)
            
            # Update z Momentum (Wymóg na 5.0)
            self.v2 = self.momentum * self.v2 + self.lr * layer1.T.dot(d_output)
            self.v1 = self.momentum * self.v1 + self.lr * X.T.dot(d_layer1)
            
            self.w2 += self.v2
            self.w1 += self.v1

# --- TEST 1: XOR ---
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([[0], [1], [1], [0]])

nn_xor = AdvancedNN(2, 4, 1)
nn_xor.train(X_xor, y_xor, epochs=5000)

# --- TEST 2: TITANIC (Uproszczony przykład danych) ---
# Płeć (0/1), Klasa (1/2/3), Wiek (skalowany) -> Przeżył (0/1)
X_titanic = np.array([[1, 3, 0.22], [0, 1, 0.38], [0, 3, 0.26], [0, 1, 0.35]])
y_titanic = np.array([[0], [1], [1], [1]])

nn_titanic = AdvancedNN(3, 5, 1)
nn_titanic.train(X_titanic, y_titanic, epochs=5000)

# WYKRESY BŁĘDÓW (Wymóg na 3.0 i 5.0)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(nn_xor.errors)
plt.title("Błąd uczenia: XOR (Momentum + Adaptive LR)")
plt.xlabel("Epoka")

plt.subplot(1, 2, 2)
plt.plot(nn_titanic.errors)
plt.title("Błąd uczenia: Titanic (Momentum + Adaptive LR)")
plt.xlabel("Epoka")

plt.tight_layout()
plt.show()

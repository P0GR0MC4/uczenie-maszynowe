import numpy as np
import matplotlib.pyplot as plt

# Funkcja aktywacji Sigmoid - mapuje wartości na zakres (0, 1)
def sigmoid(x): return 1 / (1 + np.exp(-x))

# Pochodna funkcji sigmoid
def sigmoid_derivative(x): return x * (1 - x)

class AdvancedNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, momentum=0.9):
        self.lr = lr             # Współczynnik uczenia
        self.momentum = momentum # Współczynnik Momentum
        self.initial_lr = lr

        # Inicjalizacja wag losowymi wartościami z zakresu (-1, 1)
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))

        # Inicjalizacja wektorów prędkości dla mechanizmu Momentum
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.errors = [] # Lista do przechowywania historii błędu

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            #1. ADAPTACYJNY LEARNING RATE
            # Mechanizm Scheduler: zmniejszenie LR o połowę co 2000 epok.
            # Pozwala to na duże kroki na początku i precyzyjne "dostrajanie" na końcu.
            if epoch > 0 and epoch % 2000 == 0:
                self.lr *= 0.5

            #2. FORWARD PROPAGATION
            # Obliczenie aktywacji warstwy ukrytej i wyjściowej
            layer1 = sigmoid(np.dot(X, self.w1))
            output = sigmoid(np.dot(layer1, self.w2))

            #3. OBLICZENIE BŁĘDU
            error = y - output
            self.errors.append(np.mean(np.abs(error))) # Średni błąd bezwzględny

            #4. BACKPROPAGATION (Propagacja wsteczna)
            # Wyznaczenie gradientów dla wag warstwy wyjściowej i ukrytej
            d_output = error * sigmoid_derivative(output)
            d_layer1 = d_output.dot(self.w2.T) * sigmoid_derivative(layer1)

            #5. AKTUALIZACJA WAG Z MOMENTUM
            # v = momentum * poprzednia_predkosc + lr * gradient
            # Mechanizm ten pozwala "przeskoczyć" minima lokalne i przyspiesza zbieżność.
            self.v2 = self.momentum * self.v2 + self.lr * layer1.T.dot(d_output)
            self.v1 = self.momentum * self.v1 + self.lr * X.T.dot(d_layer1)

            self.w2 += self.v2
            self.w1 += self.v1

# TESTY MODELU
# XOR: Klasyczny problem nieliniowy, niemożliwy do rozwiązania przez 1-warstwową sieć.
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([[0], [1], [1], [0]])

# Titanic (uproszczony)
X_titanic = np.array([[1, 3, 0.22], [0, 1, 0.38], [0, 3, 0.26], [0, 1, 0.35]])
y_titanic = np.array([[0], [1], [1], [1]])

# Trening modeli
nn_xor = AdvancedNN(2, 4, 1)
nn_xor.train(X_xor, y_xor, epochs=5000)

nn_titanic = AdvancedNN(3, 5, 1)
nn_titanic.train(X_titanic, y_titanic, epochs=5000)

# --- NOWA SEKCJA: GENEROWANIE WYKRESÓW DO SPRAWOZDANIA ---
plt.figure(figsize=(12, 5))

# Wykres błędu dla XOR
plt.subplot(1, 2, 1)
plt.plot(nn_xor.errors, color='blue')
plt.title("Błąd uczenia: XOR\n(Momentum + Adaptive LR)")
plt.xlabel("Epoka")
plt.ylabel("Średni błąd bezwzględny")
plt.grid(True, linestyle='--', alpha=0.6)

# Wykres błędu dla Titanic
plt.subplot(1, 2, 2)
plt.plot(nn_titanic.errors, color='red')
plt.title("Błąd uczenia: Titanic\n(Momentum + Adaptive LR)")
plt.xlabel("Epoka")
plt.ylabel("Średni błąd bezwzględny")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show() # Wyświetla okno z wykresami

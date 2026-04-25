import tensorflow as tf
from tensorflow.keras import layers, models

#1. PRZYGOTOWANIE DANYCH
# Pobieranie zbioru MNIST
# Zwraca krotki: dane treningowe (60k) i testowe (10k).
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Zmiana kształtu danych: 
# Obrazy 28x28 musimy zapisać jako (28, 28, 1), gdzie 1 to liczba kanałów (skala szarości).
# Normalizacja: dzielimy przez 255, aby wartości pikseli były w zakresie [0, 1] zamiast [0, 255].
# Ułatwia to i przyspiesza proces uczenia sieci.
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255

#2. ARCHITEKTURA MODELU (CNN)
model = models.Sequential([
    # Warstwa konwolucyjna: 32 filtry o rozmiarze 3x3. 
    # Wykrywa cechy obrazu (krawędzie, kształty). Funkcja 'relu' wprowadza nieliniowość.
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Warstwa MaxPooling: redukuje wymiary obrazu o połowę (bierze max wartość z okna 2x2).
    # Zmniejsza liczbę parametrów i zapobiega przeuczeniu.
    layers.MaxPooling2D((2, 2)),
    
    # Warstwa Flatten: "prostuje" macierz 2D do jednego długiego wektora 1D.
    # Przygotowuje dane do wejścia do klasycznych warstw gęstych (Dense).
    layers.Flatten(),
    
    # Warstwa ukryta gęsta: 64 neurony wyciągające wnioski z cech wykrytych przez CNN.
    layers.Dense(64, activation='relu'),
    
    # Warstwa wyjściowa: 10 neuronów (odpowiadających cyfrom 0-9).
    # Funkcja 'softmax' zamienia sygnały na prawdopodobieństwo (suma wszystkich = 100%).
    layers.Dense(10, activation='softmax')
])

#3. KOMPILACJA I TRENOWANIE
# Optimizer 'adam': inteligentny algorytm aktualizacji wag (adaptacyjny).
# Loss 'sparse_categorical_crossentropy': funkcja błędu dla klasyfikacji wieloklasowej.
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Uruchomienie procesu uczenia na 3 epoki (trzykrotne przejście przez cały zbiór danych).
model.fit(train_images, train_labels, epochs=3)

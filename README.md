# Projekt – Uczenie maszynowe i sieci neuronowe

## Opis projektu
Projekt obejmuje zastosowanie wybranych metod z zakresu uczenia maszynowego oraz sieci neuronowych. 
W ramach pracy wykonano modele klasyfikacyjne, analizę danych oraz autorską implementację sieci neuronowej.

## Zawartość plików

### 3.0_Bayes.py
Skrypt realizujący klasyfikację binarną użytkowników na podstawie wieku oraz czasu spędzonego w serwisie.
Zastosowane modele:
- Gaussian Naive Bayes
- Decision Tree

Metryki oceny:
- precision
- recall
- F1-score

---

### 3.0_mnist.py
Model CNN do rozpoznawania cyfr ze zbioru MNIST.

Architektura:
- Conv2D
- MaxPooling
- Dense
- Softmax

Uzyskana skuteczność: około 98%.

---

### 4.0_Iris.py
Analiza eksploracyjna danych Iris oraz model drzewa decyzyjnego.

Elementy:
- pairplot (EDA)
- Decision Tree
- wizualizacja drzewa

---

### 5.0_sieci_dwuwarstwowe_update.py
Autorska implementacja wielowarstwowego perceptrona (MLP).

Zastosowane mechanizmy:
- momentum
- adaptive learning rate
- obsługa problemów XOR i Titanic

---

## Wymagane biblioteki
pip install numpy pandas matplotlib scikit-learn tensorflow seaborn

## Przykład uruchomienia
python 3.0_Bayes.py
python 3.0_mnist.py
python 4.0_Iris.py
python 5.0_sieci_dwuwarstwowe_update.py

## Sprawozdanie
Plik PDF zawiera opis działania modeli, analizę wyników oraz wnioski końcowe.

## Autor
Krystian Figlak 100940
Antoni Szczuraszek 102585

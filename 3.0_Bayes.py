import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

#1. PRZYGOTOWANIE DANYCH
# Słownik z danymi symulującymi zachowania użytkowników.
data = {
    'age': [25, 45, 35, 50, 23, 40, 60, 18, 22, 33],
    'time_on_site': [10, 20, 5, 25, 2, 15, 30, 1, 8, 12],
    'subscribed': [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
}
# Konwersja słownika na tabelę biblioteki Pandas.
df = pd.DataFrame(data)

# Podział na cechy (X) oraz etykietę celu (y).
# X zawiera parametry wejściowe, y zawiera informację o tym, czy zakup nastąpił.
X = df.drop('subscribed', axis=1)
y = df['subscribed']

# Podział zbioru na dane treningowe (80%) i testowe (20%).
# random_state=42 zapewnia powtarzalność wyników przy każdym uruchomieniu.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2. KLASYFIKATOR NAIWNY BAYES
# Inicjalizacja modelu GaussianNB (zakłada rozkład normalny cech ciągłych).
nb = GaussianNB()
# Proces uczenia modelu na danych treningowych.
nb.fit(X_train, y_train)
# Wyświetlenie raportu wydajności (precyzja, czułość, F1-score) na danych testowych.
print("Naiwny Bayes Wyniki:\n", classification_report(y_test, nb.predict(X_test)))

#3. KLASYFIKATOR DRZEWO DECYZYJNE
# Inicjalizacja modelu drzewa (buduje logiczne rozgałęzienia na podstawie cech).
dt = DecisionTreeClassifier()
# Proces uczenia drzewa.
dt.fit(X_train, y_train)
# Porównanie wyników predykcji drzewa z faktycznymi danymi testowymi.
print("Drzewo Decyzyjne Wyniki:\n", classification_report(y_test, dt.predict(X_test)))

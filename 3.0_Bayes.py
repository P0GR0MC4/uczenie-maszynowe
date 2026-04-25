import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Symulacja danych subskrybentów (możesz wczytać własny plik .csv)
data = {
    'age': [25, 45, 35, 50, 23, 40, 60, 18, 22, 33],
    'time_on_site': [10, 20, 5, 25, 2, 15, 30, 1, 8, 12],
    'subscribed': [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df.drop('subscribed', axis=1)
y = df['subscribed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naiwny Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
print("Naiwny Bayes Wyniki:\n", classification_report(y_test, nb.predict(X_test)))

# Drzewo Decyzyjne (dodatkowo)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Drzewo Decyzyjne Wyniki:\n", classification_report(y_test, dt.predict(X_test)))

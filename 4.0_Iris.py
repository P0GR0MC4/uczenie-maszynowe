import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

#1. EKSPLORACYJNA ANALIZA DANYCH
# Załadowanie klasycznego zbioru danych Iris (cechy kwiatów: kosaćców).
iris = load_iris()
# Konwersja do formatu DataFrame dla łatwiejszej manipulacji danymi.
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Dodanie kolumny z nazwami gatunków (Setosa, Versicolor, Virginica).
df['species'] = [iris.target_names[i] for i in iris.target]

# Generowanie wykresu macierzowego (Pairplot).
# hue='species' koloruje punkty zależnie od gatunku, co pozwala wzrokowo ocenić, które cechy najlepiej rozdzielają klasy (np. długość płatka).
sns.pairplot(df, hue='species')
plt.suptitle("Analiza EDA - Zbiór Iris", y=1.02)
plt.show()

#2. BUDOWA I WIZUALIZACJA DRZEWA DECYZYJNEGO
# Inicjalizacja modelu drzewa. max_depth=3 zapobiega przeuczeniu.
model = DecisionTreeClassifier(max_depth=3)
# Trenowanie modelu na danych surowych (wymiary płatków) i etykietach (gatunki).
model.fit(iris.data, iris.target)

# Tworzenie graficznej reprezentacji drzewa.
plt.figure(figsize=(10,7))
# plot_tree rysuje schemat blokowy decyzji. 
# filled=True koloruje węzły zgodnie z dominującą klasą.
plot_tree(model, filled=True, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names)
plt.title("Zaprojektowane Drzewo Decyzyjne dla Iris")
plt.show()

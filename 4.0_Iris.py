import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Eksploracyjna Analiza Danych (EDA) - wymagana na 4.0/5.0
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

sns.pairplot(df, hue='species')
plt.suptitle("Analiza EDA - Zbiór Iris", y=1.02)
plt.show()

# 2. Drzewo Decyzyjne
model = DecisionTreeClassifier(max_depth=3)
model.fit(iris.data, iris.target)

plt.figure(figsize=(10,7))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Zaprojektowane Drzewo Decyzyjne dla Iris")
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Eksploracyjna Analiza Danych (EDA)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

sns.pairplot(df, hue='species')
plt.savefig('eda_plot.png') # Wykres do sprawozdania
plt.show()

# 2. Drzewo decyzyjne
model = DecisionTreeClassifier(max_depth=3)
model.fit(iris.data, iris.target)

plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=iris.feature_names)
plt.savefig('decision_tree.png') # Screen do sprawozdania
plt.show()

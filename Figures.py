import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IRIS.csv')

setosa = df[df['species'] == 'Iris-setosa']
versicolor = df[df['species'] == 'Iris-versicolor']
virginica = df[df['species'] == 'Iris-virginica']

# Create a 1x3 grid for subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Figure 1 =================================================================================================
axes[0].scatter(setosa['sepal_length'], setosa['sepal_width'], label='setosa', color='red')
axes[0].scatter(versicolor['sepal_length'], versicolor['sepal_width'], label='versicolor', color='green')
axes[0].scatter(virginica['sepal_length'], virginica['sepal_width'], label='virginica', color='blue')

axes[0].set_title('Figure 1')
axes[0].set_xlabel('sepal_length')
axes[0].set_ylabel('sepal_width')

axes[0].legend()

# Figure 2 ================================================================================================
axes[1].scatter(setosa['sepal_length'], setosa['petal_width'], label='setosa', color='red')
axes[1].scatter(versicolor['sepal_length'], versicolor['petal_width'], label='versicolor', color='green')
axes[1].scatter(virginica['sepal_length'], virginica['petal_width'], label='virginica', color='blue')

axes[1].set_title('Figure 2')
axes[1].set_xlabel('sepal_length')
axes[1].set_ylabel('petal_width')

axes[1].legend()

# Figure 3 ================================================================================================
axes[2].scatter(setosa['sepal_length'], setosa['petal_length'], label='setosa', color='red')
axes[2].scatter(versicolor['sepal_length'], versicolor['petal_length'], label='versicolor', color='green')
axes[2].scatter(virginica['sepal_length'], virginica['petal_length'], label='virginica', color='blue')

axes[2].set_title('Figure 3')
axes[2].set_xlabel('sepal_length')
axes[2].set_ylabel('petal_length')

axes[2].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

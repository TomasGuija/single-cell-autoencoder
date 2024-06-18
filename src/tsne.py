import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne():
    # Cargar el conjunto de datos de ejemplo (iris)
    with open('data/X_test_melanoma.csv') as x:
        ncols = len(x.readline().split(','))

    X = pd.read_csv('data/X_test_melanoma.csv', usecols=range(1, ncols))
    y = pd.read_csv('data/X_test_melanoma.csv', usecols=[0])

    # Aplicar t-SNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    # Visualizar los resultados
    plt.figure()

    for i in np.unique(y):
        plt.scatter(X_tsne[y.values.flatten() == i, 0], X_tsne[y.values.flatten() == i, 1], label=f'Class {i}', s = 5)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    tsne()

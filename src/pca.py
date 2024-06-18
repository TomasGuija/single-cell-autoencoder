import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca():
    """
    Script para realizar un PCA sobre un dataset
    """
    
    # Abrimos el archivo para leer su longitud
    with open('data/X_test_Lung_norm_treatement_timepoint.csv') as x:
        ncols = len(x.readline().split(','))

    # Guardamos la tabla de datos y el nombre de las observaciones
    X = pd.read_csv('data/X_test_Lung_norm_treatement_timepoint.csv', usecols=range(1, ncols))
    y = pd.read_csv('data/X_test_Lung_norm_treatement_timepoint.csv', usecols=[0])

    # Aplicamos PCA
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    # Visualizamos los resultados
    plt.figure()

    # Mostramos cada clase
    for i in np.unique(y):
        plt.scatter(X_r[y.values.flatten() == i, 0], X_r[y.values.flatten() == i, 1], label=f'Class {i}', s = 5)

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('PCA de los datos')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    pca()

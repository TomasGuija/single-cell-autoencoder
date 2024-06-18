import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def prepare_data(dataset):

    """
    Script  para procesar un dataset en csv, filtrando los genes a través de una lista de genes biológicamente relevantes.
    """

    # Leemos los genes biológicamente relevantes
    with open('data/important_genes.txt', 'r') as f:
        important_genes = [line.strip() for line in f]

    # Establecemos una semilla para asegurar la reproducibilidad
    seed = 42

    # Leemos el dataset
    df = pd.read_csv(dataset)
    X = df.values[:, :-1].astype(float) # X guarda la tabla de valores.
    obs = df.values[:, -1] # obs guarda los nombres de las observaciones.
    var_names = df.columns[:-1] # var_names almacena los nombres de las variables.
    var = pd.DataFrame(index=var_names) # dataframe con los nombres de los genes por índice.

    adata = ad.AnnData(X=X, obs=obs, var=var) # creamos un Annotated Data con la tabla de valores X, los nombres de observaciones obs, y las variables var.

    adata = adata[:, adata.var_names.isin(important_genes)] # Filtramos  adata según los genes biológicamente relevantes

    scaler = MinMaxScaler()
    adata.X = scaler.fit_transform(adata.X) # Normalizamos los datos

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(adata.X, adata.obs_names, test_size=0.2, random_state=seed)

    # Obtenemos los nombres de observaciones correspondientes a los índices guardados en Y_train e Y_test.
    train_indeces = obs[np.array(Y_train.values).astype(int)]
    test_indeces = obs[np.array(Y_test.values).astype(int)] 

    # Generamos dataframes de entrenamiento y prueba.
    df_train = pd.DataFrame(X_train, index=train_indeces, columns=adata.var_names)
    df_test = pd.DataFrame(X_test, index=test_indeces, columns=adata.var_names)

    # Guardamos los DataFrames como archivos CSV
    df_train.to_csv('X_train_melanoma.csv')
    df_test.to_csv('X_test_melanoma.csv')

    return X_train, X_test 

if __name__ == "__main__":
    prepare_data("data/melanoma.csv")
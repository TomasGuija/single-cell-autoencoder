import anndata as ad
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def process_data(route):
    """
    Script  para procesar un dataset en .h5ad, filtrando los genes a través de una lista de genes biológicamente relevantes.
    """
        
    # Leemos los genes biológicamente relevantes
    with open('data/important_genes.txt', 'r') as f:
        important_genes = [line.strip() for line in f]
    
    # Leemos el annotated data y lo filtramos según los genes biológicamente relevantes
    data = ad.read_h5ad(route)
    n_vars = data.n_vars
    print(f"El dataset tiene {n_vars} variables.")
    adata = data[:, data.var_names.isin(important_genes)]
    
    # Normalizamos el dataset
    scaler = MinMaxScaler()
    adata.X = scaler.fit_transform(adata.X)

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(adata.X, adata.obs_names, test_size=0.2, random_state=42)

    # Escogemos la clase que usamos para cada observación del dataset
    index_train = adata.obs['treatement_timepoint'][Y_train]
    index_test = adata.obs['treatement_timepoint'][Y_test]

    # Generamos los dataframes de entrenamiento y de prueba
    df_train = pd.DataFrame(X_train, index=index_train, columns=adata.var_names)
    df_test = pd.DataFrame(X_test, index=index_test, columns=adata.var_names)


    # Guardamos los DataFrames como archivos CSV
    df_train.to_csv('X_train_Lung_norm_treatement_timepoint.csv')
    df_test.to_csv('X_test_Lung_norm_treatement_timepoint.csv')

if __name__ == "__main__":
    process_data("data/project_interceptmedml_lung.h5ad")
import optuna
import pandas as pd
import numpy as np
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from src.models.topo_vae import VariationalTopoAE
from src.training import TrainingLoop
from src.topology import TopologicalSignatureDistance
from exp.test_model import test_model
from torch.utils.data import DataLoader


# Función de objetivo para la optimización de Optuna
def objective(trial):
    # Datos de entrenamiento y prueba
    with open('data/X_train_Lung_norm_v2.csv') as x:
        ncols = len(x.readline().split(','))

    X_train = pd.read_csv('data/X_train_Lung_norm_v2.csv', usecols=range(1, ncols))
    X_test = pd.read_csv('data/X_test_Lung_norm_v2.csv', usecols=range(1, ncols))

    # Definir el espacio de búsqueda de hiperparámetros
    layer_size_1 = trial.suggest_int("layer_size_1", 1000, 2200)
    layer_size_2 = trial.suggest_int("layer_size_2", 170, 500)
    activation = trial.suggest_categorical("activation", ['Tanh', 'ReLU'])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_categorical("n_epochs", [50, 75, 100, 125])
    lam = trial.suggest_categorical("lam", [0.75, 1, 1.25, 1.5])

    # Entrenamiento y evaluación del modelo con los hiperparámetros sugeridos
    model = VariationalTopoAE(
        lam=lam,
        ae_kwargs={"input_size": X_train.shape[1], "latent_dim": 100, "layer_size_1": layer_size_1, "layer_size_2": layer_size_2, "activation": activation},
        toposig_kwargs={"match_edges": "symmetric"}
    )

    training_loop = TrainingLoop(
        model, X_train.to_numpy().astype(np.float32), n_epochs, batch_size, learning_rate, weight_decay
    )

    training_loop()

    test_loader = DataLoader(X_test.to_numpy().astype(np.float32), batch_size=32, shuffle=False)
    model.eval()
    topo_distance = TopologicalSignatureDistance({"match_edges": "symmetric"})

    loss = test_model(model, test_loader, topo_distance)

    print(f"Evaluación completada. Pérdida final en conjunto de prueba: {loss:.4f}")

    return loss

if __name__ == '__main__':

    # Ejecutar la optimización bayesiana con Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=75)

    # Imprimir los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:", study.best_params)

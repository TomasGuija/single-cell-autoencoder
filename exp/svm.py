import pandas as pd
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from src.models.topo_vae import VariationalTopoAE

def evaluate_separability(model_path):
    with open('data/X_test_Lung_norm_cell_type.csv') as x:
        ncols = len(x.readline().split(','))

    x_train = pd.read_csv('data/X_train_Lung_norm_cell_type.csv', usecols=range(1, ncols))
    y_train = pd.read_csv('data/X_train_Lung_norm_cell_type.csv', usecols=[0])
    
    x_test = pd.read_csv('data/X_test_Lung_norm_cell_type.csv', usecols=range(1, ncols))
    y_test = pd.read_csv('data/X_test_Lung_norm_cell_type.csv', usecols=[0])

    X_train_np = x_train.to_numpy().astype(np.float32)
    X_test_np = x_test.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy().ravel()
    y_test_np = y_test.to_numpy().ravel()

    model = VariationalTopoAE(lam=1, ae_kwargs={"input_size": x_train.shape[1], "latent_dim": 100, "layer_size_1": 666, "layer_size_2": 256, "activation": 'ReLU'},
                              toposig_kwargs={"match_edges": "symmetric"})
    model.load_state_dict(torch.load(model_path))
    model.eval()

    latent_train = model.encode(torch.tensor(X_train_np)).detach().numpy()
    latent_test = model.encode(torch.tensor(X_test_np)).detach().numpy()

    # Entrenar y evaluar el clasificador en el espacio original
    
    clf_original = SVC(kernel='linear', random_state=42)
    clf_original.fit(X_train_np, y_train_np)
    y_pred_original = clf_original.predict(X_test_np)
    accuracy_original = accuracy_score(y_test_np, y_pred_original)

    print("Evaluación en el espacio original:")
    print(f'Accuracy: {accuracy_original}')
    print(classification_report(y_test_np, y_pred_original)) 
    
    # Entrenar y evaluar el clasificador en el espacio latente
    clf_latent = SVC(kernel='linear', random_state=42)
    clf_latent.fit(latent_train, y_train_np)
    y_pred_latent = clf_latent.predict(latent_test)
    accuracy_latent = accuracy_score(y_test_np, y_pred_latent)

    print("Evaluación en el espacio latente:")
    print(f'Accuracy: {accuracy_latent}')
    print(classification_report(y_test_np, y_pred_latent))


if __name__ == '__main__':
    evaluate_separability('trained_models/topo_vae_Lung_666_256_50epochs.pth')

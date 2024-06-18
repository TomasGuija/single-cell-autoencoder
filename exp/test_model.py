import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from src.models.topo_vae import VariationalTopoAE
from src.models.topo_ae import TopoAE  
from src.topology import TopologicalSignatureDistance

def test_model(model, test_loader, topo_distance):
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    total_topo_loss = 0.0

    model.eval()
    model.to('cuda')
    with torch.no_grad():
        for data in test_loader:
            inputs = data.to('cuda') 
            outputs = model.forward(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()

            latent = model.encode(inputs)

            x_distances = TopoAE._compute_distance_matrix(inputs)
            x_distances = x_distances / x_distances.max()

            latent_distances = TopoAE._compute_distance_matrix(latent)
            latent_distances = latent_distances / model.latent_norm

            total_topo_loss += topo_distance(x_distances, latent_distances)[0]

    average_loss = total_loss / len(test_loader)
    average_topo_loss = total_topo_loss / len(test_loader)
    print(f'Test Loss: {average_loss:.4f}')
    print(f'Topological Loss: {average_topo_loss:.4f}' )

    return average_loss + average_topo_loss

if __name__ == "__main__":
    # Configuración
    model_path = "trained_models/topo_vae_Lung_666_256_50epochs.pth" 


    with open('data/X_test_Lung_norm_cell_type.csv') as x:
        ncols = len(x.readline().split(','))

    X_test = pd.read_csv('data/X_test_Lung_norm_cell_type.csv', usecols=range(1, ncols))

    test_loader = DataLoader(X_test.to_numpy().astype(np.float32), batch_size=32, shuffle=False)

    loaded_model = VariationalTopoAE(lam = 1, ae_kwargs={"input_size": X_test.shape[1], "latent_dim": 100, "layer_size_1":666, "layer_size_2":256, "activation": 'ReLU'},
                                 toposig_kwargs={"match_edges": "symmetric"})
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    topo_distance = TopologicalSignatureDistance({"match_edges": "symmetric"})
    # Evaluación del modelo
    final_loss = test_model(loaded_model, test_loader, topo_distance)

    print(f"Evaluación completada. Pérdida final en conjunto de prueba: {final_loss:.4f}")

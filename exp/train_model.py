import os
import sys
import torch
import pandas as pd
import torch
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

#from src import callbacks
from src.models import topo_ae
from src.models import topo_vae
from src.training import TrainingLoop
import argparse

# Establecer una semilla global para la reproducibilidad
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_model(config):

    # Inicializamos parámetros de entrenamiento
    name = config['name']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    latent_dim = config['latent_dim']
    layer_size_1 = config['layer_size_1']
    layer_size_2 = config['layer_size_2']
    lam = config['lam']
    activation = config['activation']

    # Cargamos el dataset
    with open('data/'+config['dataset']) as x:
        ncols = len(x.readline().split(','))

    X_train = pd.read_csv('data/'+config['dataset'], usecols=range(1, ncols))

    seed = 42
    set_seed(seed)

    #Creamos un modelo nuevo y lo entrenamos
    model = topo_vae.VariationalTopoAE(lam = lam, ae_kwargs = {"input_size":X_train.shape[1], "latent_dim" : latent_dim, "layer_size_1":layer_size_1, "layer_size_2":layer_size_2, "activation":activation},
                                        toposig_kwargs = {"match_edges":"symmetric"})    
    
    training_loop = TrainingLoop(
        model, X_train.to_numpy().astype(np.float32), n_epochs, batch_size, learning_rate, weight_decay
    )

    training_loop()

    #Guardamos el modelo
    torch.save(model.state_dict(), name)
    print("Modelo guardado")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='modelo_entrenado.pth')
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="X_train_Lung_norm_cell_type.csv")
    parser.add_argument("--layer_size_1", type=int, default=666)
    parser.add_argument("--layer_size_2", type=int, default=256)
    parser.add_argument("--lam", type=float, default = 1)
    parser.add_argument("--activation", type=str, default='ReLU')

    args = parser.parse_args()

    config = {'name': args.name, 
              'n_epochs': args.n_epochs, 
              'batch_size': args.batch_size,
              'learning_rate': args.learning_rate,
              'weight_decay': args.weight_decay,
              'latent_dim': args.latent_dim,
              'dataset': args.dataset,
              'layer_size_1': args.layer_size_1,
              'layer_size_2': args.layer_size_2,
              'lam' : args.lam,
              'activation': args.activation
            }
    
    train_model(config)




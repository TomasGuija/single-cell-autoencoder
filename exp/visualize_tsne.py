import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from src.models.topo_vae import VariationalTopoAE
from src.models.topo_ae import TopoAE  

SEED=91
np.random.seed(SEED)

FIGURE_TITLE_SIZE = 18
SUBPLOT_TITLE_SIZE = 14
LEGEND_TITLE_SIZE = 12
FONT_SIZE = 10

def main():   
    plt.rc('font', size=16) # controls default text sizes
    plt.rc('axes', titlesize=SUBPLOT_TITLE_SIZE) # fontsize of the axes title
    plt.rc('axes', labelsize=LEGEND_TITLE_SIZE) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LEGEND_TITLE_SIZE) # fontsize of the tick labels
    plt.rc('ytick', labelsize=LEGEND_TITLE_SIZE) # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_TITLE_SIZE) # legend fontsize
    plt.rc('legend', title_fontsize=SUBPLOT_TITLE_SIZE) # legend fontsize
    plt.rc('figure', titlesize=FIGURE_TITLE_SIZE) # fontsize of the figure title

    with open('data/X_test_melanoma.csv') as x:
        ncols = len(x.readline().split(','))

    x = pd.read_csv('data/X_test_melanoma.csv', usecols=range(1, ncols))
    y = pd.read_csv('data/X_test_melanoma.csv', usecols=[0])

    order_plot = list(np.unique(y))
    plot_label = dict(zip(order_plot, range(len(order_plot))))

    model_path = "trained_models/topo_vae_Optuna_melanoma_15epochs.pth" 

    loaded_model = VariationalTopoAE(lam = 1.5, ae_kwargs={"input_size": x.shape[1], "latent_dim": 100, "layer_size_1":1994, "layer_size_2":281, "activation":'ReLU'},
                                 toposig_kwargs={"match_edges": "symmetric"})
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    df_encoding_immune = convert_to_tsne(x, y, loaded_model)


    fig, axes = plt.subplots(figsize=(15,8))
    df_encoding_tsne = df_encoding_immune.sort_values(by=['cell_type'], key=lambda x: x.map(plot_label))

    palette = sns.color_palette("Set3", len(order_plot))  # Usar la paleta 'Set3'

    sns.scatterplot(data=df_encoding_tsne, x='tsne1', y='tsne2', hue='cell_type', ax=axes, marker='.', s=80, palette=palette)
    axes.axes.get_yaxis().set_visible(False)
    axes.axes.get_xaxis().set_visible(False)

    plt.xlim(df_encoding_immune['tsne1'].min(), df_encoding_immune['tsne1'].max())  # Ajustar límites x
    plt.ylim(df_encoding_immune['tsne2'].min(), df_encoding_immune['tsne2'].max())  # Ajustar límites y

    plt.show()
    plt.legend(title='cell type', loc='upper center', bbox_to_anchor=(.5, -0.01), ncol=7, frameon=False, prop={'size': 16} )

def convert_to_tsne(X, y, model, info=None):
    np.random.seed(SEED)
    # Convertir DataFrame a tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():  # Deshabilitar el cálculo de gradientes
        encoded = model.encode(X_tensor)
    encoded_np = encoded.detach().numpy()  # Obtener un tensor sin gradientes y convertirlo a numpy array
    df_encoding = pd.DataFrame(encoded_np)
    tsne = TSNE()
    df_tsne_array = tsne.fit_transform(np.array(df_encoding))
    df_tsne = pd.DataFrame(df_tsne_array, columns=['tsne1', 'tsne2'])
    df_tsne['cell_type'] = y
    df_tsne['dataset'] = info
    return df_tsne

if __name__ == '__main__':
    main()

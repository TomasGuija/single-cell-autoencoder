import matplotlib.pyplot as plt
import re


def plot_evolution(file_path):
    # Listas para almacenar los valores de pérdida vae y topo
    vae_losses = []
    topo_losses = []

    # Expresión regular para extraer los valores de pérdida
    loss_pattern = r'Epoch \d+ loss: vae: ([\d.]+), topo: ([\d.]+)'

    # Leer el archivo de texto
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Buscar coincidencias con la expresión regular
            match = re.search(loss_pattern, line)
            if match:
                # Extraer los valores de pérdida vae y topo
                vae_loss = float(match.group(1))
                topo_loss = float(match.group(2))
                vae_losses.append(vae_loss)
                topo_losses.append(topo_loss)


    # Verificar si se encontraron valores de pérdida
    if not vae_losses or not topo_losses:
        print("No se encontraron valores de pérdida en el archivo.")
        return

    # Crear la gráfica de VAE Loss
    epochs = range(1, len(vae_losses) + 1)
    plt.plot(epochs, vae_losses, label='VAE Loss')
    plt.title('Entrenamiento: VAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('evolución_Lung_666_256_50epochs_MSE.png')
    plt.show()

    # Crear la gráfica de Topo Loss
    plt.plot(epochs, topo_losses, label='Topo Loss', color='orange')
    plt.title('Entrenamiento: Topo Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('evolución_Lung_666_256_50epochs_TOPO.png')
    plt.show()


if __name__ == '__main__':
    plot_evolution('other/data/evolución_Lung_666_256_50epochs.txt')

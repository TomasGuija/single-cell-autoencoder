import re
import csv
import matplotlib.pyplot as plt

def extract_values():
    # Listas para almacenar los valores de Topological Loss y Test Loss
    topological_loss = []
    test_loss = []

    # Ruta al archivo de texto
    file_path = 'other/data/optuna_Lung.txt'

    # Expresiones regulares para buscar los valores de Topological Loss y Test Loss
    topological_loss_pattern = r"Topological Loss: ([\d\.]+)"
    test_loss_pattern = r"Test Loss: ([\d\.]+)"

    # Abrir el archivo de texto
    with open(file_path, 'r') as file:
        # Leer cada línea del archivo
        for line in file:
            # Buscar coincidencias con la expresión regular de Topological Loss
            topological_loss_match = re.search(topological_loss_pattern, line)
            if topological_loss_match:
                # Extraer y almacenar el valor de Topological Loss
                topological_loss_value = float(topological_loss_match.group(1))
                topological_loss.append(topological_loss_value)
            
            # Buscar coincidencias con la expresión regular de Test Loss
            test_loss_match = re.search(test_loss_pattern, line)
            if test_loss_match:
                # Extraer y almacenar el valor de Test Loss
                test_loss_value = float(test_loss_match.group(1))
                test_loss.append(test_loss_value)

    # Guardar los valores en un archivo CSV
    output_file = 'experiment_losses.csv'
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Topological Loss', 'Test Loss'])
        for t_loss, te_loss in zip(topological_loss, test_loss):
            writer.writerow([t_loss, te_loss])

    print("Valores de Topological Loss y Test Loss guardados en", output_file)

    return topological_loss, test_loss

def plot_values(topological_loss, test_loss):
    # Gráfica para Topological Loss
    plt.figure(figsize=(8, 6))
    plt.plot(topological_loss, linewidth=2, label='Topological Loss')
    plt.xlabel('Experimento')
    plt.ylabel('Topological Loss')
    plt.title('Evolución de Topological Loss en los experimentos')
    plt.legend()
    plt.grid(True)
    plt.savefig('topological_loss_plot_Lung.png')
    plt.close()

    # Gráfica para Test Loss
    plt.figure(figsize=(8, 6))
    plt.plot(test_loss, linewidth=2, label='Test Loss')
    plt.xlabel('Experimento')
    plt.ylabel('Test Loss')
    plt.title('Evolución de Test Loss en los experimentos')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_loss_plot_Lung.png')
    plt.close()

if __name__ == '__main__':
    topological_loss, test_loss = extract_values()
    plot_values(topological_loss, test_loss)

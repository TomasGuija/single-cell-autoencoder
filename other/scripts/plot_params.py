import pandas as pd
import matplotlib.pyplot as plt

def plot_values(df):
    # Plot solo para batch_size
    plt.figure(figsize=(10, 5))
    plt.plot(df['batch_size'], marker='o', linestyle='', markersize=5, label='batch_size')
    plt.xlabel('Experimento')
    plt.ylabel('batch_size')
    plt.title('Valores de batch_size para cada experimento')
    plt.legend()
    plt.grid(True)
    plt.savefig("batch_size_plot.png")
    plt.show()

    # Plot solo para learning_rate
    plt.figure(figsize=(10, 5))
    plt.plot(df['learning_rate'], linestyle='-', linewidth=2, label='learning_rate')
    plt.xlabel('Experimento')
    plt.ylabel('learning_rate')
    plt.title('Valores de learning_rate para cada experimento')
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_rate_plot.png")
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('other/experiment_params.csv')
    plot_values(df)

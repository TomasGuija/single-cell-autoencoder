import pandas as pd

def count_cell_types(csv_file):
    """
    Script para leer un dataset y mostrar el número de observaciones por cada clase.
    """

    data = pd.read_csv(csv_file)
    
    # Etiquetas en la última columna
    cell_type_column = data.columns[-1]
    
    cell_type_counts = data[cell_type_column].value_counts()
    
    total_samples = cell_type_counts.sum()
    
    # Imprimir la tabla
    print(f"{'Tipo de Célula':<15} {'Número de Muestras':<20}")
    print("-" * 35)
    for cell_type, count in cell_type_counts.items():
        print(f"{cell_type:<15} {count:<20}")
    print("-" * 35)
    print(f"{'Total':<15} {total_samples:<20}")


if __name__ == '__main__':
    # Ejemplo de uso
    csv_file = 'data/melanoma.csv'
    count_cell_types(csv_file)

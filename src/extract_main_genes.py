def extract():

    """
    Código creado para extraer los genes biológicamente relevantes a partir del output proporcionado por David Tool.
    """
    # Abrimos el archivo original
    with open('data/conv_1.txt', 'r') as f:
        lines = f.readlines()

    # Abrimos el archivo de salida, y por cada línea del archivo original extraemos el nombre del gen biológicamente relevante
    with open('data/important_genes.txt', 'w') as f_out:
        for line in lines:
            columns = line.split('\t')
            f_out.write(columns[1] + '\n')

if __name__ == '__main__':
    extract()

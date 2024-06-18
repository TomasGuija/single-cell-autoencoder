import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim, layer_size_1, layer_size_2, activation):
        
        """
        Estructura del autoencoder en que se basará el autoencoder variacional topológico.

        Args:
            input_size: tamaño del input
            latent_dim: dimensión del espacio latente
            layer_size_1: tamaño de la primera capa oculta
            layer_size_2: tamaño de la segunda capa oculta
            activation: función de activación no lineal  

        La estructura del autoencoder es simétrica.
        Se toma el MSE como error de reconstrucción.
        """

        super(SimpleAutoencoder, self).__init__()
        
        
        if activation == 'Tanh':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, layer_size_1),
                nn.Tanh(),
                nn.Linear(layer_size_1, layer_size_2),
                nn.Tanh(),
                nn.Linear(layer_size_2, latent_dim)
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, layer_size_2),
                nn.Tanh(),
                nn.Linear(layer_size_2, layer_size_1),
                nn.Tanh(),
                nn.Linear(layer_size_1, input_size)
            )
        elif activation == 'ReLU':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, layer_size_1),
                nn.ReLU(),
                nn.Linear(layer_size_1, layer_size_2),
                nn.ReLU(),
                nn.Linear(layer_size_2, latent_dim)
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, layer_size_2),
                nn.ReLU(),
                nn.Linear(layer_size_2, layer_size_1),
                nn.ReLU(),
                nn.Linear(layer_size_1, input_size)
            )


        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def compute_loss(self, x):
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error

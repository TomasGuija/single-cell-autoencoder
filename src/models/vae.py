from .simple_autoencoder import SimpleAutoencoder
import torch.nn as nn
import torch

class Variational_autoencoder(SimpleAutoencoder):
    def __init__(self, input_size, latent_dim, layer_size_1, layer_size_2, activation):
        """
        Autoencoder varacional basado en SimpleAutoencoder
        """
        super(Variational_autoencoder, self).__init__(input_size, latent_dim, layer_size_1, layer_size_2, activation)

        # Nuevas capas para el encoder que producen la media y la desviación estándar
        self.encoder_mu = nn.Linear(latent_dim, latent_dim)
        self.encoder_log_var = nn.Linear(latent_dim, latent_dim)

        self.reconst_error = nn.MSELoss()

    def get_parameters(self, x):
        """
        Función que devuelve los valores de media y varianza aprendidos
        """
        mu = self.encoder_mu(x)
        log_var = self.encoder_log_var(x)
        return mu, log_var
    
    def encode(self, x):
        return super().encode(x)
    
    def decode(self, x):
        """
        Se muestrea a partir de la media y varianza aprendidas y se proporciona una salida
        """
        mu, log_var = self.get_parameters(x)
        h = self.reparameterize(mu, log_var) 
        return super().decode(h)   

    def reparameterize(self, mu, log_var):
        """
        Reparametrización para muestreo estocástico
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.get_parameters(super().encode(x))
        latent = self.reparameterize(mu, log_var)
        x_reconst = super().decode(latent)
        return x_reconst

    def compute_loss(self, x):
        mu, log_var = self.get_parameters(super().encode(x))
        x_reconst = self.forward(x)

        # Pérdida de reconstrucción
        reconst_error = self.reconst_error(x, x_reconst)

        # Pérdida KL para regularizar la distribución latente
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Combina ambas pérdidas
        total_loss = reconst_error + kl_loss

        return total_loss

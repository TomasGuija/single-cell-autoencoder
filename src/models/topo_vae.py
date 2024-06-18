import torch
import torch.nn as nn
from ..topology import TopologicalSignatureDistance
from .vae import Variational_autoencoder
from .base import AutoencoderModel

class VariationalTopoAE(AutoencoderModel):
    def __init__(self, lam=1., ae_kwargs=None, toposig_kwargs=None):
        """
        Autoencoder topológico variacional con estructura basada en Variational_autoencoder.
        Args:
            lam: peso del error topológico sobre la función de pérdida. 
            ae_kwargs: argumentos a pasar a Variational_autoencoder
            toposig_kwargs: argumentos a pasar a TopologicalSignatureDistance
        """
                
        super().__init__()
        self.lam = lam
        ae_kwargs = ae_kwargs if ae_kwargs else {}
        toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        self.variational_autoencoder = Variational_autoencoder(**ae_kwargs)
        self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1), requires_grad=True)
        
        self.activation = ae_kwargs.get("activation") 


        self._initialize_weights()

    def _initialize_weights(self):
        """
        Método creado para inicializar los pesos y los sesgos. 
        Se aplica la inicialización Kaiming Uniforme a las capas lineales con ReLU y 
        la inicialización Xavier Uniforme a las capas lineales con Tanh.
        Todos los sesgos se inicializan a 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation and self.activation.lower() == 'relu':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.activation and self.activation.lower() == 'tanh':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)  # Por defecto usa Xavier para otras activaciones

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        return self.variational_autoencoder.encode(x)

    def decode(self, z):
        return self.variational_autoencoder.decode(z)
    
    def forward(self, x):
        return self.variational_autoencoder.forward(x)

    def compute_loss(self, x):
        """
        Produce la función de pérdida del autoencoder topológico.

        Calcula el error cometido por el autoencoder variacional y la distancia topológica entre el espacio original y el latente para un lote de datos (input x).

        Devuelve la suma de ambos errores, multiplicando la distancia topológica por un factor lam.
        """
        latent = self.variational_autoencoder.encode(x)

        x_distances = self._compute_distance_matrix(x)
        x_distances = x_distances / x_distances.max()

        vae_latent_distances = self._compute_distance_matrix(latent)
        vae_latent_distances = vae_latent_distances / self.latent_norm

        topo_loss, _ = self.topo_sig(x_distances, vae_latent_distances)

        vae_loss = self.variational_autoencoder.compute_loss(x)

        total_loss = vae_loss + self.lam * topo_loss

        return total_loss, vae_loss, topo_loss

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances


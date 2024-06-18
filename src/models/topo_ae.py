from ..topology import TopologicalSignatureDistance
import torch
from .base import AutoencoderModel
from .simple_autoencoder import SimpleAutoencoder

class TopoAE(AutoencoderModel):

    def __init__(self, lam=1., ae_kwargs=None, toposig_kwargs=None):
        """
        Autoencoder topológico con estructura basada en SimpleAutoencoder.
        Args:
            lam: peso del error topológico sobre la función de pérdida. 
            ae_kwargs: argumentos a pasar a SimpleAutoencoder
            toposig_kwargs: argumentos a pasar a TopologicalSignatureDistance
        """
        super().__init__()
        self.lam = lam
        ae_kwargs = ae_kwargs if ae_kwargs else {}
        toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        self.autoencoder = SimpleAutoencoder(**ae_kwargs)
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),requires_grad=True)

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances
    
    def forward(self, x):
        return self.autoencoder.forward(x)

    def compute_loss(self, x):
        """
        Produce la función de pérdida del autoencoder topológico.

        Calcula el error de reconstrucción (MSE) y la distancia topológica entre el espacio original y el latente para un lote de datos (input x).

        Devuelve la suma de ambos errores, multiplicando la distancia topológica por un factor lam.
        """
        latent = self.autoencoder.encode(x)

        x_distances = self._compute_distance_matrix(x)
        x_distances = x_distances / x_distances.max()

        latent_distances = self._compute_distance_matrix(latent)
        latent_distances = latent_distances / self.latent_norm

        ae_loss = self.autoencoder.compute_loss(x)

        topo_error, _ = self.topo_sig(x_distances, latent_distances)

        # Normalizamos la distancia topológica en función del tamaño del lote
        dimensions = x.size()
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size) 
        loss = ae_loss + self.lam * topo_error

        return loss

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

import numpy as np
import torch.nn as nn
import torch

class UnionFind:
    '''
    Implementación de la clase Union Find. 
    '''

    def __init__(self, n_vertices):
        '''
        Inicializa una estructura donde cada vértice es un conjunto disjunto.
        '''

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Busca y devuelve el representante o raíz del conjunto al que pertenece u.
        '''

        if self._parent[u] == u:
            return u
        else:
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Une el vértice u a la componente del vértice v. Busca la raíz de u y hace que apunte a la raíz de v.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Busca componentes que sean sus propios padres (raíces de sus conjuntos).
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    """
    Cálculo de la homología persistente dada una matriz de distancias
    """
    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)


        # Se ordenan las distancias de menor a mayor
        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        persistence_pairs = []

        # Iteramos sobre las aristas
        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Juntamos las componentes de los vértices corresponodientes a la arista
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))
        return np.array(persistence_pairs)
    

class TopologicalSignatureDistance(nn.Module):
    """
    Distancia topológica 
    """

    def __init__(self, sort_selected=False, match_edges=None):

        super().__init__()

        self.match_edges = match_edges

        self.signature_calculator = PersistentHomologyCalculation()


    def _get_pairings(self, distances):

        """
        Obtenemos las parejas de persistencia
        """
        pairs_0 = self.signature_calculator(
            distances.detach().cpu().numpy())
        return pairs_0

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        """
        Distancias correspondientes a los pares de persistencia
        """
        pairs_0 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """
        Calcula el error cuadrático entre dos firmas topológicas
        """
        return ((signature1 - signature2)**2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        """
        Cuenta el número de pares coincidentes entre dos conjuntos de pares de persistencia.
        """
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))


    def forward(self, distances1, distances2):
        """Devuelve la distancia topológica entre dos matrices de distancias.

        Args:
            distances1: matrz de distancias en el espacio 1
            distances2: matriz de distancias en el espacio 2

        Devuelve:
            distancia, dict(output opcional)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1, pairs2)
        }        

        # Dependiendo del valor de match_edges se hace una cosa u otra
        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)
        # En nuestro caso utilizaremos el simétrico
        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components


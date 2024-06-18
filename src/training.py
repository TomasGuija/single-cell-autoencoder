import torch
from torch.utils.data import DataLoader, TensorDataset

#Copiado del paper, ligeras modificaciones

class TrainingLoop():
    """
    Entrenamiento del modelo utilizando un dataset
    """

    def __init__(self, model, dataset, n_epochs, batch_size, learning_rate,
                 weight_decay=1e-5, device='cuda', callbacks=None):

        self.model = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.callbacks = callbacks if callbacks else []

    def __call__(self):
        """Ejecutar el bucle de entrenamiento"""
        model = self.model
        dataset = self.dataset
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True)
        
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=0)
        
        epoch = 1
        loss_vae_values = []
        loss_topo_values = []
        for epoch in range(1, n_epochs+1):
            print(f"Epoch {epoch}/{n_epochs}")
            epoch_vae_loss = 0.0
            epoch_topo_loss = 0.0
            for batch, data in enumerate(train_loader):
                if self.device == 'cuda':
                    data = data.cuda(non_blocking=True)
                        
                # Ponemos el modelo en modo de entrenamiento y se calcula la pérdida
                model.train()
                model = model.to(self.device)
                loss, vae_loss, topo_loss = self.model.compute_loss(data)
                epoch_vae_loss += vae_loss
                epoch_topo_loss += topo_loss
                
                #Regularización L1
                l1_regularization = torch.tensor(0., requires_grad=True).to(self.device)
                for param in model.parameters():
                    l1_regularization += torch.norm(param, p=1)
                loss += self.weight_decay * l1_regularization

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_epoch_vae_loss = epoch_vae_loss / len(train_loader)
            avg_epoch_topo_loss = epoch_topo_loss / len(train_loader)
            #avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch} loss: vae: {avg_epoch_vae_loss}, topo: {avg_epoch_topo_loss}")
            loss_vae_values.append(avg_epoch_vae_loss.item())  # Añadir el valor de la pérdida a la lista
            loss_topo_values.append(avg_epoch_topo_loss.item())  # Añadir el valor de la pérdida a la lista

        save_loss_values(loss_vae_values, 'reports/epoch_vae_losses_2.csv')
        save_loss_values(loss_topo_values, 'reports/epoch_topo_losses_2.csv')
        return epoch
    
def save_loss_values(loss_values, route):
    import csv

    # Ruta del archivo CSV
    csv_file = route

    # Escribir los valores de pérdida en el archivo CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        for epoch, loss in enumerate(loss_values, start=1):
            writer.writerow([epoch, loss])
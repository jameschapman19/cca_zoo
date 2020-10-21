import copy

import torch
from sklearn.cross_decomposition import CCA
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import cca_zoo.DCCAE
import cca_zoo.DVCCA
import cca_zoo.plot_utils


class Wrapper:
    """
    This is a wrapper class for Deep CCA
    We create an instance with a method and number of latent dimensions.

    The class has a number of methods intended to align roughly with the linear Wrapper:

    fit(): gives us train correlations and stores the variables needed for out of sample prediction as well as some
    method-specific variables

    predict_corr(): allows us to predict the out of sample correlation for supplied views

    predict_view(): allows us to predict a reconstruction of missing views from the supplied views

    transform_view(): allows us to transform given views to the latent variable space

    recon_loss(): gets the reconstruction loss for out of sample data - if the model has an autoencoder piece
    """

    def __init__(self, latent_dims: int = 2, learning_rate=1e-3, epoch_num: int = 1, batch_size: int = 16,
                 method: str = 'DCCAE', loss_type: str = 'cca', lam=0, private: bool = False,
                 patience: int = 10, both_encoders: bool = True, hidden_layer_sizes_1: list = None,
                 hidden_layer_sizes_2: list = None,
                 model_1='fcn', model_2='fcn'):
        self.latent_dims = latent_dims
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        # Default - may change during training due to needing batch size greater than 1
        self.batch_size = batch_size
        # the regularization parameter of the network
        # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
        self.method = method
        self.both_encoders = both_encoders
        self.private = private
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.hidden_layer_sizes_1 = hidden_layer_sizes_1
        self.hidden_layer_sizes_2 = hidden_layer_sizes_2
        self.lam = lam
        self.model_1 = model_1
        self.model_2 = model_2

    def fit(self, X_train, Y_train):

        # Split the subjects randomly into train and validation
        num_subjects = X_train.shape[0]
        all_inds = np.arange(num_subjects)
        np.random.shuffle(all_inds)
        train_inds, val_inds = np.split(all_inds, [int(round(0.8 * num_subjects, 0))])
        X_val = X_train[val_inds]
        Y_val = Y_train[val_inds]
        X_train = X_train[train_inds]
        Y_train = Y_train[train_inds]
        # Remove the training mean from train and validation to avoid leakage
        self.X_mean = X_train.mean(axis=0)
        self.Y_mean = Y_train.mean(axis=0)
        X_train -= self.X_mean
        Y_train -= self.Y_mean
        X_val -= self.X_mean
        Y_val -= self.Y_mean

        # transform to a torch tensor dataset
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))  # create your datset
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))  # create your datset
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # For CCA loss functions, we require that the number of samples in each batch is greater than the number of
        # latent dimensions. This attempts to alter the batch size to fulfil this condition
        while X_train.shape[0] % self.batch_size < self.latent_dims:
            self.batch_size += 1

        # First we get the model class.
        # These have a forward method which takes data inputs and outputs the variables needed to calculate their
        # respective loss. The models also have loss functions as methods but we can also customise the loss by calling
        # a_loss_function(model(data))
        if self.method == 'DCCAE':
            self.model = cca_zoo.DCCAE.DCCAE(input_size_1=X_train.shape[-1], input_size_2=Y_train.shape[-1],
                                             hidden_layer_sizes_1=self.hidden_layer_sizes_1,
                                             hidden_layer_sizes_2=self.hidden_layer_sizes_2, lam=self.lam,
                                             latent_dims=self.latent_dims, loss_type=self.loss_type,
                                             model_1=self.model_1, model_2=self.model_2)
        elif self.method == 'DVCCA':
            self.model = cca_zoo.DVCCA.DVCCA(input_size_1=X_train.shape[-1], input_size_2=Y_train.shape[-1],
                                             hidden_layer_sizes_1=self.hidden_layer_sizes_1,
                                             hidden_layer_sizes_2=self.hidden_layer_sizes_2,
                                             both_encoders=self.both_encoders, latent_dims=self.latent_dims,
                                             private=self.private)
        model_params = sum(p.numel() for p in self.model.parameters())
        best_model = copy.deepcopy(self.model.state_dict())
        print("Number of model parameters {}".format(model_params))
        self.model.double().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        min_val_loss = self.latent_dims
        epochs_no_improve = 0
        early_stop = False
        all_train_loss = []
        all_val_loss = []

        for epoch in range(self.epoch_num):
            if early_stop == False:
                epoch_train_loss = self.train_epoch(train_dataloader)
                print('====> Epoch: {} Average train loss: {:.4f}'.format(
                    epoch, epoch_train_loss))
                epoch_val_loss = self.val_epoch(val_dataloader)
                print('====> Epoch: {} Average val loss: {:.4f}'.format(
                    epoch, epoch_val_loss))

                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    print('Min loss %0.2f' % min_val_loss)
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == self.patience:
                        print('Early stopping!')
                        early_stop = True
                        self.model.load_state_dict(best_model)

                all_train_loss.append(epoch_train_loss)
                all_val_loss.append(epoch_val_loss)
        cca_zoo.plot_utils.plot_training_loss(all_train_loss, all_val_loss)

        if self.method == 'DCCAE':
            self.train_correlations = self.predict_corr(X_train, Y_train, train=True)
        elif self.method =='DVCCA':
            if self.both_encoders:
                self.train_correlations = self.predict_corr(X_train, Y_train, train=True)

        self.train_recon_loss_x, self.train_recon_loss_y = self.recon_loss(X_train, Y_train)

        return self

    def train_epoch(self, train_dataloader: torch.utils.data.DataLoader):
        self.model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            model_outputs = self.model(x, y)
            loss = self.model.loss(x, y, *model_outputs)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        return train_loss / len(train_dataloader)

    def val_epoch(self, val_dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch_idx, (x, y) in enumerate(val_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                model_outputs = self.model(x, y)
                loss = self.model.loss(x, y, *model_outputs)
                total_val_loss += loss.item()
        return total_val_loss / len(val_dataloader)

    def predict_corr(self, X_test, Y_test, train=False):
        X_test -= self.X_mean
        Y_test -= self.Y_mean
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))  # create your datset
        test_dataloader = DataLoader(test_dataset, batch_size=100)
        z_x = np.empty((0, self.latent_dims))
        z_y = np.empty((0, self.latent_dims))
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                if self.method == 'DCCAE':
                    z_x_batch, z_y_batch, recon_x_batch, recon_y_batch = self.model(x, y)
                elif self.method == 'DVCCA':
                    if self.both_encoders:
                        if self.private:
                            recon_batch_1, recon_batch_2, z_x_batch, logvar_x, z_y_batch, logvar_y, _, _, _, _ = self.model(x, y)
                        else:
                            recon_batch_1, recon_batch_2, z_x_batch, logvar_x, z_y_batch, logvar_y = self.model(x, y)
                    else:
                        print('No correlation method for single encoding')
                        return
                z_x = np.append(z_x, z_x_batch.detach().cpu().numpy(), axis=0)
                z_y = np.append(z_y, z_y_batch.detach().cpu().numpy(), axis=0)
        if train:
            self.cca = CCA(n_components=self.latent_dims)
            view_1, view_2 = self.cca.fit_transform(z_x, z_y)
        else:
            view_1, view_2 = self.cca.transform(np.array(z_x), np.array(z_y))
        correlations = np.diag(np.corrcoef(view_1, view_2, rowvar=False)[:self.latent_dims, self.latent_dims:])
        return correlations

    def transform_view(self, X_new=None, Y_new=None):
        if X_new is not None:
            X_new -= self.X_mean
            tensor_x_new = torch.from_numpy(X_new)
        if Y_new is not None:
            Y_new -= self.Y_mean
            tensor_y_new = torch.from_numpy(Y_new)
        if self.method == 'DCCAE':
            if X_new is not None:
                U_new = self.model.encode_1(tensor_x_new)
            if Y_new is not None:
                V_new = self.model.encode_2(tensor_y_new)
        elif self.method == 'DVCCA':
            if X_new is not None:
                U_new = self.model.encode_1(tensor_x_new)[0]
            if Y_new is not None:
                V_new = self.model.encode_2(tensor_y_new)[0]
        if X_new is not None and Y_new is not None:
            return U_new / np.linalg.norm(U_new, axis=0, keepdims=True), V_new / np.linalg.norm(V_new, axis=0,
                                                                                                keepdims=True)
        if X_new is not None and Y_new is None:
            return U_new / np.linalg.norm(U_new, axis=0, keepdims=True), None
        if X_new is None and Y_new is not None:
            return None, V_new / np.linalg.norm(V_new, axis=0, keepdims=True)

    def predict_view(self, X_new=None, Y_new=None):
        U_new, V_new = self.transform_view(X_new=X_new, Y_new=Y_new)
        if self.method == 'DCCAE':
            if U_new is not None:
                Y_pred = self.model.decode_2(U_new)
                X_pred = X_new
            if V_new is not None:
                X_pred = self.model.decode_1(V_new)
                Y_pred = Y_new
        elif self.method == 'DVCCA':
            if U_new is not None:
                Y_pred = self.model.decode_2(U_new)
                X_pred = X_new
            if V_new is not None:
                X_pred = self.model.decode_1(V_new)
                Y_pred = Y_new
        return X_pred, Y_pred

    def recon_loss(self, X_new, Y_new):
        X_new -= self.X_mean
        Y_new -= self.Y_mean
        test_dataset = TensorDataset(torch.tensor(X_new), torch.tensor(Y_new))  # create your datset
        test_dataloader = DataLoader(test_dataset, batch_size=100)
        with torch.no_grad():
            recon_loss_x = 0
            recon_loss_y = 0
            for batch_idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                if self.method == 'DCCAE':
                    z_x, z_y, recon_x, recon_y = self.model(x, y)
                if self.method == 'DGCCAE':
                    z_x, z_y, recon_x, recon_y = self.model(x, y)
                elif self.method == 'DVCCA':
                    model_outputs = self.model(x, y)
                    recon_x = model_outputs[0]
                    recon_y=model_outputs[1]
                recon_loss_x += F.mse_loss(recon_x, x, reduction='sum').detach().cpu().numpy() / x.shape[0]
                recon_loss_y += F.mse_loss(recon_y, y, reduction='sum').detach().cpu().numpy() / y.shape[0]
        return recon_loss_x, recon_loss_y

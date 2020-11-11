import copy

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cross_decomposition import CCA

import cca_zoo.DCCA
import cca_zoo.DCCAE
import cca_zoo.DVCCA
import cca_zoo.plot_utils
import DeCCA


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
                 patience: int = 0, both_encoders: bool = True, hidden_layer_sizes_1: list = None,
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

    def process_training_data(self, *args):
        # Split the subjects randomly into train and validation
        num_subjects = args[0].shape[0]
        all_inds = np.arange(num_subjects)
        np.random.shuffle(all_inds)
        train_inds, val_inds = np.split(all_inds, [int(round(0.8 * num_subjects, 0))])
        self.dataset_list_train = []
        self.dataset_list_val = []
        self.dataset_means = []
        for i, dataset in enumerate(args):
            self.dataset_means.append(dataset[train_inds].mean(axis=0))
            self.dataset_list_train.append(dataset[train_inds] - self.dataset_means[i])
            self.dataset_list_val.append(dataset[val_inds] - self.dataset_means[i])

        # For CCA loss functions, we require that the number of samples in each batch is greater than the number of
        # latent dimensions. This attempts to alter the batch size to fulfil this condition
        while num_subjects % self.batch_size < self.latent_dims:
            self.batch_size += 1

    def fit(self, *args):
        self.process_training_data(*args)

        # transform to a torch tensor dataset
        train_dataset = TensorDataset(
            *[torch.tensor(dataset) for dataset in self.dataset_list_train])  # create your datset
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        val_dataset = TensorDataset(*[torch.tensor(dataset) for dataset in self.dataset_list_val])
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # First we get the model class.
        # These have a forward method which takes data inputs and outputs the variables needed to calculate their
        # respective loss. The models also have loss functions as methods but we can also customise the loss by calling
        # a_loss_function(model(data))
        if self.method == 'DCCAE':
            self.model = cca_zoo.DCCAE.DCCAE(input_size_1=self.dataset_list_train[0].shape[-1],
                                             input_size_2=self.dataset_list_train[1].shape[-1],
                                             hidden_layer_sizes_1=self.hidden_layer_sizes_1,
                                             hidden_layer_sizes_2=self.hidden_layer_sizes_2, lam=self.lam,
                                             latent_dims=self.latent_dims, loss_type=self.loss_type,
                                             model_1=self.model_1, model_2=self.model_2)
        elif self.method == 'DVCCA':
            self.model = cca_zoo.DVCCA.DVCCA(input_size_1=self.dataset_list_train[0].shape[-1],
                                             input_size_2=self.dataset_list_train[1].shape[-1],
                                             hidden_layer_sizes_1=self.hidden_layer_sizes_1,
                                             hidden_layer_sizes_2=self.hidden_layer_sizes_2,
                                             both_encoders=self.both_encoders, latent_dims=self.latent_dims,
                                             private=self.private)
        elif self.method == 'DCCA':
            self.model = cca_zoo.DCCA.DCCA(input_size_1=self.dataset_list_train[0].shape[-1],
                                           input_size_2=self.dataset_list_train[1].shape[-1],
                                           hidden_layer_sizes_1=self.hidden_layer_sizes_1,
                                           hidden_layer_sizes_2=self.hidden_layer_sizes_2, lam=self.lam,
                                           latent_dims=self.latent_dims, loss_type=self.loss_type,
                                           model_1=self.model_1, model_2=self.model_2)
        elif self.method == 'DeCCA':
            self.model = DeCCA.DeCCA(input_size_1=self.dataset_list_train[0].shape[-1],
                                     input_size_2=self.dataset_list_train[1].shape[-1],
                                     input_size_c=self.dataset_list_train[2].shape[-1],
                                     hidden_layer_sizes_1=self.hidden_layer_sizes_1,
                                     hidden_layer_sizes_2=self.hidden_layer_sizes_2,
                                     latent_dims=self.latent_dims, loss_type=self.loss_type,
                                     model_1=self.model_1, model_2=self.model_2)
        elif self.method == 'DeLCCA':
            self.model = DeCCA.DeCCA(input_size_1=self.dataset_list_train[0].shape[-1],
                                     input_size_2=self.dataset_list_train[1].shape[-1],
                                     input_size_c=self.dataset_list_train[2].shape[-1],
                                     hidden_layer_sizes_1=self.hidden_layer_sizes_1,
                                     hidden_layer_sizes_2=self.hidden_layer_sizes_2,
                                     latent_dims=self.latent_dims, loss_type=self.loss_type,
                                     model_1=self.model_1, model_2=self.model_2)

        model_params = sum(p.numel() for p in self.model.parameters())
        best_model = copy.deepcopy(self.model.state_dict())
        print("Number of model parameters {}".format(model_params))
        self.model.double().to(self.device)
        min_val_loss = self.latent_dims
        epochs_no_improve = 0
        early_stop = False
        all_train_loss = []
        all_val_loss = []

        for epoch in range(1, self.epoch_num + 1):
            if early_stop == False:
                epoch_train_loss = self.train_epoch(train_dataloader)
                print('====> Epoch: {} Average train loss: {:.4f}'.format(
                    epoch, epoch_train_loss))
                epoch_val_loss = self.val_epoch(val_dataloader)
                print('====> Epoch: {} Average val loss: {:.4f}'.format(
                    epoch, epoch_val_loss))

                if epoch_val_loss < min_val_loss or epoch == 1:
                    min_val_loss = epoch_val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    print('Min loss %0.2f' % min_val_loss)
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == self.patience and self.patience > 0:
                        print('Early stopping!')
                        early_stop = True
                        self.model.load_state_dict(best_model)

                all_train_loss.append(epoch_train_loss)
                all_val_loss.append(epoch_val_loss)
        cca_zoo.plot_utils.plot_training_loss(all_train_loss, all_val_loss)

        if self.method in ['DCCA', 'DCCAE', 'DGCCA', 'DGCCAE', 'DeCCA', 'DeLCCA']:
            self.train_correlations = self.predict_corr(*self.dataset_list_train, train=True)
        elif self.method == 'DVCCA':
            if self.both_encoders:
                self.train_correlations = self.predict_corr(*self.dataset_list_train, train=True)
        return self

    def train_epoch(self, train_dataloader: torch.utils.data.DataLoader):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_dataloader):
            data = [d.to(self.device) for d in list(data)]
            loss = self.model.update_weights(*data)
            train_loss += loss.item()
        return train_loss / len(train_dataloader)

    def val_epoch(self, val_dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch_idx, data in enumerate(val_dataloader):
                data = [d.to(self.device) for d in list(data)]
                loss = self.model.loss(*data)
                total_val_loss += loss.item()
        return total_val_loss / len(val_dataloader)

    def predict_corr(self, *args, train=False):
        z_list = self.transform_view(*args, train=train)
        if train:
            self.cca = CCA(n_components=self.latent_dims)
            view_1, view_2 = self.cca.fit_transform(z_list[0], z_list[1])
        else:
            view_1, view_2 = self.cca.transform(np.array(z_list[0]), np.array(z_list[1]))
        correlations = np.diag(np.corrcoef(view_1, view_2, rowvar=False)[:self.latent_dims, self.latent_dims:])
        return correlations

    def transform_view(self, *args, train=False):
        dataset_list_test = [arg if train else arg - self.dataset_means[i] for i, arg in enumerate(args)]
        test_dataset = TensorDataset(*[torch.tensor(dataset) for dataset in dataset_list_test])
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        z_list = [np.empty((0, self.latent_dims)) for _ in range(len(args))]
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader):
                data = [d.to(self.device) for d in list(data)]
                if self.method in ['DCCA', 'DCCAE', 'DGCCA', 'DGCCAE', 'DeCCA', 'DeLCCA']:
                    z = self.model.encode(*data)
                elif self.method == 'DVCCA':
                    if self.both_encoders:
                        z_1 = self.model.reparameterize(*self.model.encode_1(data[0]))
                        z_2 = self.model.reparameterize(*self.model.encode_2(data[1]))
                        z = (z_1, z_2)
                    else:
                        print('No correlation method for single encoding')
                        return
                z_list = [np.append(z_list[i], z_i.detach().cpu().numpy(), axis=0) for i, z_i in enumerate(z)]
        return z_list

    def predict_view(self, *args, train=False):
        dataset_list_test = [arg if train else arg - self.dataset_means[i] for i, arg in enumerate(args)]
        test_dataset = TensorDataset(*[torch.tensor(dataset) for dataset in dataset_list_test])
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        x_list = [np.empty((0, arg.shape[1:])) for i, arg in enumerate(args)]
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader):
                data = [d.to(self.device) for d in list(data)]
                if self.method in ['DCCAE', 'DGCCAE']:
                    x = self.model(*data)
                elif self.method == 'DVCCA':
                    if self.both_encoders:
                        x = self.model(*data)
                    else:
                        print('No reconstruction method for single encoding')
                        return
                x_list = [np.append(x_i, x[i].detach().cpu().numpy(), axis=0) for i, x_i in enumerate(x_list)]
        return x_list

import copy

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import cca_zoo.plot_utils
from sklearn.cross_decomposition import CCA
from cca_zoo.configuration import Config


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

    def __init__(self, config: Config = Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def fit(self, *args):
        self.process_training_data(*args)

        # transform to a torch tensor dataset
        train_dataset = TensorDataset(
            *[torch.tensor(dataset) for dataset in self.dataset_list_train])  # create your datset
        val_dataset = TensorDataset(*[torch.tensor(dataset) for dataset in self.dataset_list_val])
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))

        self.config.input_sizes = [dataset.shape[-1] for dataset in self.dataset_list_train]

        # First we get the model class.
        # These have a forward method which takes data inputs and outputs the variables needed to calculate their
        # respective loss. The models also have loss functions as methods but we can also customise the loss by calling
        # a_loss_function(model(data))
        self.model = self.config.method(self.config)
        best_model = copy.deepcopy(self.model.state_dict())
        self.model.double().to(self.device)
        min_val_loss = torch.tensor(np.inf)
        epochs_no_improve = 0
        early_stop = False
        all_train_loss = []
        all_val_loss = []

        for epoch in range(1, self.config.epoch_num + 1):
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
                    if epochs_no_improve == self.config.patience and self.config.patience > 0:
                        print('Early stopping!')
                        early_stop = True
                        self.model.load_state_dict(best_model)

                all_train_loss.append(epoch_train_loss)
                all_val_loss.append(epoch_val_loss)
        cca_zoo.plot_utils.plot_training_loss(all_train_loss, all_val_loss)

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
            self.cca = CCA(n_components=self.config.latent_dims)
            view_1, view_2 = self.cca.fit_transform(z_list[0], z_list[1])
        else:
            view_1, view_2 = self.cca.transform(np.array(z_list[0]), np.array(z_list[1]))
        correlations = np.diag(
            np.corrcoef(view_1, view_2, rowvar=False)[:self.config.latent_dims, self.config.latent_dims:])
        return correlations

    def transform_view(self, *args, train=False):
        dataset_list_test = [arg if train else arg - self.dataset_means[i] for i, arg in enumerate(args)]
        test_dataset = TensorDataset(*[torch.tensor(dataset) for dataset in dataset_list_test])
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader):
                data = [d.to(self.device) for d in list(data)]
                z = self.model(*data)
                if batch_idx == 0:
                    z_list = [np.empty((0, self.config.latent_dims)) for _ in range(len(z))]
                z_list = [np.append(z_list[i], z_i.detach().cpu().numpy(), axis=0) for i, z_i in enumerate(z)]
        return z_list

    def predict_view(self, *args, train=False):
        dataset_list_test = [arg if train else arg - self.dataset_means[i] for i, arg in enumerate(args)]
        test_dataset = TensorDataset(*[torch.tensor(dataset) for dataset in dataset_list_test])
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
        x_list = [np.empty((0, arg.shape[1:])) for i, arg in enumerate(args)]
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader):
                data = [d.to(self.device) for d in list(data)]
                x = self.model.recon(*data)
                x_list = [np.append(x_i, x[i].detach().cpu().numpy(), axis=0) for i, x_i in enumerate(x_list)]
        return x_list

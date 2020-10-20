import copy
import shutil

import torch.nn.functional as F
import torch.utils.data
from sklearn.cross_decomposition import CCA
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from CCA_methods import DGCCAE
from CCA_methods.plot_utils import *


class GraphWrapper:

    def __init__(self, latent_dims=2, learning_rate=1e-3, epoch_num=1, batch_size=100,
                 reg_par=1e-5, use_all_singular_values=True, method='DGCCAE', lam=0, both_encoders=False,
                 print_batch=False, patience=10):
        self.latent_dims = latent_dims
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        # Default - may change during training due to needing batch size greater than 1
        self.batch_size = batch_size
        # the regularization parameter of the network
        # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
        self.reg_par = reg_par
        self.use_all_singular_values = use_all_singular_values
        self.method = method
        self.lam = lam
        self.both_encoders = both_encoders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_interval = 10
        self.print_batch = print_batch
        self.patience = patience

    def fit(self, X_train, Y_train):

        if self.method == 'DGCCAE':
            self.model = DGCCAE(input_size_1=X_train.shape[1], input_size_2=Y_train.shape[1], lam=self.lam,
                                latent_dims=self.latent_dims).double().to(self.device)

        num_subjects = X_train.shape[0]
        all_inds = np.arange(num_subjects)
        np.random.shuffle(all_inds)
        train_inds, val_inds = np.split(all_inds, [int(round(0.8 * num_subjects, 0))])
        X_val = X_train[val_inds]
        Y_val = Y_train[val_inds]
        X_train = X_train[train_inds]
        Y_train = Y_train[train_inds]
        # Demeaning
        self.X_mean = X_train.mean(axis=0)
        self.Y_mean = Y_train.mean(axis=0)
        X_train -= self.X_mean
        Y_train -= self.Y_mean
        X_val -= self.X_mean
        Y_val -= self.Y_mean

        train_dataset = MyOwnDataset(X_train, Y_train, root='train')  # create your datset
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
        val_dataset = MyOwnDataset(X_val, Y_val, root='val')  # create your datset
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))

        while X_train.shape[0] % self.batch_size < 10 or Y_train.shape[0] % self.batch_size < 10:
            self.batch_size += 1

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        min_val_loss = 0
        epochs_no_improve = 0
        early_stop = False
        epoch_train_loss = []
        epoch_val_loss = []

        for epoch in range(self.epoch_num):
            if early_stop == False:
                self.model.train()
                train_loss = 0
                for batch_idx,data in enumerate(train_dataloader):
                    self.optimizer.zero_grad()
                    model_outputs = self.model(data.to(self.device))
                    loss = self.model.loss(data.edge_attr, data.behaviour, *model_outputs)
                    loss.backward()
                    train_loss += loss.item()
                    self.optimizer.step()

                print('====> Epoch: {} Average train loss: {:.4f}'.format(
                    epoch, train_loss / len(train_dataloader)))

                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for batch_idx,data in enumerate(val_dataloader):
                        model_outputs = self.model(data.to(self.device))
                        # a = CCA(n_components=self.latent_dims).fit(model_outputs[0].detach().numpy(), model_outputs[1].detach().numpy())
                        # np.sum(np.diag(np.corrcoef(a.y_scores_.T,a.x_scores_.T)[:self.latent_dims, self.latent_dims:]))
                        loss = self.model.loss(data.edge_attr, data.behaviour, *model_outputs)
                        val_loss += loss.item()

                    print('====> Epoch: {} Average val loss: {:.4f}'.format(
                        epoch, val_loss / len(val_dataloader)))

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
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

                epoch_train_loss.append(train_loss / len(train_dataloader))
                epoch_val_loss.append(val_loss / len(val_dataloader))
        plot_training_loss(epoch_train_loss, epoch_val_loss)

        if self.method == 'DGCCAE':
            self.train_correlations = self.predict_corr(X_train, Y_train, train=True)

        self.train_recon_loss_x, self.train_recon_loss_y = self.predict_recon(X_train, Y_train)

        return self

    def predict_corr(self, X_test, Y_test, train=False):
        X_test -= self.X_mean
        Y_test -= self.Y_mean
        test_dataset = TensorDataset(X_test, Y_test)  # create your datset
        test_dataloader = DataLoader(test_dataset, batch_size=100)
        z_x = np.empty((0, self.latent_dims))
        z_y = np.empty((0, self.latent_dims))
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader):
                z_x_batch, z_y_batch, recon_x_batch, recon_y_batch = self.model(data.to(self.device))
                z_x = np.append(z_x, z_x_batch.detach().cpu().numpy(), axis=0)
                z_y = np.append(z_y, z_y_batch.detach().cpu().numpy(), axis=0)
        if train:
            self.cca = CCA(n_components=self.latent_dims)
            view_1, view_2 = self.cca.fit_transform(z_x, z_y)
        else:
            view_1, view_2 = self.cca.transform(np.array(z_x), np.array(z_y))
        correlations = np.diag(np.corrcoef(view_1, view_2, rowvar=False)[:self.latent_dims, self.latent_dims:])
        return correlations

    def predict_recon(self, X_new, Y_new):
        X_new -= self.X_mean
        Y_new -= self.Y_mean
        test_dataset = MyOwnDataset(X_new, Y_new)  # create your datset
        test_dataloader = DataLoader(test_dataset, batch_size=100)
        with torch.no_grad():
            recon_loss_x = 0
            recon_loss_y = 0
            for batch_idx, data in enumerate(test_dataloader):
                z_x, z_y, recon_x, recon_y = self.model(data.to(self.device))

            recon_loss_x += F.binary_cross_entropy(recon_x, X_new, reduction='sum').detach().cpu().numpy() / \
                            X_new.size()[0]
            recon_loss_y += F.binary_cross_entropy(recon_y, Y_new, reduction='sum').detach().cpu().numpy() / \
                            Y_new.size()[0]
        return recon_loss_x, recon_loss_y

    def transform_view(self, X_new=None, Y_new=None):
        if X_new is not None:
            X_new -= self.X_mean
            tensor_x_new = torch.DoubleTensor(X_new).to(self.device)
        if Y_new is not None:
            Y_new -= self.Y_mean
            tensor_y_new = torch.DoubleTensor(Y_new).to(self.device)
        if X_new is not None:
            U_new = self.model.encode_1(tensor_x_new)
        if Y_new is not None:
            V_new = self.model.encode_2(tensor_y_new)

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


class MyOwnDataset(InMemoryDataset):
    def __init__(self, X, Y, root='', transform=None, pre_transform=None):
        try:
            shutil.rmtree('./' + root)
        except:
            print("doesn't exist")

        self.X = X
        self.Y = Y

        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for idx, connectivity in enumerate(self.X):
            x = torch.ones(connectivity.shape[1])
            edge_index = np.stack(np.where(np.squeeze(connectivity) != 0))
            edge_attr = connectivity[edge_index[0, :], edge_index[1, :], 0]
            edge_attr = torch.from_numpy(edge_attr)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.behaviour = self.Y[idx]
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

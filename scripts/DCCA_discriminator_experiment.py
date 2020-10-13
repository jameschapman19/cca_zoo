import torch
from James.CCA_methods_2 import CCA_GD
from James.CCA_methods_2.DCCA_adversary import DCCA_adversary
from James.CCA_methods_2.hcp_utils import load_hcp_data, reduce_dimensions
from James.CCA_methods_2.plot_utils import *
from sklearn.cross_decomposition import CCA
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    learning_rate = 1e-3
    epoch_num = 100
    outdims = 2
    conf, behaviour, brain = load_hcp_data(subjects='1000', smith=False)
    view_1 = brain.astype(np.float32)
    view_2 = behaviour.astype(np.float32)
    all_inds = np.arange(view_1.shape[0])
    np.random.shuffle(all_inds)
    train_inds, test_inds = np.split(all_inds, [int(round(0.8 * view_1.shape[0], 0))])
    train_set_1 = view_1[train_inds]
    train_set_2 = view_2[train_inds]
    test_set_1 = view_1[test_inds]
    test_set_2 = view_2[test_inds]

    train_set_1, train_set_2, test_set_1, test_set_2 = reduce_dimensions(train_set_1, train_set_2,
                                                                         test_x=test_set_1,
                                                                         test_y=test_set_2)

    model = CCA_GD(input_size_1=train_set_1.shape[1], input_size_2=train_set_2.shape[1], lam=0,
                   outdim_size=outdims).double().to(device)

    adversary = DCCA_adversary(outdim_size=outdims, confounds=conf.shape[1], alpha=1).double().to(device)

    num_subjects = train_set_1.shape[0]
    all_inds = np.arange(num_subjects)
    np.random.shuffle(all_inds)
    train_inds, val_inds = np.split(all_inds, [int(round(0.8 * num_subjects, 0))])
    X_val = train_set_1[val_inds]
    Y_val = train_set_2[val_inds]
    Conf_val = conf[val_inds]
    X_train = train_set_1[train_inds]
    Y_train = train_set_2[train_inds]
    Conf_train = conf[train_inds]
    tensor_x_train = torch.DoubleTensor(X_train)  # transform to torch tensor
    tensor_y_train = torch.DoubleTensor(Y_train)
    tensor_conf_train = torch.DoubleTensor(Conf_train)
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train, tensor_conf_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
    tensor_x_val = torch.DoubleTensor(X_val)  # transform to torch tensor
    tensor_y_val = torch.DoubleTensor(Y_val)
    tensor_conf_val = torch.DoubleTensor(Conf_val)
    val_dataset = TensorDataset(tensor_x_val, tensor_y_val, tensor_conf_val)  # create your datset
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    while X_train.shape[0] % batch_size < 10 or Y_train.shape[0] % batch_size < 10:
        batch_size += 1

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    adv_optimizer = optim.Adam(adversary.parameters(), lr=learning_rate)

    # Pre train DCCA
    for epoch in range(epoch_num):
        model.train()
        train_loss = 0
        for batch_idx, (x, y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            model_outputs = model(x, y)
            loss = model.loss(x, y, *model_outputs)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average train loss: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader)))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (x, y, _) in enumerate(val_dataloader):
                x, y = x.to(device), y.to(device)
                model_outputs = model(x, y)
                loss = model.loss(x, y, *model_outputs)
                val_loss += loss.item()

            print('====> Epoch: {} Average val loss: {:.4f}'.format(
                epoch, val_loss / len(val_dataloader)))

    # Train Adversary
    for epoch in range(epoch_num):
        train_adv_loss = 0
        train_model_loss = 0
        for batch_idx, (x, y, confound) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            z_x, z_y, x_recon, y_recon = model(x, y)
            # Train adversary
            adv_optimizer.zero_grad()
            adversary_outputs = adversary(z_x, z_y)
            adversary_loss = adversary.loss(confound, adversary_outputs)
            adversary_loss.backward()
            adv_optimizer.step()
            # Train DCCA
            optimizer.zero_grad()
            z_x, z_y, x_recon, y_recon = model(x, y)
            model_loss = model.loss(x, y, z_x, z_y, x_recon, y_recon)
            loss = model_loss - 10000 * adversary.loss(confound, adversary(z_x, z_y))
            loss.backward()
            optimizer.step()
            train_adv_loss += adversary_loss.item()
            train_model_loss += model_loss.item()

        print('====> Epoch: {} Average train loss: {:.4f}'.format(
            epoch, train_model_loss / len(train_dataloader)))

        print('====> Epoch: {} Average train adversarial loss: {:.4f}'.format(
            epoch, train_adv_loss / len(train_dataloader)))

        model.eval()
        # TRAIN DATA
        z_x_train = np.empty((0, outdims))
        z_y_train = np.empty((0, outdims))
        confound_errors = np.empty((0, conf.shape[1]))
        with torch.no_grad():
            train_loss = 0
            for batch_idx, (x, y, confound) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                z_x, z_y, x_recon, y_recon = model(x, y)
                loss = model.loss(x, y, z_x, z_y, x_recon, y_recon)
                adversary_outputs = adversary(z_x, z_y)
                breakdown = adversary.breakdown(confound, adversary_outputs)
                train_loss += loss.item()
                z_x_train = np.append(z_x_train, z_x.detach().cpu().numpy(), axis=0)
                z_y_train = np.append(z_y_train, z_y.detach().cpu().numpy(), axis=0)
                confound_errors = np.append(confound_errors, breakdown.detach().cpu().numpy(), axis=0)

        linear_cca = CCA(n_components=outdims)
        view_1_train, view_2_train = linear_cca.fit_transform(z_x_train, z_y_train)

        # VAL DATA
        z_x_val = np.empty((0, outdims))
        z_y_val = np.empty((0, outdims))
        confound_errors = np.empty((0, conf.shape[1]))
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (x, y, confound) in enumerate(val_dataloader):
                x, y = x.to(device), y.to(device)
                z_x, z_y, x_recon, y_recon = model(x, y)
                loss = model.loss(x, y, z_x, z_y, x_recon, y_recon)
                adversary_outputs = adversary(z_x, z_y)
                breakdown = adversary.breakdown(confound, adversary_outputs)
                val_loss += loss.item()
                z_x_val = np.append(z_x_val, z_x.detach().cpu().numpy(), axis=0)
                z_y_val = np.append(z_y_val, z_y.detach().cpu().numpy(), axis=0)
                confound_errors = np.append(confound_errors, breakdown.detach().cpu().numpy(), axis=0)
        print('====> Epoch: {} Average val loss: {:.4f}'.format(
            epoch, val_loss / len(val_dataloader)))

        if epoch % 20 == 0:
            view_1_val, view_2_val = linear_cca.transform(z_x_val, z_y_val)
            plot_latent_space(view_1_train, view_2_train, conf=confound_errors)
            plot_latent_space(view_1_val, view_2_val, conf=confound_errors)

    # Standard
    plt.show()
    return


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt


def plot_reconstruction(model, dataset, idx):
    (x, y), _ = dataset[idx]
    recon_x, recon_y = model.recon(x, y)
    if isinstance(recon_x, list):
        recon_x = recon_x[0]
        recon_y = recon_y[0]
    recon_x = recon_x.detach().numpy()
    recon_y = recon_y.detach().numpy()
    fig, ax = plt.subplots(ncols=4)
    ax[0].set_title('Original View 1')
    ax[1].set_title('Original View 2')
    ax[2].set_title('Reconstruction View 1')
    ax[3].set_title('Reconstruction View 2')
    ax[0].imshow(x.detach().numpy().reshape((28, 28)))
    ax[1].imshow(y.detach().numpy().reshape((28, 28)))
    ax[2].imshow(recon_x.reshape((28, 28)))
    ax[3].imshow(recon_y.reshape((28, 28)))


def plot_latent_label(model, dataloader, num_batches=100):
    fig, ax = plt.subplots(ncols=model.latent_dims)
    for j in range(model.latent_dims):
        ax[j].set_title(f'Dimension {j}')
        ax[j].set_xlabel('View 1')
        ax[j].set_ylabel('View 2')
    for i, (data, label) in enumerate(dataloader):
        z = model(*data)
        zx, zy = z
        zx = zx.to('cpu').detach().numpy()
        zy = zy.to('cpu').detach().numpy()
        for j in range(model.latent_dims):
            ax[j].scatter(zx[:, j], zy[:, j], c=label.numpy(), cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

import matplotlib.pyplot as plt
from CCA_methods.generate_data import generate_witten,generate_mai

plt.style.use('ggplot')
plt.rcParams['lines.markersize'] = 2
plt.rcParams.update({'font.size': 5})
outdim_size = 1

X_witten, Y_witten, true_x, true_y = generate_witten(200, outdim_size, 200, 200, sigma=0.3, tau=0.3,
                                                     sparse_variables_1=10,
                                                     sparse_variables_2=10)

X_identity, Y_identity, true_x, true_y = generate_mai(200, outdim_size, 200, 200, sparse_variables_1=10,
                                                      sparse_variables_2=10, structure='identity', sigma=0.1)

X_correlated, Y_correlated, true_x, true_y = generate_mai(200, outdim_size, 200, 200, sparse_variables_1=10,
                                                          sparse_variables_2=10, structure='correlated', sigma=0.1)

X_toeplitz, Y_toeplitz, true_x, true_y = generate_mai(200, outdim_size, 200, 200, sparse_variables_1=10,
                                                          sparse_variables_2=10, structure='toeplitz', sigma=0.8)

X_toeplitzh, Y_toeplitzh, true_x, true_y = generate_mai(200, outdim_size, 200, 200, sparse_variables_1=10,
                                                          sparse_variables_2=10, structure='toeplitz', sigma=0.9)

from mpl_toolkits.axes_grid1 import ImageGrid

# Set up figure and image grid
fig = plt.figure(figsize=(9.75, 3))

axs = ImageGrid(fig, 111,  # as in plt.subplot(111)
                nrows_ncols=(1, 4),
                axes_pad=0.15,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="7%",
                cbar_pad=0.15,
                )
witten_cov = X_witten.T @ X_witten
identity_cov = X_identity.T @ X_identity
toeplitz_cov = X_toeplitz.T @ X_toeplitz
toeplitzh_cov = X_toeplitzh.T @ X_toeplitzh
witten = axs[0].imshow(witten_cov / witten_cov.max())
axs[0].set_title('Witten')
identity = axs[1].imshow(identity_cov / identity_cov.max())
axs[1].set_title('Identity')
toeplitz = axs[2].imshow(toeplitz_cov / toeplitz_cov.max())
axs[2].set_title('Toeplitz AR(0.8)')
toeplitz_high = axs[3].imshow(toeplitzh_cov / toeplitzh_cov.max())
axs[3].set_title('Toeplitz AR(0.9)')
axs[3].cax.colorbar(toeplitz_high)
axs[3].cax.toggle_label(True)
plt.savefig('covariancestructures')
plt.show()
print('done')

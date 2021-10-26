# %%

# Imports
import sys
sys.executable
#%%

import datasets
from ccagame.pls import Game, SGD, Incremental, Batch, Numpy, MSG
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir('./results/ukbb_results/211026')
print(os.getcwd())
# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 3
epochs = 200
riemannian_projection = False
lr = 1e-3
batch_size = 1
# This implements the version with unbiased updates analagous to Eigengame:Unloaded
PATH = '/mnt/c/Users/anala/Documents/PhD/year_1/project_work/Epilepsy/data/PLS/210326'
train_1, train_2 = datasets.ukbiobank(path=PATH, save=True)
print(train_1.shape)
# Model
numpy = Numpy(scale=False, n_components=latent_dims).fit(train_1, train_2)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(train_1, train_2))
print("\n Time :\n", numpy.fit_time)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=False,
            mu=True).fit(train_1, train_2)
print("\n Eigenvalues calculated using game are :\n", game.score(train_1, train_2))
print("\n Time :\n", game.fit_time)
np.save('game_train', list(zip(*game.obj))[0])
np.save('game_val', list(zip(*game.obj))[1])

sgd = SGD(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=False).fit(train_1,
                                                                                                                train_2)
print("\n Eigenvalues calculated using sgd are :\n", sgd.score(train_1, train_2))
print("\n Time :\n", sgd.fit_time)

np.save('sgd_train', list(zip(*sgd.obj))[0])
np.save('sgd_val',list(zip(*sgd.obj))[1])

incremental = Incremental(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=False).fit(train_1,
                                                                                                         train_2)
print("\n Eigenvalues calculated using incremental are :\n", incremental.score(train_1, train_2))
print("\n Time :\n", incremental.fit_time)
np.save('inc_train', list(zip(*incremental.obj))[0])
np.save('inc_val', list(zip(*incremental.obj))[1])

msg = MSG(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=False).fit(train_1,
                                                                                                                train_2)
print("\n Eigenvalues calculated using msg are :\n", msg.score(train_1, train_2))
print("\n Time :\n", msg.fit_time)
np.save('msg_train', list(zip(*msg.obj))[0])
np.save('msg_val', list(zip(*msg.obj))[1])

batch = Batch(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=False).fit(train_1, train_2)
print("\n Eigenvalues calculated using batch are :\n", batch.score(train_1, train_2))
print("\n Time :\n", batch.fit_time)


# %%
plt.figure()
plt.suptitle('UKBB results for batch size: {0}'.format(batch_size))
plt.subplot(1,2,1)
plt.title('Training covariance')
plt.plot(np.load('sgd_train.npy'), label='sgd')
plt.plot(np.load('game_train.npy'), label='game')
plt.plot(np.load('msg_train.npy'), label='msg')
plt.plot(np.load('inc_train.npy'), label='inc')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Validation covariance')
plt.plot(np.load('sgd_val.npy'), label='sgd')
plt.plot(np.load('game_val.npy'), label='game')
plt.plot(np.load('msg_val.npy'), label='msg')
plt.plot(np.load('inc_val.npy'), label='inc')
plt.legend()
plt.savefig("testing.png")








# %%

# Imports
import sys
sys.executable
#%%

from ccagame import datasets
from ccagame.pls import Game, SGD, Incremental, Batch, Numpy, MSG
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('ukbb_results')
# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 3
epochs = 1
riemannian_projection = False
lr = 1e-3
batch_size = 1
path = 'C:/Users/anala/Documents/PhD/year_1/project_work/Epilepsy/data/PLS/210326'
# This implements the version with unbiased updates analagous to Eigengame:Unloaded
train_1, train_2, test_1, test_2 = datasets.ukbiobank(path)

# Model
numpy = Numpy(scale=False, n_components=latent_dims).fit(train_1, train_2)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(train_1, train_2))
print("\n Time :\n", numpy.fit_time)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(train_1, train_2)
print("\n Eigenvalues calculated using game are :\n", game.score(train_1, train_2))
print("\n Time :\n", game.fit_time)
np.save('game', game.obj)

sgd = SGD(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1,
                                                                                                                train_2)
print("\n Eigenvalues calculated using sgd are :\n", sgd.score(train_1, train_2))
print("\n Time :\n", sgd.fit_time)
np.save('sgd', sgd.obj)

incremental = Incremental(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1,
                                                                                                         train_2)
print("\n Eigenvalues calculated using incremental are :\n", incremental.score(train_1, train_2))
print("\n Time :\n", incremental.fit_time)
np.save('inc', incremental.obj)

msg = MSG(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1,
                                                                                                                train_2)
print("\n Eigenvalues calculated using msg are :\n", msg.score(train_1, train_2))
print("\n Time :\n", msg.fit_time)
np.save('msg', msg.obj)

batch = Batch(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1, train_2)
print("\n Eigenvalues calculated using batch are :\n", batch.score(train_1, train_2))
print("\n Time :\n", batch.fit_time)


# %%
fig=plt.figure()
plt.plot(np.load('ukbb_results/sgd.npy'), label='sgd')
plt.plot(np.load('ukbb_results/game.npy'), label='game')
plt.plot(np.load('ukbb_results/msg.npy'), label='msg')
plt.plot(np.load('ukbb_results/inc.npy'), label='inc')
plt.legend()
plt.show()









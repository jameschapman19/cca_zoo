# %%

# Imports

from ccagame import datasets
from ccagame.pls import Game, SGD, Incremental, Batch, Numpy, MSG

# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 3
epochs = 50
riemannian_projection = False
lr = 1e-2
batch_size = 100
# This implements the version with unbiased updates analagous to Eigengame:Unloaded
mu = True
train, train_labels, test_1, test_labels = datasets.mnist()
train_1 = train[:, :392]
train_2 = train[:, 392:]

# Model
numpy = Numpy(scale=False, n_components=latent_dims).fit(train_1, Y)
print("\n Eigenvalues calculated using numpy are :\n", numpy.score(train_1, train_2))
print("\n Time :\n", numpy.fit_time)

msg = MSG(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1,
                                                                                                                train_2)
print("\n Eigenvalues calculated using msg are :\n", msg.score(train_1, train_2))
print("\n Time :\n", msg.fit_time)

game = Game(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True,
            mu=True).fit(train_1, train_2)
print("\n Eigenvalues calculated using game are :\n", game.score(train_1, train_2))
print("\n Time :\n", game.fit_time)

sgd = SGD(scale=False, lr=lr, batch_size=batch_size, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1,
                                                                                                                train_2)
print("\n Eigenvalues calculated using sgd are :\n", sgd.score(train_1, train_2))
print("\n Time :\n", sgd.fit_time)

batch = Batch(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1, train_2)
print("\n Eigenvalues calculated using batch are :\n", batch.score(train_1, train_2))
print("\n Time :\n", batch.fit_time)

incremental = Incremental(scale=False, lr=lr, epochs=epochs, n_components=latent_dims, verbose=True).fit(train_1,
                                                                                                         train_2)
print("\n Eigenvalues calculated using incremental are :\n", incremental.score(train_1, train_2))
print("\n Time :\n", incremental.fit_time)

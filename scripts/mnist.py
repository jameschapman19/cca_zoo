import itertools
import time

import numpy.random as npr
from jax import jit, grad, random
from jax.experimental import optimizers

from ccagame import datasets

if __name__ == "__main__":
    rng = random.PRNGKey(0)

    step_size = 0.001
    num_epochs = 10
    batch_size = 128

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)


    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]


    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)


    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)


    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))

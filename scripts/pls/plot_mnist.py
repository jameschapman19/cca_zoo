import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.plot(np.load('mnist_results/sgd_1.npy'), label='sgd-1')
plt.plot(np.load('mnist_results/sgd_128.npy'), label='sgd-128')
plt.plot(np.load('mnist_results/game.npy'), label='game-1')
plt.plot(np.load('mnist_results/game_128.npy'), label='game-128')
# plt.plot(np.load('mnist_results/msg.npy'), label='msg')
plt.plot(np.load('mnist_results/inc.npy'), label='inc-1')
plt.legend()
plt.title('Objective (total variance in 3 latent dimensions)\n in holdout data against number of iterations')
plt.xscale('log')
plt.xlabel('log iterations')
plt.ylabel('objective in holdout')
plt.show()

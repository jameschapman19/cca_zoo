import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure()
plt.plot(np.load('mnist_results/sgd.npy'), label='sgd')
plt.plot(np.load('mnist_results/game.npy'), label='game')
plt.plot(np.load('mnist_results/msg.npy'), label='msg')
plt.plot(np.load('mnist_results/inc.npy'), label='inc')
plt.legend()
plt.show()

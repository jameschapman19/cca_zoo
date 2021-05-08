from .simulated import generate_covariance_data, generate_simple_data

try:
    from .mnist import CCA_Dataset, Noisy_MNIST_Dataset, Tangled_MNIST_Dataset
except ModuleNotFoundError:
    pass

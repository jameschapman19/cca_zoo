from .simulated import generate_covariance_data, generate_simple_data

try:
    from .toy import Noisy_MNIST_Dataset, Tangled_MNIST_Dataset, Split_MNIST_Dataset
    from .utils import CCA_Dataset
except ModuleNotFoundError:
    pass

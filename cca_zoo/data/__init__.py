from .simulated import generate_covariance_data, generate_simple_data

try:
    from .toy import NoisyMNISTDataset, TangledMNISTDataset, SplitMNISTDataset
    from .utils import CCA_Dataset
except ModuleNotFoundError:
    pass

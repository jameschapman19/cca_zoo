import cca_zoo.deepmodels.architectures
import cca_zoo.deepmodels.objectives
from ._dcca_base import _DCCA_base
from .dcca import DCCA
from .dcca_noi import DCCA_NOI
from .dccae import DCCAE
from .dtcca import DTCCA
from .dvcca import DVCCA
from .splitae import SplitAE
from .trainers import CCALightning
from .utils import get_dataloaders, process_data

import cca_zoo.deepmodels.architectures
import cca_zoo.deepmodels.objectives
from ._dcca_base import _DCCA_base
from .dcca import DCCA
from .dcca_barlow_twins import BarlowTwins
from .dcca_noi import DCCA_NOI
from .dcca_sdl import DCCA_SDL
from .dccae import DCCAE
from .dtcca import DTCCA
from .dvcca import DVCCA
from .splitae import SplitAE
from .trainers import CCALightning
from .utils import get_dataloaders, process_data
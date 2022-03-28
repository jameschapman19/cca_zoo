import cca_zoo.data
import cca_zoo.model_selection
import cca_zoo.models

try:
    import cca_zoo.deepmodels
except ModuleNotFoundError:
    pass
try:
    import cca_zoo.probabilisticmodels
except ModuleNotFoundError:
    pass
from re import I
from sys import argv

from absl import flags
from absl.testing import parameterized
from absl.testing import absltest
from ccagame import cca, pca, pls
from jaxline_fork import platform
from ml_collections import config_flags
from ccagame.test.config import get_config

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    help_string="Training configuration file.",
    default="",
)
MODEL_DICT = {
            "game": cca.Game,
            "vicreggame": cca.VicRegGame,
            "genoja": cca.GenOja,
            "sgha": cca.SGHA,
            "appgrad": cca.AppGrad,
        }

class CCATest(parameterized.TestCase):
    @parameterized.parameters(
       {'model': 'game',
       'model': 'appgrad',
       'model': 'genoja',
       'model': 'sgha',
       'model': 'vicreggame',},
    )
    def test_cca(self,model):
        FLAGS.config = get_config()#FLAGS.config.is_type_safe
        FLAGS.config.experiment_kwargs = {
            "n_components": FLAGS.config.n_components,
            "num_devices": FLAGS.config.num_devices,
            "data": FLAGS.config.data,
            "batch_size": FLAGS.config.batch_size,
            "learning_rate": FLAGS.config.learning_rate,
            "validate": FLAGS.config.validate,
        }
        platform.main(MODEL_DICT[model], argv)


if __name__ == "__main__":
    absltest.main()
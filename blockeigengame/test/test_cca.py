from re import I
from sys import argv

from absl import flags
from absl.testing import parameterized
from absl.testing import absltest
from ccagame import cca
from jaxline_fork import platform
from ml_collections import config_flags
import os

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    help_string="Training configuration file.",
    default="config.py",
)
flags.DEFINE_string(name="model", default="game", help="model name")


class TestCCA(parameterized.TestCase):
    """
    @parameterized.parameters(
       {'model': 'game',
       'model': 'appgrad',
       'model': 'genoja',
       'model': 'sgha',
       'model': 'vicreggame',},
    )
    """

    def test_cca(self, model="game"):
        MODEL_DICT = {
            "game": cca.Game,
            "vicreggame": cca.VicRegGame,
            "genoja": cca.GenOja,
            "sgha": cca.SGHA,
            "appgrad": cca.AppGrad,
        }
        FLAGS.model = model
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

from re import I
from sys import argv

from absl import flags
from absl.testing import absltest
from ccagame import cca, pca, pls
from jaxline_fork import platform
from ml_collections import config_flags

from config import get_config

FLAGS = flags.FLAGS

flags.DEFINE_string(name="model", default="game", help="model name")
config_flags.DEFINE_config_file(
    "config",
    help_string="Training configuration file.",
    default="",
)


class PLSTest(absltest.TestCase):

    def test_pls(self):
        FLAGS.config = get_config()
        FLAGS.config.experiment_kwargs = {
            "n_components": FLAGS.config.n_components,
            "num_devices": FLAGS.config.num_devices,
            "data": FLAGS.config.data,
            "batch_size": FLAGS.config.batch_size,
            "learning_rate": FLAGS.config.learning_rate,
            "validate": FLAGS.config.validate,
        }
        platform.main(pls.Game, argv)


if __name__ == "__main__":
    absltest.main()

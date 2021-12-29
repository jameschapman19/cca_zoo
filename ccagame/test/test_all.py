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
    default="/home/chapmajw/ccagame/experiments/pls/mnist/config.py",
)
flags.DEFINE_string(name="model", default="game", help="model name")

class TestPCA(parameterized.TestCase):
    @parameterized.parameters(
            {'model': 'game',
            'model': 'oja',
            'model': 'gha'},
            )
    def test_pca(self,model):
        MODEL_DICT = {
            "game": pca.Game,
            "oja": pca.Oja,
            "gha": pca.GHA,
        }
        FLAGS.model=model
        FLAGS.config.experiment_kwargs = {
            "n_components": FLAGS.config.n_components,
            "num_devices": FLAGS.config.num_devices,
            "data": FLAGS.config.data,
            "batch_size": FLAGS.config.batch_size,
            "learning_rate": FLAGS.config.learning_rate,
            "validate": FLAGS.config.validate,
        }
        platform.main(MODEL_DICT[model], argv)

class TestCCA(parameterized.TestCase):
    @parameterized.parameters(
       {'model': 'game',
       'model': 'appgrad',
       'model': 'genoja',
       'model': 'sgha',
       'model': 'vicreggame',},
    )
    def test_cca(self,model):
        MODEL_DICT = {
            "game": cca.Game,
            "vicreggame": cca.VicRegGame,
            "genoja": cca.GenOja,
            "sgha": cca.SGHA,
            "appgrad": cca.AppGrad,
        }
        FLAGS.model=model
        FLAGS.config.experiment_kwargs = {
            "n_components": FLAGS.config.n_components,
            "num_devices": FLAGS.config.num_devices,
            "data": FLAGS.config.data,
            "batch_size": FLAGS.config.batch_size,
            "learning_rate": FLAGS.config.learning_rate,
            "validate": FLAGS.config.validate,
        }
        platform.main(MODEL_DICT[model], argv)
    
class TestPLS(parameterized.TestCase):
    @parameterized.parameters(
            {'model': 'game',
            'model': 'msg',
            'model': 'oja',
            'model': 'power',
            'model': 'incremental'},
            )
    def test_pls(self,model):
        MODEL_DICT = {
            "game": pls.Game,
            "msg": pls.MSG,
            "oja": pls.Oja,
            "power": pls.StochasticPower,
            "incremental": pls.Incremental,
        }
        FLAGS.model=model
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

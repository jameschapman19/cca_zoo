from re import I
from sys import argv

from absl import flags
from absl.testing import absltest, parameterized
from ccagame import cca, pca, pls
from jaxline_fork import platform
from ccagame.test.config import get_config

FLAGS = flags.FLAGS
MODEL_DICT = {
    "game": pls.Game,
    "msg": pls.MSG,
    "oja": pls.Oja,
    "power": pls.StochasticPower,
    "incremental": pls.Incremental,
}

class PLSTest(parameterized.TestCase):
    @parameterized.parameters(
            {'model': 'game',
            'model': 'msg',
            'model': 'oja',
            'model': 'power',
            'model': 'incremental'},
            )
    def test_pls(self,model):
        flags.DEFINE_string(name="model", default="game", help="model name")
        FLAGS.model=model
        FLAGS.config = get_config()
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

import functools
from os import environ

import jax.numpy as jnp
from absl import app, flags
from ccagame.baseexperiment import get_config
from ccagame.utils import data_stream
from datasets.mnist import mnist
from jaxline import platform
import jax
from ccagame.pca.experiments import PCAExperiment
import optax

CORES = 1
environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
FLAGS = flags.FLAGS


class GHA(PCAExperiment):
    def __init__(
        self,
        mode,
        init_rng=None,
        num_devices=1,
        k_per_device=1,
        dims=1,
        data_stream=None,
        whole_batch=False
    ):
        super(GHA, self).__init__(
            mode,
            init_rng=init_rng,
            num_devices=num_devices,
            k_per_device=k_per_device,
            dims=dims,
            data_stream=data_stream,
            whole_batch=whole_batch
        )
        """
        Constructs the experiment.
        Args:
          mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
          init_rng: A `PRNGKey` to use for experiment initialization.
        """
        """
        Initialization function for a Jaxline experiment.
        """
        self._V = jax.random.normal(self.local_rng, (k_per_device, dims))
        self._optimizer = optax.sgd(learning_rate=1e-5, momentum=0.9, nesterov=True)
        self._opt_state = self._optimizer.init(self._V)

    def _update(self, inputs, global_step):
        dv = (
            self._V @ inputs.T @ inputs
            - jnp.triu(self._V @ X.T @ X @ self._V.T) @ self._V
        )
        self._V, self._opt_state = self._update_with_grads(
            self._V, dv, self._optimizer, self._opt_state
        )
        return self._V

    def _update_with_grads(self, vi, grads, opt, opt_state):
        """Compute and apply updates with optax optimizer.
        Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = opt.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new = jnp.diag(1 / jnp.linalg.norm(vi_new, axis=1)) @ vi_new
        return vi_new, opt_state


# TO RUN AN EXPERIMENT YOU HAVE TO TINKER HERE A BIT.
if __name__ == "__main__":
    X, _, X_te, _ = mnist()
    input_data_iterator = data_stream(X, Y=None, batch_size=None)
    k_per_device = 5
    FLAGS.config = get_config(input_data_iterator, CORES, X.shape[1],k_per_device=k_per_device)
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, GHA))

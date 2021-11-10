# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training script to run Jaxline experiments.

Your experiment file must implement the API in experiment.py to be
compatible with this pipeline.
"""

import functools
import inspect
import time

from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
from jaxline import utils

FLAGS = flags.FLAGS


def _log_outputs(step, scalar_values):
  # f_list for less verbosity; e.g., "4." instead of "array(4., dtype=float32)".
  f_list = lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x
  logging.info("global_step: %d, %s", step, jax.tree_map(f_list, scalar_values))


def _initialize_experiment(experiment_class, mode, rng, experiment_kwargs):
  """Initializes experiment catching old style init methods."""
  init_args = inspect.getfullargspec(experiment_class).args
  if "init_rng" in init_args:
    experiment = experiment_class(
        mode, init_rng=rng, **experiment_kwargs)
  else:
    logging.warning(
        "As of cl/339673620 you should add init_rng to your Experiment"
        " constructor, which we will use to pass you"
        " jaxline.base_config.random_seed. Please deprecate any use of"
        " experiment_kwargs.config.random_seed: model initialization should"
        " be performed with init_rng and any sweeps directly with"
        " jaxline.base_config.random_seed. The use of"
        " experiment_kwargs.config.random_seed was flawed design introduced"
        " by some of our JAXline examples and meant that the rng used for"
        " initialization (and sweeps) was decoupled from that used by the"
        " step function. We will enforce the change, breaking any remaining"
        " users who haven't transitioned, on 01/01/21")
    experiment = experiment_class(mode, **experiment_kwargs)
  return experiment


@utils.disable_pmap_jit
def train(experiment_class,
          config,
          checkpointer: utils.Checkpointer,
          writer: utils.Writer,
          periodic_actions=()):
  """Main training loop."""
  logging.info("Training with config:\n%s", config)
  is_chief = jax.host_id() == 0

  rng = jax.random.PRNGKey(config.random_seed)
  with utils.log_activity("experiment init"):
    experiment = _initialize_experiment(
        experiment_class, "train", rng, config.experiment_kwargs)

  state = checkpointer.get_experiment_state("latest")
  state.global_step = 0
  state.experiment_module = experiment
  state.train_step_rng = utils.bcast_local_devices(rng)

  if checkpointer.can_be_restored("latest"):
    with utils.log_activity("checkpoint restore"):
      checkpointer.restore("latest")

  periodic_actions += (
      utils.PeriodicAction(
          _log_outputs,
          interval_type=config.logging_interval_type or config.interval_type,
          interval=config.log_tensors_interval),
      )

  if config.train_checkpoint_all_hosts or is_chief:
    if config.save_checkpoint_interval > 0:
      periodic_actions += (
          utils.PeriodicAction(
              lambda *_: checkpointer.save("latest"),
              interval_type=(config.checkpoint_interval_type
                             or config.interval_type),
              interval=config.save_checkpoint_interval,
              run_async=False),)  # run_async True would not be thread-safe.

  if is_chief:
    if writer is not None:
      def write_scalars(global_step: int, scalar_values):
        writer.write_scalars(global_step, scalar_values)

      periodic_actions += (
          utils.PeriodicAction(
              write_scalars,
              interval_type=(config.logging_interval_type
                             or config.interval_type),
              interval=config.log_train_data_interval,
              log_all_data=config.log_all_train_data),)

  for pa in periodic_actions:
    pa.update_time(time.time(), state.global_step)

  experiment.train_loop(config, state, periodic_actions, writer)

  if is_chief:
    with utils.log_activity("final checkpoint"):
      checkpointer.save("latest")

  # Join all async periodic actions that are unfinished.
  for pa in periodic_actions:
    pa.wait_to_finish()

  # We occasionally see errors when the final checkpoint is being written if
  # the other hosts exit. Here we force all hosts to participate in one final
  # collective so the non-master hosts cannot exit before the master writes out
  # the final checkpoint.
  utils.rendezvous()


@utils.disable_pmap_jit
def evaluate(experiment_class,
             config,
             checkpointer: utils.Checkpointer,
             writer: utils.Writer,
             jaxline_mode=None):
  """Main evaluation loop."""
  if jaxline_mode is None:
    jaxline_mode = FLAGS.jaxline_mode

  logging.info("Evaluating with config:\n%s", config)

  global_step = 0
  eval_rng = jax.random.PRNGKey(config.random_seed)
  experiment = _initialize_experiment(
      experiment_class, jaxline_mode, eval_rng, config.experiment_kwargs)

  if config.best_model_eval_metric and jax.host_id() == 0:
    # Initialize best state.
    best_state = checkpointer.get_experiment_state("best")
    if config.best_model_eval_metric_higher_is_better:
      best_state.best_eval_metric_value = float("-inf")
      eval_metric_is_better_op = jnp.greater
      eval_metric_comparison_str = ">"
    else:
      best_state.best_eval_metric_value = float("inf")
      eval_metric_is_better_op = jnp.less
      eval_metric_comparison_str = "<"
    best_state.best_model_eval_metric = config.best_model_eval_metric

    best_state.experiment_module = experiment

    # Restore to preserve 'best_eval_metric_value' if evaluator was preempted.
    if checkpointer.can_be_restored("best"):
      with utils.log_activity("best checkpoint restore"):
        checkpointer.restore("best")

  # Will evaluate the latest checkpoint in the directory.
  state = checkpointer.get_experiment_state("latest")
  state.global_step = global_step
  state.experiment_module = experiment
  state.train_step_rng = None

  eval_rng = jnp.broadcast_to(
      eval_rng, (jax.local_device_count(),) + eval_rng.shape)
  host_id_devices = utils.host_id_devices_for_rng(config.random_mode_eval)
  eval_rng = jax.pmap(functools.partial(
      utils.specialize_rng_host_device, axis_name="i",
      mode=config.random_mode_eval), axis_name="i")(eval_rng, host_id_devices)

  if config.one_off_evaluate:
    checkpointer.restore("latest")
    global_step_devices = utils.bcast_local_devices(
        jnp.asarray(state.global_step))
    scalar_values = utils.evaluate_should_return_dict(experiment.evaluate)(
        global_step=global_step_devices, rng=eval_rng, writer=writer)
    if writer is not None:
      writer.write_scalars(state.global_step, scalar_values)
    logging.info("Evaluated specific checkpoint, exiting.")
    return

  old_checkpoint_path = None
  initial_weights_are_evaluated = False

  while True:
    checkpoint_path = checkpointer.restore_path("latest")

    if (checkpoint_path is None and config.eval_initial_weights
        and not initial_weights_are_evaluated):
      # Skip restoring a checkpoint and directly call evaluate if
      # `config.eval_initial_weights` but don"t do it more than once.
      initial_weights_are_evaluated = True
    else:
      if checkpoint_path in (None, old_checkpoint_path):
        logging.info("Checkpoint %s invalid or already evaluated, waiting.",
                     checkpoint_path)
        time.sleep(10)
        continue

      checkpointer.restore("latest")

    global_step_devices = utils.bcast_local_devices(
        jnp.asarray(state.global_step))
    scalar_values = utils.evaluate_should_return_dict(experiment.evaluate)(
        global_step=global_step_devices, rng=eval_rng, writer=writer)
    if writer is not None:
      writer.write_scalars(state.global_step, scalar_values)
    old_checkpoint_path = checkpoint_path
    # Decide whether to save a "best checkpoint".
    if config.best_model_eval_metric and jax.host_id() == 0:
      if config.best_model_eval_metric not in scalar_values:
        raise ValueError(f"config.best_model_eval_metric has been specified "
                         f"as {config.best_model_eval_metric}, but this key "
                         f"was not returned by the evaluate method. Got: "
                         f"{scalar_values.keys()}")
      current_eval_metric_value = scalar_values[config.best_model_eval_metric]
      old_eval_metric_value = best_state.best_eval_metric_value
      if eval_metric_is_better_op(current_eval_metric_value,
                                  old_eval_metric_value):
        logging.info("%s: %s %s %s, saving new best checkpoint.",
                     config.best_model_eval_metric, current_eval_metric_value,
                     eval_metric_comparison_str, old_eval_metric_value)
        best_state.global_step = state.global_step
        best_state.experiment_module = experiment
        best_state.best_eval_metric_value = current_eval_metric_value
        best_state.train_step_rng = state.train_step_rng
        checkpointer.save("best")

    if not experiment.should_run_step(state.global_step, config):
      logging.info("Last checkpoint (iteration %d) evaluated, exiting.",
                   state.global_step)
      break

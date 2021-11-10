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
"""Tests for Jaxline's train."""

import copy
from unittest import mock

from absl.testing import absltest
from jaxline import base_config
from jaxline import experiment
from jaxline import train
from ml_collections import config_dict


_IMPROVEMENT_STEPS = [2, 5, 15, 27, 99]  # Arbitrary.
_FITNESS_METRIC_KEY = "A_GOOD_METRIC"


class DummyExperiment(experiment.AbstractExperiment):
  """An experiment whose evaluate improves at set intervals."""

  def __init__(self, mode):
    super().__init__(mode=mode)
    self.evaluate_counter = 0
    self.fitness_metric = 0

  def step(self, **kwargs):
    """Only needed for API matching."""
    pass

  def evaluate(self, *args, **kwargs):
    if self.evaluate_counter in _IMPROVEMENT_STEPS:
      self.fitness_metric += 1
    self.evaluate_counter += 1
    return {_FITNESS_METRIC_KEY: self.fitness_metric}


class DummyCheckpoint:
  """Do nothing but record when save is called."""

  def __init__(self, **kwargs):
    del kwargs  # Unused for this class.
    self._state = config_dict.ConfigDict()
    self._state_list = []
    self._checkpoint_path_int = 0
    self._global_step_int = -1

  def get_experiment_state(self, unused_sequence_key):
    return self._state

  def save(self, unused_sequence_key):
    self._state_list.append(copy.copy(self._state))

  def can_be_restored(self, unused_sequence_key):
    return False

  def restore(self, unused_sequence_key):
    self._global_step_int += 1
    self._state.global_step = self._global_step_int

  def restore_path(self, unused_sequence_key):
    """Always return something new so there"s no waiting."""
    self._checkpoint_path_int += 1
    return str(self._checkpoint_path_int)


class TrainTest(absltest.TestCase):

  def test_best_checkpoint_saves_only_at_improved_best_metrics(self):
    config = base_config.get_base_config()
    config.best_model_eval_metric = _FITNESS_METRIC_KEY
    config.training_steps = 100
    ckpt = DummyCheckpoint()
    writer = mock.Mock()
    train.evaluate(DummyExperiment, config, ckpt, writer, jaxline_mode="eval")

    # The first step will always checkpoint.
    self.assertLen(
        ckpt._state_list, len(_IMPROVEMENT_STEPS) + 1)
    checkpointed_states = [
        s.global_step for s in ckpt._state_list]
    self.assertEqual(checkpointed_states, [0] + _IMPROVEMENT_STEPS)


if __name__ == "__main__":
  absltest.main()

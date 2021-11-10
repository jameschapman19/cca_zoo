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
"""Utility functions for Jaxline experiments."""

import collections
from concurrent import futures
import contextlib
import copy
import enum
import functools
import pdb
import queue
import sys
import threading

from typing import Any, Callable, Dict, Generator, Iterable, Mapping, Optional, TypeVar

from absl import flags
from absl import logging

import chex
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from typing_extensions import Protocol
import wrapt

# TODO(mjoneill): Make flag more informative after copybara is set up
_JAXLINE_POST_MORTEM = flags.DEFINE_bool(
    "jaxline_post_mortem", False,
    "Whether to enter into post-mortem after an exception. ")

_JAXLINE_DISABLE_PMAP_JIT = flags.DEFINE_bool(
    "jaxline_disable_pmap_jit", False,
    "Whether to disable all pmaps and jits, making it easier to inspect and "
    "trace code in a debugger.")


def _get_function_name(function) -> str:
  if isinstance(function, functools.partial):
    return f"partial({function.func.__name__})"
  return function.__name__


SnapshotNT = collections.namedtuple("SnapshotNT", ["id", "pickle_nest"])
CheckpointNT = collections.namedtuple("CheckpointNT", ["active", "history"])

T = TypeVar("T")


class Writer(Protocol):
  """Interface for writers/loggers."""

  def write_scalars(self, global_step: int, scalars: Mapping[str, Any]):
    """Writes a dictionary of scalars."""


class Checkpointer(Protocol):
  """An interface for checkpointer objects."""

  def save(self, ckpt_series: str) -> None:
    """Saves the checkpoint."""

  def restore(self, ckpt_series: str) -> None:
    """Restores the checkpoint."""

  def get_experiment_state(self, ckpt_series: str):
    """Returns the experiment state for a given checkpoint series."""

  def restore_path(self, ckpt_series: str) -> Optional[str]:
    """Returns the restore path for the checkpoint, or None."""

  def can_be_restored(self, ckpt_series: str) -> bool:
    """Returns whether or not a given checkpoint series can be restored."""

  def wait_for_checkpointing_to_finish(self) -> None:
    """Waits for any async checkpointing to complete."""


def py_prefetch(
    iterable_function: Callable[[], Iterable[T]],
    buffer_size: int = 5) -> Generator[T, None, None]:
  """Performs prefetching of elements from an iterable in a separate thread.

  Args:
    iterable_function: A python function that when called with no arguments
      returns an iterable. This is used to build a fresh iterable for each
      thread (crucial if working with tensorflow datasets because tf.graph
      objects are thread local).
    buffer_size (int): Number of elements to keep in the prefetch buffer.

  Yields:
    Prefetched elements from the original iterable.
  Raises:
    ValueError if the buffer_size <= 1.
    Any error thrown by the iterable_function. Note this is not raised inside
      the producer, but after it finishes executing.
  """

  if buffer_size <= 1:
    raise ValueError("the buffer_size should be > 1")

  buffer = queue.Queue(maxsize=(buffer_size - 1))
  producer_error = []
  end = object()

  def producer():
    """Enques items from iterable on a given thread."""
    try:
      # Build a new iterable for each thread. This is crucial if working with
      # tensorflow datasets because tf.graph objects are thread local.
      iterable = iterable_function()
      for item in iterable:
        buffer.put(item)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Error in producer thread for %s",
                        _get_function_name(iterable_function))
      producer_error.append(e)
    finally:
      buffer.put(end)

  threading.Thread(target=producer, daemon=True).start()

  # Consumer.
  while True:
    value = buffer.get()
    if value is end:
      break
    yield value

  if producer_error:
    raise producer_error[0]

# TODO(tomhennigan) Remove this alias.
tree_psum = jax.lax.psum


def double_buffer_on_gpu(ds):
  if jax.default_backend() == "gpu":
    # This keeps two batches per-device in memory at all times, allowing
    # h2d transfers to overlap with execution (see b/173483287 for details).
    return double_buffer(ds)
  else:
    return ds


def _device_put_sharded(sharded_tree, devices):
  leaves, treedef = jax.tree_flatten(sharded_tree)
  n = leaves[0].shape[0]
  return jax.device_put_sharded(
      [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)],
      devices)


def double_buffer(ds: Iterable[T]) -> Generator[T, None, None]:
  """Keeps at least two batches on the accelerator.

  The current GPU allocator design reuses previous allocations. For a training
  loop this means batches will (typically) occupy the same region of memory as
  the previous batch. An issue with this is that it means we cannot overlap a
  host->device copy for the next batch until the previous step has finished and
  the previous batch has been freed.

  By double buffering we ensure that there are always two batches on the device.
  This means that a given batch waits on the N-2'th step to finish and free,
  meaning that it can allocate and copy the next batch to the accelerator in
  parallel with the N-1'th step being executed.

  Args:
    ds: Iterable of batches of numpy arrays.

  Yields:
    Batches of sharded device arrays.
  """
  batch = None
  devices = jax.local_devices()
  for next_batch in ds:
    assert next_batch is not None
    next_batch = _device_put_sharded(next_batch, devices)
    if batch is not None:
      yield batch
    batch = next_batch
  if batch is not None:
    yield batch


def get_first(xs):
  """Gets values from the first device."""
  return jax.tree_map(lambda x: x[0], xs)


def bcast_local_devices(value):
  """Broadcasts an object to all local devices."""
  devices = jax.local_devices()
  return jax.tree_map(
      lambda v: jax.device_put_sharded(len(devices) * [v], devices), value)


def make_async(thread_name_prefix=""):
  """Returns a decorator that runs any function it wraps in a background thread.

   When called, the decorated function will immediately return a future
   representing its result.
   The function being decorated can be an instance method or normal function.
   Consecutive calls to the decorated function are guaranteed to be in order
   and non overlapping.
   An error raised by the decorated function will be raised in the background
   thread at call-time. Raising the error in the main thread is deferred until
   the next call, so as to be non-blocking.
   All subsequent calls to the decorated function after an error has been raised
   will not run (regardless of whether the arguments have changed); instead
   they will re-raise the original error in the main thread.

  Args:
    thread_name_prefix: Str prefix for the background thread, for easier
    debugging.

  Returns:
    decorator that runs any function it wraps in a background thread, and
    handles any errors raised.
  """
  # We have a single thread pool per wrapped function to ensure that calls to
  # the function are run in order (but in a background thread).
  pool = futures.ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix=thread_name_prefix)
  errors = []
  @wrapt.decorator
  def decorator(wrapped, instance, args, kwargs):
    """Runs wrapped in a background thread so result is non-blocking.

    Args:
      wrapped: A function to wrap and execute in background thread.
        Can be instance method or normal function.
      instance: The object to which the wrapped function was bound when it was
        called (None if wrapped is a normal function).
      args: List of position arguments supplied when wrapped function
        was called.
      kwargs: Dict of keyword arguments supplied when the wrapped function was
        called.

    Returns:
      A future representing the result of calling wrapped.
    Raises:
      Exception object caught in background thread, if call to wrapped fails.
      Exception object with stacktrace in main thread, if the previous call to
        wrapped failed.
    """

    def trap_errors(*args, **kwargs):
      """Wraps wrapped to trap any errors thrown."""

      if errors:
        # Do not execute wrapped if previous call errored.
        return
      try:
        return wrapped(*args, **kwargs)
      except Exception as e:
        errors.append(sys.exc_info())
        logging.exception("Error in producer thread for %s",
                          thread_name_prefix)
        raise e

    if errors:
      # Previous call had an error, re-raise in main thread.
      exc_info = errors[-1]
      raise exc_info[1].with_traceback(exc_info[2])
    del instance
    return pool.submit(trap_errors, *args, **kwargs)
  return decorator


def kwargs_only(f):
  @functools.wraps(f)
  def wrapped(**kwargs):
    return f(**kwargs)
  return wrapped


@contextlib.contextmanager
def log_activity(activity_name):
  logging.info("[jaxline] %s starting...", activity_name)
  try:
    yield
  finally:
    if sys.exc_info()[0] is not None:
      logging.exception("[jaxline] %s failed with error.", activity_name)
    else:
      logging.info("[jaxline] %s finished.", activity_name)


class DistributedRNGMode(enum.Enum):
  """Enumeration of the allowed modes for distributed rng handling."""

  UNIQUE_HOST_UNIQUE_DEVICE = "unique_host_unique_device"
  UNIQUE_HOST_SAME_DEVICE = "unique_host_same_device"
  SAME_HOST_UNIQUE_DEVICE = "same_host_unique_device"
  SAME_HOST_SAME_DEVICE = "same_host_same_device"

  @property
  def unique_host(self):
    return self in {DistributedRNGMode.UNIQUE_HOST_UNIQUE_DEVICE,
                    DistributedRNGMode.UNIQUE_HOST_SAME_DEVICE}

  @property
  def unique_device(self):
    return self in {DistributedRNGMode.UNIQUE_HOST_UNIQUE_DEVICE,
                    DistributedRNGMode.SAME_HOST_UNIQUE_DEVICE}


def host_id_devices_for_rng(mode="unique_host_unique_device"):
  if not DistributedRNGMode(mode).unique_host:
    return None
  return jnp.broadcast_to(jax.host_id(), (jax.local_device_count(),))


def specialize_rng_host_device(
    rng, host_id, axis_name, mode="unique_host_unique_device"):
  """Specializes a rng to the host/device we are on.

  Must be called from within a pmapped function.

  Args:
    rng: a jax.random.PRNGKey.
    host_id: the host ID to fold in, or None. Must be specified (not None) for
      the "unique_host_*" modes.
    axis_name: the axis of the devices we are specializing across.
    mode: str mode. Must be one of "unique_host_unique_device",
      "unique_host_same_device", "same_host_unique_device",
      "same_host_same_device".
  Returns:
    jax.random.PRNGKey specialized to host/device.
  """
  # Will throw an error if mode is not a valid enumeration.
  enum_mode = DistributedRNGMode(mode)
  if enum_mode.unique_host:
    # Note that we intentionally do NOT call `jax.host_id()` here, taking it as
    # an input instead. This is because we don't want to (effectively) use a
    # hard-coded Python int inside a potentially `pmap`ped context as that
    # results in different executable fingerprints across hosts.
    if host_id is None:
      raise ValueError(f"host_id must be given in RNG mode: {enum_mode}")
    rng = jax.random.fold_in(rng, host_id)
  if enum_mode.unique_device:
    rng = jax.random.fold_in(rng, jax.lax.axis_index(axis_name))
  return rng


def rendezvous():
  """Forces all hosts to check in."""
  with log_activity("rendezvous"):
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, "i"), "i")(x))
    if x[0] != jax.device_count():
      raise ValueError(f"Expected {jax.device_count()} got {x}")


class PeriodicAction:
  """An action that executes periodically (e.g. logging)."""

  def __init__(self,
               fn: Callable[[int, Dict[str, float]], None],
               interval_type: str,
               interval: float,
               start_time: float = 0.0,
               start_step: int = 0,
               run_async: bool = True,
               log_all_data: bool = False):
    """Initializes attributes for periodic action.

    Args:
      fn: Function representing the action to be run periodically. Takes global
        step and scalars returned by `Experiment.step` as arguments.
      interval_type: "secs" or "steps".
      interval: Interval between function calls.
      start_time: The start epoch time as a float to calculate time intervals
        with respect to.
      start_step: The start step number to calculate step intervals with respect
        to.
      run_async: boolean whether to run this perodic action in a background
        thread.
      log_all_data: boolean whether to accumulate scalar_outputs at each step.
    """
    if interval_type not in ["secs", "steps"]:
      raise ValueError(f"Unrecognized interval type {interval_type}.")
    self._fn = fn
    self._interval_type = interval_type
    self._interval = interval
    self._prev_time = start_time
    self._prev_step = start_step
    self._apply_fn_future = None
    if run_async:
      self._apply_fn = make_async(self._fn.__name__)(self._apply_fn)  # pylint: disable=no-value-for-parameter
    self.log_all_data = log_all_data
    self.log = {}

  def _apply_fn(self, step, steps_per_sec, scalar_outputs):
    """Runs periodic action, optionally dumping all intermediate logged data."""
    # Add data for this step to the log.
    self.log[step] = scalar_outputs
    # Note device_get copies from device <> host so is expensive.
    # However, JAX's DeviceArray has a cache which be reused for all
    # subsequent PeriodicActions that make the same call. Also, in async mode
    # this function runs in a background thread.
    log = jax.device_get(self.log)
    # Reset self.log here to prevent async weirdness
    self.log = {}
    # Steps per sec must be added after device-get
    log[step]["steps_per_sec"] = steps_per_sec

    for logged_step, logged_scalars in log.items():
      self._fn(logged_step, logged_scalars)

  def _apply_condition(self, t: float, step: int):
    if self._interval_type == "secs":
      return t - self._prev_time >= self._interval
    else:
      assert self._interval_type == "steps"  # error should've be caught in init
      return step % self._interval == 0

  def update_time(self, t: float, step: int):
    """Updates the internal time measurements."""
    self._prev_time = t
    self._prev_step = step

  def wait_to_finish(self):
    """Waits for any periodic actions running in own threads to complete."""
    if not (self._apply_fn_future is None or self._apply_fn_future.done()):
      logging.info("Waiting for a periodic action to finish...")
      self._apply_fn_future.result()

  def __call__(self, t: float, step: int, scalar_outputs: Dict[str,
                                                               jnp.ndarray]):
    """Calls periodic action if interval since last call sufficiently large.

    Args:
      t: The current epoch time as a float.
      step: The current step number.
      scalar_outputs: Scalars to be processed by the periodic action.
    """
    if self._apply_condition(t, step):
      steps_per_sec = (step - self._prev_step) / (t - self._prev_time)
      self._apply_fn_future = self._apply_fn(step, steps_per_sec,
                                             scalar_outputs)
      self.update_time(t, step)
    elif self.log_all_data:
      # Log data for dumping at next interval.
      self.log[step] = scalar_outputs


def debugger_fallback(f):
  """Maybe wraps f with a pdb-callback."""
  @functools.wraps(f)
  def inner_wrapper(*args, **kwargs):
    """Main entry function."""
    try:
      return f(*args, **kwargs)
    # KeyboardInterrupt and SystemExit are not derived from BaseException,
    # hence not caught by the post-mortem.
    except Exception as e:  # pylint: disable=broad-except
      if _JAXLINE_POST_MORTEM.value:
        pdb.post_mortem(e.__traceback__)
      raise
  return inner_wrapper


def evaluate_should_return_dict(f):
  """Prints a deprecation warning for old-usage of evaluate.

  As of cl/302532551 the evaluate method on an experiment should
  return a dictionary of scalars to be logged, just like the step method.

  Until May 1st 2020, evaluate is also allowed to return nothing (the
  older behavior). After that date, returning nothing will be an error.
  Please update old code. If you do not wish Jaxline to log anything for you,
  return an empty dictionary. Otherwise a dictionary of scalars may be returned
  like `step`.

  Args:
    f: The evaluate method.

  Returns:
    The evaluate function wrapped with a deprecation warning.
  """
  none_return_is_deprecated_msg = (
      "Your experiment\'s evaluate function returned no output, this is "
      "deprecated behavior. `evaluate` should now return a dictionary of "
      "scalars to log, just like `step`. Please update your code. "
      "After May 1st 2020 this code will be updated and returning None will "
      "error.")

  def evaluate_with_warning(*args, **kwargs):
    evaluate_out = f(*args, **kwargs)
    if evaluate_out is None:
      logging.log_first_n(logging.WARNING, none_return_is_deprecated_msg, 1)
      return {}
    return evaluate_out
  return evaluate_with_warning


# We use a global dictionary so that multiple different checkpoints can share
# underlying data.
GLOBAL_CHECKPOINT_DICT = {}


class InMemoryCheckpointer:
  """A Checkpointer reliant on an in-memory global dictionary."""

  def __init__(self, config, mode):
    self._max_checkpoints_to_keep = config.max_checkpoints_to_keep
    del mode

  def _override_or_insert(self, current_state, snapshot):
    """Update the current state based on a snapshot."""
    for sk, sv in snapshot.items():
      # Duck-typing for "is this a Jaxline Experiment class?".
      if (sk in current_state
          and hasattr(current_state[sk], "CHECKPOINT_ATTRS")
          and hasattr(current_state[sk], "NON_BROADCAST_CHECKPOINT_ATTRS")):
        for kk in sv.CHECKPOINT_ATTRS:
          setattr(current_state[sk], kk, getattr(sv, kk))
        for kk in sv.NON_BROADCAST_CHECKPOINT_ATTRS:
          setattr(
              current_state[sk], kk,
              jax.tree_map(copy.copy, getattr(sv, kk)))
      else:
        current_state[sk] = sv

  def get_experiment_state(self, ckpt_series):
    """Returns the experiment state for a given checkpoint series."""
    if ckpt_series not in GLOBAL_CHECKPOINT_DICT:
      active = threading.local()
      new_series = CheckpointNT(active, [])
      GLOBAL_CHECKPOINT_DICT[ckpt_series] = new_series
    if not hasattr(GLOBAL_CHECKPOINT_DICT[ckpt_series].active, "state"):
      GLOBAL_CHECKPOINT_DICT[ckpt_series].active.state = (
          config_dict.ConfigDict())
    return GLOBAL_CHECKPOINT_DICT[ckpt_series].active.state

  def save(self, ckpt_series) -> None:
    """Saves the checkpoint."""
    series = GLOBAL_CHECKPOINT_DICT[ckpt_series]
    active_state = self.get_experiment_state(ckpt_series)
    id_ = 0 if not series.history else series.history[-1].id + 1
    snapshot = copy.copy(active_state)
    series.history.append(SnapshotNT(id_, snapshot))
    if len(series.history) > self._max_checkpoints_to_keep:
      GLOBAL_CHECKPOINT_DICT[ckpt_series] = series._replace(
          history=series.history[-self._max_checkpoints_to_keep:])
    logging.info("Saved checkpoint %s with id %s.", ckpt_series, id_)

  def can_be_restored(self, ckpt_series) -> bool:
    """Returns whether or not a given checkpoint series can be restored."""
    return ((ckpt_series in GLOBAL_CHECKPOINT_DICT) and
            GLOBAL_CHECKPOINT_DICT[ckpt_series].history)

  def restore(self, ckpt_series) -> None:
    """Restores the checkpoint."""
    snapshot = GLOBAL_CHECKPOINT_DICT[ckpt_series].history[-1].pickle_nest
    current_state = self.get_experiment_state(ckpt_series)
    self._override_or_insert(current_state, snapshot)
    logging.info("Returned checkpoint %s with id %s.", ckpt_series,
                 GLOBAL_CHECKPOINT_DICT[ckpt_series].history[-1].id)

  def restore_path(self, ckpt_series) -> Optional[str]:
    """Returns the restore path for the checkpoint, or None."""
    if not self.can_be_restored(ckpt_series):
      return None
    return GLOBAL_CHECKPOINT_DICT[ckpt_series].history[-1].id

  def wait_for_checkpointing_to_finish(self) -> None:
    """Waits for any async checkpointing to complete."""


def disable_pmap_jit(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Disables pmaps/jits inside a function if `--jaxline_disable_pmap_jit=True`.

  Args:
    fn: function to be wrapped, with arbitrary call-signature and return type.

  Returns:
    A function that when called, calls fn within a chex context that strips out
    all pmaps and jits if `--jaxline_disable_pmap_jit=True`, and otherwise calls
    fn unmodified.
  """
  @functools.wraps(fn)
  def inner_wrapper(*args, **kwargs):
    if _JAXLINE_DISABLE_PMAP_JIT.value:
      with chex.fake_pmap_and_jit():
        return fn(*args, **kwargs)
    else:
      return fn(*args, **kwargs)
  return inner_wrapper

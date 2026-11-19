"""Diagnostics for shared-component dominance and rank collapse in H and Z.

Following Sun et al. (2026), arXiv:2611.21596: with a batch-normalised
projector, a component shared by all inputs can dominate the encoder
representation ``H`` and sits in the projector's null space, so a loss on the
endpoint ``Z`` never corrects it.
"""

from __future__ import annotations

from collections.abc import Callable

import lightning.pytorch as pl
import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep


def representation_stats(u: torch.Tensor) -> dict[str, torch.Tensor]:
    """Shared-energy fraction and effective rank of a batch of features.

    Args:
        u: Features of shape ``(..., d)``; leading dimensions are flattened
            into samples.

    Returns:
        ``shared``: fraction of the energy ``||U||_F^2`` in the batch mean (the
        component shared by every sample). ``variation`` is ``1 - shared``.
        ``erank``: exp of the entropy of the normalised eigenvalues of the
        sample covariance, between 1 and ``min(d, N - 1)``.
    """
    u = u.detach().flatten(0, -2).float()
    mean = u.mean(dim=0)
    centered = u - mean
    shared = u.shape[0] * mean.square().sum() / u.square().sum()
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0)
    p = eigenvalues / eigenvalues.sum()
    erank = torch.exp(-(p * torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))).sum())
    return {"shared": shared, "variation": 1 - shared, "erank": erank}


class HZDiagnostics(pl.Callback):
    """Log the stats of the encoder representation H and endpoint Z during training.

    Hooks the forward pass of each view's encoder (giving Z) and of a
    submodule you choose (giving H), so no extra forward pass is run and
    BatchNorm statistics are untouched. Logged as ``diag/H{view}/{stat}`` and
    ``diag/Z{view}/{stat}``.

    Args:
        h_module: Maps ``(model, view)`` to the submodule whose output is H,
            e.g. ``lambda model, view: model.encoders[view].backbone``.
        every_n_steps: Logging period, in training steps.
    """

    def __init__(
        self,
        h_module: Callable[[BaseDeep, int], nn.Module],
        every_n_steps: int = 50,
    ) -> None:
        self.h_module = h_module
        self.every_n_steps = every_n_steps
        self._captured: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseDeep) -> None:
        for view, encoder in enumerate(pl_module.encoders):
            for name, module in (
                (f"H{view}", self.h_module(pl_module, view)),
                (f"Z{view}", encoder),
            ):
                self._handles.append(
                    module.register_forward_hook(
                        lambda _m, _inp, out, name=name: self._captured.__setitem__(
                            name, out
                        )
                    )
                )

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseDeep, outputs, batch, batch_idx
    ) -> None:
        if trainer.global_step % self.every_n_steps == 0:
            for name, features in self._captured.items():
                for stat, value in representation_stats(features).items():
                    pl_module.log(f"diag/{name}/{stat}", value)
        self._captured.clear()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: BaseDeep) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

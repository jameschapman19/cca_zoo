# Deep Methods

`cca_zoo.deep` trains neural encoders, one per view, with CCA-style and self-supervised
losses. The models are [PyTorch Lightning](https://lightning.ai/) modules and need the
`deep` extra:

```bash
pip install cca-zoo[deep]
```

---

## Workflow

A model takes one `nn.Module` per view, each mapping that view to `n_components` outputs,
and is trained and used with a Lightning `Trainer`:

```python
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cca_zoo.deep import DCCA, MultiviewDataset

model = DCCA(
    n_components=4,
    encoders=[
        nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 4)),
        nn.Sequential(nn.Linear(80, 64), nn.ReLU(), nn.Linear(64, 4)),
    ],
)

train_loader = DataLoader(MultiviewDataset([X1, X2]), batch_size=128, shuffle=True)
val_loader = DataLoader(MultiviewDataset([X1_val, X2_val]), batch_size=256)
test_loader = DataLoader(MultiviewDataset([X1_test, X2_test]), batch_size=256)

trainer = L.Trainer(max_epochs=50)
trainer.fit(model, train_loader, val_loader)

# Canonical variates, one array per view
z1, z2 = (torch.cat(z).numpy() for z in zip(*trainer.predict(model, test_loader)))
```

Batches are dictionaries with a `"views"` list of tensors. `MultiviewDataset` builds them
from in-memory arrays; any `Dataset` whose items are `{"views": [x_1, ..., x_m]}` works
(a `TensorDataset` yields tuples and does not).

Every loss term is logged under `train/`, `val/` and `test/`, so callbacks such as
`EarlyStopping(monitor="val/objective")` and `ModelCheckpoint` work as usual.

### Outputs

Calling the model, `model([x1, x2])`, returns each view's raw encoding.
`trainer.predict` returns **canonical variates**: when training ends, a linear CCA is
fitted to the training encodings and applied to every prediction, so each view's outputs
are uncorrelated with unit variance and ordered by canonical correlation, as `transform`
gives for the other models. Call `model.fit_cca(loader)` to refit that projection on
other data, or to predict from a model that was not trained with `trainer.fit`.

Evaluate with `cca_zoo.metrics` on the predicted arrays:

```python
from cca_zoo.metrics import average_pairwise_correlations, pairwise_correlations

average_pairwise_correlations(pairwise_correlations([z1, z2]))  # per dimension
```

### Checkpoints

Hyperparameters are saved with the model; the encoders and decoders are modules, so pass
them again when loading:

```python
model = DCCA.load_from_checkpoint(path, encoders=[make_encoder(100), make_encoder(80)])
```

---

## Models

| Model | Views | Loss |
|---|---|---|
| `DCCA` | 2, or any with a multiview `objective` | Deep CCA (Andrew et al., 2013); pluggable |
| `DCCAEY` | any | Eckart-Young loss (Chapman et al., 2024); stable on small batches |
| `DMCCA` | any | Sum of pairwise CCA losses |
| `DGCCA` | any | Generalized CCA (Benton et al., 2019) |
| `DTCCA` | any | Tensor CCA (Wong et al., 2021) |
| `DCCANOI` | any | Nonlinear orthogonal iterations (Wang et al., 2015) |
| `DCCASDL` | any | Alignment plus within-view soft decorrelation (Chang et al., 2018) |
| `BarlowTwins` | any | Cross-correlation to the identity (Zbontar et al., 2021) |
| `VICReg` | any | Variance, invariance and covariance terms (Bardes et al., 2022) |
| `DCCAE` | any | Deep CCA plus per-view reconstruction (Wang et al., 2015) |
| `SplitAE` | any | Every view reconstructed from all encodings |
| `DVCCA` | any | Variational: a latent inferred from the first view generates every view (Wang et al., 2016) |

As on the linear side, only `DCCA`'s default loss is two-view, like `CCA`; `DMCCA` and
`DGCCA` generalise it as `MCCA` and `GCCA` do. Losses defined between two views
(`BarlowTwins`, `VICReg`, `DCCASDL`, and `DCCAE`'s default `MCCALoss`) are summed over
pairs of views, and with two views each is the published loss.

`DCCA` and `DCCAE` take an `objective` from `cca_zoo.deep.objectives`, or any module
mapping a list of encodings to a scalar:

| Objective | Loss |
|---|---|
| `CCALoss` | $-\lVert \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2} \rVert_F^2$ (two views) |
| `MCCALoss` | Sum of pairwise `CCALoss` |
| `GCCALoss` | Minus the top $k$ eigenvalues of $\sum_i H_i H_i^\top$ |
| `TCCALoss` | Minus the norm of the whitened cross-moment tensor |

```python
from cca_zoo.deep.objectives import MCCALoss

model = DCCA(n_components=4, encoders=[e1, e2, e3], objective=MCCALoss())
```

The autoencoder models also take decoders. `DCCAE` decodes each view from its own
encoding and `SplitAE` from the concatenation of all encodings (decoder input
`n_views * n_components`). `DVCCA` has a single `encoder`, of the first view, which
outputs `2 * n_components` values, a mean and a log-variance; every view is decoded from
the latent. Its prediction is one array, the posterior mean, with no linear CCA.

```python
from cca_zoo.deep import DCCAE

model = DCCAE(n_components=4, encoders=[e1, e2], decoders=[d1, d2], lam=0.1)
```

`DCCAEY` also accepts `"independent_views"` in a batch, an independent batch whose
encodings give an unbiased estimate of the loss's penalty term.

---

## Custom models

Subclass `BaseDeep` and implement `loss(batch)`, returning a dictionary whose
`"objective"` entry is minimised; every entry is logged. Override `configure_optimizers`
to change the optimiser.

```python
from cca_zoo.deep import BaseDeep


class MyModel(BaseDeep):
    def loss(self, batch):
        z1, z2 = self(batch["views"])
        objective = ((z1 - z2) ** 2).mean()
        return {"objective": objective}
```

---

## Tips

- **Batch size matters.** The CCA losses estimate covariances per mini-batch; use batches
  well above `n_components`, or `DCCAEY` when they must be small.
- **Encoders end in exactly `n_components` outputs** (`2 * n_components` for `DVCCA`).

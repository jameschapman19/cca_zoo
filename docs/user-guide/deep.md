# Deep Methods

The `cca_zoo.deep` module provides CCA variants that use neural network encoders to learn
nonlinear representations of each view. All deep models are built on
[PyTorch Lightning](https://lightning.ai/) and require the `[deep]` extra:

```bash
pip install cca-zoo[deep]
```

---

## Design

Deep models in CCA-Zoo are **encoder-only**: you supply your own `nn.Module` for each view,
and the model optimises a CCA-based objective over the encoder outputs. This keeps the models
flexible — any architecture (MLP, CNN, Transformer) can be used.

```python
import torch.nn as nn
from cca_zoo.deep import DCCA

# Define encoders for each view
encoder1 = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 16))
encoder2 = nn.Sequential(nn.Linear(80, 64), nn.ReLU(), nn.Linear(64, 16))

model = DCCA(
    latent_dimensions=8,
    encoders=[encoder1, encoder2],
    lr=1e-3,
)
```

All deep models are Lightning modules and can be trained with a standard
`lightning.Trainer`:

```python
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch

dataset = TensorDataset(
    torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32)
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

trainer = L.Trainer(max_epochs=50)
trainer.fit(model, loader)
```

After training, `transform` takes a DataLoader and returns the encoded representations:

```python
test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X1_test, dtype=torch.float32),
        torch.tensor(X2_test, dtype=torch.float32),
    ),
    batch_size=256,
)

z1, z2 = model.transform(test_loader)
```

---

## Available models

### DCCA — Deep CCA

The original Deep CCA (Andrew et al. 2013). Optimises a differentiable CCA loss on
mini-batches of encoder outputs.

The `objective` parameter controls which CCA loss is used:

```python
from cca_zoo.deep import DCCA
from cca_zoo.deep.objectives import CCALoss, GCCALoss

model = DCCA(latent_dimensions=8, encoders=[e1, e2], objective=CCALoss(eps=1e-4))
```

Available objectives (from `cca_zoo.deep.objectives`):

| Class | Description |
|---|---|
| `CCALoss` | Negative sum of squared singular values of $\Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2}$ |
| `MCCALoss` | Sum of pairwise `CCALoss` values |
| `GCCALoss` | Negative sum of top-$k$ eigenvalues of $\sum_i H_i H_i^\top$ |
| `TCCALoss` | Negative Frobenius norm of whitened cross-moment tensor |

### DCCA_EY — Eckart-Young objective

Uses the Eckart-Young decomposition as the differentiable objective (Benton et al. 2022).
Tends to be more stable than the original CCA loss on small batches.

```python
from cca_zoo.deep import DCCA_EY

model = DCCA_EY(latent_dimensions=8, encoders=[e1, e2])
```

### DCCA_NOI — Non-linear Orthogonal Iterations

Wang et al. 2015. An iterative approach that alternately optimises each encoder
while holding the others fixed.

```python
from cca_zoo.deep import DCCA_NOI

model = DCCA_NOI(latent_dimensions=8, encoders=[e1, e2])
```

### DCCA_SDL — Stochastic Decorrelation Loss

Chang et al. 2018. Adds an explicit decorrelation term that penalises off-diagonal
cross-covariance entries.

```python
from cca_zoo.deep import DCCA_SDL

model = DCCA_SDL(latent_dimensions=8, encoders=[e1, e2])
```

### DCCAE — Deep CCA with Autoencoders

Wang et al. 2015. Adds a reconstruction loss to DCCA, encouraging each encoder to
also be a good autoencoder. Requires a matching decoder per view.

```python
from cca_zoo.deep import DCCAE

decoder1 = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 100))
decoder2 = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 80))

model = DCCAE(
    latent_dimensions=8,
    encoders=[e1, e2],
    decoders=[decoder1, decoder2],
    lam=0.01,  # reconstruction loss weight
)
```

### DVCCA — Deep Variational CCA

Wang et al. 2016. A variational autoencoder-based formulation where the shared
latent variable has an explicit probabilistic prior.

```python
from cca_zoo.deep import DVCCA

model = DVCCA(latent_dimensions=8, encoders=[e1, e2], decoders=[d1, d2])
```

### DTCCA — Deep Tensor CCA

Wong et al. 2021. Deep extension of TCCA, capturing higher-order correlations via
a tensor loss on the encoder outputs.

```python
from cca_zoo.deep import DTCCA

model = DTCCA(latent_dimensions=8, encoders=[e1, e2, e3])
```

### SplitAE — Split Autoencoder

A simple baseline that concatenates views, encodes them to a shared latent space, and
decodes back to each view independently.

```python
from cca_zoo.deep import SplitAE

model = SplitAE(latent_dimensions=8, encoders=[e1, e2], decoders=[d1, d2])
```

### BarlowTwins

Zbontar et al. 2021. A self-supervised objective that encourages the cross-correlation
matrix of the two encoded views to be close to the identity.

```python
from cca_zoo.deep import BarlowTwins

model = BarlowTwins(latent_dimensions=8, encoders=[e1, e2], lam=5e-3)
```

### VICReg

Bardes et al. 2022. Regularises the representations via **V**ariance, **I**nvariance,
and **C**ovariance terms.

```python
from cca_zoo.deep import VICReg

model = VICReg(latent_dimensions=8, encoders=[e1, e2])
```

---

## Full training example

```python
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from cca_zoo.datasets import JointData
from cca_zoo.deep import DCCA

# Simulate data
data = JointData(
    n_views=2, n_samples=1000, n_features=[100, 80], latent_dimensions=4, random_state=0
)
views = data.sample()

X1 = torch.tensor(views[0], dtype=torch.float32)
X2 = torch.tensor(views[1], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X1, X2), batch_size=64, shuffle=True)


# Build encoders
def make_encoder(in_features: int, latent_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, latent_dim),
    )


model = DCCA(
    latent_dimensions=4,
    encoders=[make_encoder(100, 4), make_encoder(80, 4)],
    lr=1e-3,
)

trainer = L.Trainer(max_epochs=30, enable_progress_bar=True)
trainer.fit(model, train_loader)

# Evaluate
test_loader = DataLoader(TensorDataset(X1, X2), batch_size=256)
z1, z2 = model.transform(test_loader)
print("Representation shape:", z1.shape)  # (1000, 4)
```

---

## Tips

- **Batch size matters.** CCA-based losses estimate covariance from mini-batches. Use
  `batch_size ≥ 4 * latent_dimensions` for stable estimates.
- **Encoder output dimension ≥ `latent_dimensions`.** The model projects down inside the
  loss; do not make encoders narrower than the requested latent space.
- **Use `DCCA_EY` for small batches.** The Eckart-Young objective is more numerically stable
  than the original `CCALoss` when batch sizes are small.
- **Score after training** via `model.score(loader)`, which fits a linear `MCCA` on the
  learned representations to report canonical correlations.

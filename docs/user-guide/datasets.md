# Datasets

`cca_zoo.datasets` provides utilities for generating synthetic multiview data and loading
small real-world datasets.

---

## JointData — simulated multiview data

`JointData` generates data from a **linear latent variable model**:

$$
X_i = Z W_i^\top + E_i
$$

where:

- $Z \in \mathbb{R}^{n \times k}$ is the shared latent variable ($k$ = `latent_dimensions`)
- $W_i \in \mathbb{R}^{p_i \times k}$ is the view-specific loading matrix (drawn once at construction)
- $E_i$ is independent Gaussian noise, with variance controlled by `signal_to_noise`

The loading matrices are fixed at construction so that successive calls to `sample()` share
the same generative parameters.

```python
from cca_zoo.datasets import JointData

data = JointData(
    n_views=2,
    n_samples=200,
    n_features=[50, 40],  # different feature counts per view
    latent_dimensions=2,
    signal_to_noise=2.0,  # higher = less noise
    random_state=0,
)

train_views = data.sample()  # list of 2 arrays: (200, 50), (200, 40)
test_views = data.sample()  # independent sample from the same model
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_views` | `int` | `2` | Number of views |
| `n_samples` | `int` | `100` | Observations per call to `sample()` |
| `n_features` | `int` or `list[int]` | `10` | Features per view (scalar broadcasts) |
| `latent_dimensions` | `int` | `1` | Shared latent dimension |
| `signal_to_noise` | `float` or `list[float]` | `1.0` | SNR per view (higher = less noise) |
| `random_state` | `int` or `None` | `None` | Seed for reproducibility |

### Usage patterns

```python
# Same features for all views (scalar broadcasts)
data = JointData(
    n_views=3, n_samples=100, n_features=20, latent_dimensions=2, random_state=0
)

# Different SNR per view
data = JointData(
    n_views=2, n_samples=100, n_features=20, signal_to_noise=[4.0, 1.0], random_state=0
)

# Callable shorthand (same as sample())
views = data()
```

---

## Toy real-world datasets

Two small real-world datasets are included for quick experimentation:

### `load_linnerud`

Wraps `sklearn.datasets.load_linnerud`. Returns two arrays:

- **View 1:** exercise measurements (chin-ups, sit-ups, jumps) — shape `(20, 3)`
- **View 2:** physiological measurements (weight, waist, pulse) — shape `(20, 3)`

```python
from cca_zoo.datasets import load_linnerud

exercise, physiological = load_linnerud()
print(exercise.shape)  # (20, 3)
print(physiological.shape)  # (20, 3)
```

### `load_breast_cancer`

Wraps `sklearn.datasets.load_breast_cancer`. Splits the 30 features into two halves to create
a two-view dataset:

- **View 1:** first 15 features — shape `(569, 15)`
- **View 2:** last 15 features — shape `(569, 15)`

```python
from cca_zoo.datasets import load_breast_cancer

view1, view2 = load_breast_cancer()
print(view1.shape)  # (569, 15)
print(view2.shape)  # (569, 15)
```

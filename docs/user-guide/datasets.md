# Datasets

`cca_zoo.datasets` provides utilities for generating synthetic multiview data and loading
small real-world datasets.

---

## `make_joint_data` — simulated multiview data

`make_joint_data` generates data from a **linear latent variable model**, in the style of
`sklearn.datasets`' `make_*` generators:

$$
X_i = Z W_i^\top + E_i
$$

where:

- $Z \in \mathbb{R}^{n \times k}$ is the shared latent variable ($k$ = `n_components`)
- $W_i \in \mathbb{R}^{p_i \times k}$ is the view-specific loading matrix
- $E_i$ is independent Gaussian noise, with variance controlled by `signal_to_noise`

For independent training and test sets, generate once and split every view together with
`sklearn.model_selection.train_test_split`:

```python
from sklearn.model_selection import train_test_split

from cca_zoo.datasets import make_joint_data

views = make_joint_data(
    n_samples=400,
    n_features=[50, 40],  # different feature counts per view
    n_components=2,
    signal_to_noise=2.0,  # higher = less noise
    random_state=0,
)
X1_train, X1_test, X2_train, X2_test = train_test_split(*views, random_state=0)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_samples` | `int` | `100` | Observations |
| `n_features` | `int` or `list[int]` | `10` | Features per view (scalar broadcasts) |
| `n_views` | `int` | `2` | Number of views |
| `n_components` | `int` | `1` | Shared latent dimension |
| `signal_to_noise` | `float` or `list[float]` | `1.0` | SNR per view (higher = less noise) |
| `random_state` | `int`, `RandomState` or `None` | `None` | Seed for reproducibility |
| `return_latent` | `bool` | `False` | Also return the latent variable `z` |

### Usage patterns

```python
# Three views, same feature count (scalar broadcasts)
views = make_joint_data(n_samples=100, n_features=20, n_views=3, n_components=2)

# Different SNR per view
views = make_joint_data(n_samples=100, n_features=20, signal_to_noise=[4.0, 1.0])

# The true latent variable too, e.g. to check recovery
views, z = make_joint_data(n_samples=100, n_components=2, return_latent=True)
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

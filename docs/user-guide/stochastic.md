# Stochastic Methods

The `cca_zoo.stochastic` module provides mini-batch CCA methods, for datasets too large to fit a
full-batch gradient step in memory.

---

## StochasticCCAEY

`StochasticCCAEY` fits the same unconstrained Eckart-Young (EY) objective as
[`cca_zoo.linear.CCAEY`](linear.md#ey-loss-methods) — see that page's Background section for the
objective itself — by mini-batch SGD instead of full-batch L-BFGS-B, with the adaptive step of
sklearn's `SGDRegressor(learning_rate="adaptive")`:

```python
from cca_zoo.stochastic import StochasticCCAEY

model = StochasticCCAEY(n_components=2, batch_size=128, max_iter=200)
model.fit([X1, X2])
```

**When to use:** the dataset doesn't fit comfortably in memory for a full-batch gradient step, or
is naturally streamed/out-of-core. Otherwise, `CCAEY` is simpler to tune (no `learning_rate` or
`batch_size`) and converges more predictably.

`learning_rate` is relative: the step is `learning_rate / L`, where `L` is the largest variance
along any direction of any view, so the default suits data at any scale without standardising.
`batch_size` trades off gradient-estimate noise against per-step cost. A mini-batch step cannot
converge at a fixed size: it hovers at a distance set by the batches' noise. So, as in sklearn,
whenever the epoch loss fails to improve by `tol` for `n_iter_no_change` epochs the step is divided
by 5, and the fit stops once the step is negligible. Too large a `learning_rate` for a small batch
can still diverge; the error says so. `random_state`
controls both the initial weights and the batch sampling order, so it must be fixed for
reproducible fits.

# Stochastic Methods

The `cca_zoo.stochastic` module provides mini-batch CCA methods, for datasets too large to fit a
full-batch gradient step in memory.

---

## StochasticCCAEY

`StochasticCCAEY` fits the same unconstrained Eckart-Young (EY) objective as
[`cca_zoo.linear.CCAEY`](linear.md#ey-loss-methods) — see that page's Background section for the
objective itself — by mini-batch momentum SGD instead of full-batch L-BFGS-B:

```python
from cca_zoo.stochastic import StochasticCCAEY

model = StochasticCCAEY(
    latent_dimensions=2, learning_rate=0.01, batch_size=128, max_iter=200
)
model.fit([X1, X2])
```

**When to use:** the dataset doesn't fit comfortably in memory for a full-batch gradient step, or
is naturally streamed/out-of-core. Otherwise, `CCAEY` is simpler to tune (no `learning_rate` or
`batch_size`) and converges more predictably.

`batch_size` trades off gradient-estimate noise against per-step cost; `learning_rate` and the
momentum term interact with it the same way they do for any mini-batch SGD method — too large a
learning rate for a given batch size can diverge, too small converges slowly. `random_state`
controls both the initial weights and the batch sampling order, so it must be fixed for
reproducible fits.

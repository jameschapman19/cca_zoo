# The case for SSL-EY over SIGReg-style regularisation

*Short position paper. Claims marked [verified] were checked in this repo or in the cited text; [untested] claims are arguments, not results.*

## Setting
Both LeJEPA (Balestriero & LeCun, 2025) and SSL-EY (Chapman et al., 2024) pair an alignment term with a penalty that prevents collapse. LeJEPA's penalty, SIGReg, pushes each view's embedding towards N(0, I) using random 1-D Epps–Pulley tests. EY's penalty is the Eckart–Young loss, `-2 tr(C) + tr(V V')`, which constrains second moments through the cross-covariance.

## Arguments for EY

**1. Its minibatch gradient is unbiased; SIGReg's is not.** EY's penalty is unbiased when `V'` comes from an independent batch. The single-batch form is biased, and both facts are checked by Monte Carlo in `tests/test_loss.py` [verified]. SIGReg's per-batch statistic carries an O(1/N) bias, which the LeJEPA paper acknowledges (its Theorem 6) while leaving unbiased U-statistic alternatives unexplored. The repo's `_sigreg` keeps the diagonal terms, so it has that bias [verified]. We have an unbiased U-statistic drop-in (`cf_distance`, tested unbiased) [verified]. Whether the bias costs accuracy is [untested].

**2. Its optimum is characterised.** The EY loss is unconstrained and stationary exactly at the canonical directions (repo docstring, `cca_zoo/_utils/_ey.py`); the minimiser is the top singular subspace of the cross-covariance, with variance `1 + ρ` along each direction (observed in CCAEY output) [verified]. In the part of the LeJEPA paper I read, the guarantees concern the power of the distributional test and gradient bounds, not what the joint predictive-plus-SIGReg objective converges to.

**3. It asks only for what the linear-probe argument needs.** LeJEPA's own Lemmas 1 and 2 (linear probes) show that an anisotropic embedding does worse than an isotropic one at the same total variance: a statement about eigenvalues. A second-moment penalty already targets that. Gaussianity of the marginals enters only in the kNN/kernel theorem. For linear evaluation, SIGReg's extra constraint is unneeded by the paper's own argument.

**4. No slices, no quadrature.** EY has no random directions, no slice count (256 by default) and no integration grid; its penalty is deterministic given the batches. [untested: whether this reduces gradient noise.]

**5. It generalises.** The same objective covers CCA, PLS (`shrinkage`) and more than two views, and yields ordered, interpretable components. SIGReg-style isotropy discards the ordering by design.

## Limits of the case
- LeJEPA has scale evidence (79% ImageNet linear probe with ViT-H/14; Galaxy10) that SSL-EY has not been shown to match here [untested].
- The Gaussianity SIGReg enforces may help non-linear probes (Theorem 1, kNN and kernel regression) [untested for EY].
- No independent head-to-head of LeJEPA and EY exists that I could find; small CIFAR-10 evidence suggests VICReg-type second-moment penalties and SIGReg are roughly tied.
- Two-batch unbiasedness costs a second forward pass, or half the batch.

## Decisive test
`DCCAEY` (single-batch and two-batch), `LeJEPA`, with the U-statistic variant of SIGReg as a third arm: same encoder and data, batch sizes 16–1024, linear probe plus embedding spectrum. EY wins the argument if the unbiased variants match or beat the biased ones at small batch and EY is within noise of SIGReg at large batch. Otherwise the case reduces to points 2 and 5, which are about interpretability, not accuracy.

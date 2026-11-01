# Prototypes

Two experiments, not library models. Run from this directory.

## `ajive_as_cca.py`: AJIVE's joint step is a CCA method

With each view cut to its signal subspace, AJIVE's joint scores are CCA's for two
views and multiset CCA (`GCCA`) for more, to numerical precision:

- Two views: the squared singular values of the stacked bases minus one equal the
  canonical correlations of the PCA-reduced views (0.9320 and 0.8923 in both), and each
  joint score correlates at 1.0000 with the sum of the two canonical variates.
- Three views: the principal cosines between AJIVE's joint space and `GCCA` on the
  PCA-reduced views are 1 and 1, and 0.999 and 0.998 against `GCCA` with shrinkage 0.5
  on the full views.
- Out of sample (150 train, 150 test, wide views, 2% and 5% shared variance): AJIVE
  recovers the shared factors as well as ridge-shrunk `GCCA` and `IncompleteGCCA`
  (0.97 and 0.99) and has slightly higher held-out correlation (0.92 against 0.90; 0.98
  against 0.97). Its automatic ranks do as well as the true ones.

So AJIVE is a hard rank cut (the signal rank) used as the regulariser in place of a ridge
penalty, plus the individual structure and variance accounting.

## `ey_shared_private.py`, `compare_shared_private.py`: an Eckart-Young shared/private fit

Fits the block covariance of the views, in squared Frobenius error, by a rank-k shared
part plus a low-rank private part per view that touches only that view's diagonal block.
The gradient is analytic and matches finite differences to 4e-8, with views missing; every
term is an n by k product, so wide views cost O(n p k).

On the simulator (4 datasets per cell; shared recovery, private recovery):

| regime | views missing | AJIVE (true ranks) | EY shared/private |
|---|---|---|---|
| moderate (10% shared, 60-80 features) | 0% | 0.95, 0.95 | 0.95, 0.94 |
| moderate | 30% | 0.82, 0.95 (views mean-filled) | 0.89, 0.91 (native) |
| faint (2% shared, 300-400 features) | 0% | 0.97, 0.98 | 0.39, 0.89 |
| fainter (1% shared) | 0% | 0.93, 0.98 | 0.17, 0.91 |

It works where the shared signal is strong and handles missing views natively, but
fails when the shared signal is faint next to large private variance, even started from
the good `IncompleteGCCA` solution. The Frobenius error of the raw covariance is the
PLS criterion, not the CCA one: it chases the high-variance private directions and the
chance cross-covariance between them. It needs whitening to compete, and a whitened
version on the signal subspaces is AJIVE.

Caveats: four datasets per cell, `IncompleteGCCA` at a fixed shrinkage of 0.5 (the
benchmark tunes it), AJIVE given the true individual ranks in this comparison, and timings
taken while other jobs used the machine.

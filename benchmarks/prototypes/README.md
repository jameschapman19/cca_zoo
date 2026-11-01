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

## `ajive_reference.py`: AJIVE against the mvlearn implementation

mvlearn's AJIVE follows the authors' reference code. Fixed ranks give the same
decomposition: over six designs and four datasets each, the joint spaces have minimum
principal cosine 1.000000 and the individual parts differ by about 1e-15. With the ranks
selected automatically, both select the same joint rank in all 24 datasets (including 0 where
there is no joint structure), although ours uses the random direction bound only. mvlearn
needs an older numpy, so its half runs in a separate environment (see the docstring).

## `ey_shared_private.py`, `compare_shared_private.py`: an Eckart-Young shared/private fit

Fits the block covariance of the views, in squared Frobenius error, by a rank-k shared
part plus a low-rank private part per view that touches only that view's diagonal block.
The gradient is analytic and matches finite differences to 4e-8, with views missing; every
term is an n by k product, so wide views cost O(n p k).

**On raw views it fails when the shared signal is faint** (shared recovery 0.39 at 2% shared
variance and 0.17 at 1%, against 0.93-0.97 for AJIVE), even started from a good solution. The
Frobenius error of the raw covariance is the PLS criterion, not the CCA one: it chases the
high-variance private directions and the chance cross-covariance between them.

**On ridge-whitened views it works.** `WhitenedSharedPrivateEY` maps each view to
`(x - mean) V diag(d)^(-1/2)`, with `x = U S V'` and `d = (1 - c) S^2 / (n - 1) + c`, the spectrum
of `(1 - c) Sigma + c I` that CCAEY's `shrinkage` uses for its constraint. The Frobenius fit of
the block covariance of those coordinates is the Eckart-Young loss on the whitened covariance:
`c = 1` is the raw fit above, small `c` is CCA-like, and in between the private part still sees
variance. The coordinates have at most `n - 1` dimensions, so wide views cost the SVD alone.

Shared and private recovery on the simulator (4 datasets per cell; the shared and private ranks
are the true ones for every method; AJIVE gets mean-filled views where samples lack a view):

| regime | views missing | raw fit | whitened, c=0.5 | AJIVE | IncompleteGCCA |
|---|---|---|---|---|---|
| moderate (10% shared, 60-80 features) | 0% | 0.95, 0.94 | 0.95, 0.95 | 0.95, 0.95 | 0.95, - |
| moderate | 30% | 0.89, 0.91 | 0.90, 0.95 | 0.82, 0.95 | 0.90, - |
| faint (2% shared, 300-400 features) | 0% | 0.39, 0.89 | 0.97, 0.98 | 0.97, 0.98 | 0.96, - |
| faint | 30% | 0.29, 0.90 | 0.85, 0.95 | 0.82, 0.98 | 0.74, - |
| fainter (1% shared) | 0% | 0.17, 0.91 | 0.92, 0.97 | 0.93, 0.98 | 0.82, - |
| fainter | 30% | 0.12, 0.91 | 0.64, 0.91 | 0.76, 0.98 | 0.25, - |

So on complete data it matches AJIVE, and with views missing it handles them natively and is
ahead of mean-filled AJIVE in the moderate and faint regimes and behind it in the hardest cell.
It is a single objective rather than a pipeline of SVD steps, so it can take penalties (sparse
loadings, say) that AJIVE cannot.

**Wrong ranks** (`robustness.py`, 8 datasets per cell, 2% shared variance, 2 shared and 3 private
factors per view; shared rank k in 1-3, private rank r in 1, 3 and 5; shared and private recovery):

| views missing | method | right ranks (k=2, r=3) | worst over the nine (k, r) |
|---|---|---|---|
| 0% | whitened EY | 0.97, 0.98 | 0.94, 0.87 |
| 0% | AJIVE | 0.97, 0.98 | 0.51, 0.84 |
| 30% | whitened EY | 0.83, 0.94 | 0.62, 0.84 |

On complete data the whitened fit recovers the shared factors at 0.94 or better for every
combination of wrong ranks, including a private rank of 1 against three true private factors,
where AJIVE's shared recovery falls to 0.51: too few individual components leave private
variation to leak into its joint scores. The private recovery drops to about 0.85 for both when
the shared rank is too large (k=3), because the extra shared dimension takes a private factor.
With 30% of views missing it is more sensitive: 0.83 at the right ranks, 0.71 when the private
rank is too large (r=5), and 0.62 when the shared rank is right but the private rank is 1.
Selecting the ranks is therefore still needed, most of all with missing views.

Caveats: four datasets per cell in the comparison above; `IncompleteGCCA` is at a
fixed shrinkage of 0.5 (the benchmark tunes it); and timings were taken while other jobs used the
machine. Not yet tried: real data, a rank-selection rule, or a version that avoids the SVD for very large n.

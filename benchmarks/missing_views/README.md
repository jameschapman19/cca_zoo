# Missing-view benchmark

Simulated multi-omics-like views with whole samples missing from each view,
comparing the missing-view CCA models with what practitioners use.

```bash
uv pip install mofapy2 matplotlib
python run.py --out output/results.json --methods "IncompleteGCCA" "CCAEY (masked)" \
    "GCCA, complete cases" "GCCA, zero-filled" "GFA, zero-filled"
python run.py --out output/results_mofa.json --methods "MOFA+ (default)" \
    "MOFA+ (best of 5 random)"
python plot.py output/results.json output/results_mofa.json output/
```

## Design

Three views of 60, 80 and 40 features on 150 samples. Two factors are shared by
all views and two are private to each view; every feature has unit variance, of
which 5% or 10% is shared and 50% private. Loadings are dense, or sparse (20% of
features per factor). Whole views are removed from 0 to 60% of samples, completely
at random (MCAR) or with a probability that depends on the first shared factor
(MNAR). Eight datasets per setting.

Methods: `IncompleteGCCA`; masked `CCAEY`; `GCCA` on complete cases only; `GCCA`
with missing views zero-filled after centring; `GFA` fitted to zero-filled views;
and MOFA+ (mofapy2) as run by default, and as the best of five random starts by
ELBO. The CCA-family shrinkage is chosen by three-fold cross-validation of the
held-out correlation. MOFA+ and GFA get as many factors as were simulated
(shared plus private) and are then reduced to the two factors that explain the
most variance in every view, as a practitioner would pick them.

Metrics: **recovery**, the mean squared cosine of the principal angles between the
estimated and true shared factors; **imputation R²**, of the withheld samples'
noise-free shared signal, which is all that the other views can tell about a view.

## Caveats

- mofapy2 starts from PCA and ignores its seed, so "best of 5" needs random starts,
  which this benchmark forces. Its default fit is deterministic.
- `predict` for the CCA-family models maps the mean of the other views' scores to
  a view by least squares. That is not a posterior mean and is not calibrated to how
  many views a sample has, so their imputation can be worse than predicting zero.
  GFA and MOFA+ are calibrated.
- Simulated Gaussian data with a known factor structure. Nothing here says how
  the methods compare on real cohorts.

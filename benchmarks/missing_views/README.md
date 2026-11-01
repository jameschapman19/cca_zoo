# Missing-view benchmark

Simulated multi-omics-like views with whole samples missing from each view,
comparing the missing-view CCA models with what practitioners use.

```bash
uv pip install mofapy2 matplotlib
python run.py --regime base --out output/results_base.json
python run.py --regime hard --out output/results_hard.json
python plot.py output/results_base.json output/results_hard.json output/
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
estimated and true shared factors; **imputation skill**, ``1 - SSE / SSE_mean`` on
the withheld samples' noise-free shared signal, where ``SSE_mean`` is the error of
predicting each view's observed mean. The mean is the baseline because, when the
shared signal is a small share of the variance, the feature means estimated from
the observed samples are about as noisy as the signal itself, so R² against the
signal is pushed below zero for every method (the results files also hold it).

## Caveats

- mofapy2 starts from PCA and ignores its seed, so "best of 5" needs random starts,
  which this benchmark forces. Its default fit is deterministic.
- GFA is fitted to zero-filled views here, as it cannot mask them. Its imputation is
  still its posterior given each sample's own views.
- Simulated Gaussian data with a known factor structure. Nothing here says how
  the methods compare on real cohorts.

## Hard regime: faint shared signal

`python run.py --regime hard` uses 300-400 features per view, 80% of the variance in
three private factors per view and 0.5-2% in the two shared factors, to test
whether a correlation-based method does better when the shared signal is faint
next to private variance. Its results are written with `--out output/results_hard.json`, and `plot.py` draws both regimes.

What it shows, over eight datasets per setting:

- **Complete data.** The CCA family recovers the shared factors at 0.64-0.73,
  0.89-0.90 and 0.96 for 0.5%, 1% and 2% shared variance; MOFA+ with the usual
  factor selection gets 0.21-0.32, 0.42-0.70 and 0.80-0.97. That gap is mostly
  the selection rule: `oracle_selection.py` shows that MOFA+ *does* find the signal
  (the true factors are 0.85, 0.94 and 0.98 explained by its factors, against
  0.74, 0.90 and 0.96 for `IncompleteGCCA`'s two components) but cannot be read off
  its factors when the shared signal is smaller than the private. What the CCA
  methods offer is the answer directly, ranked, with no selection step.
- **30% of views missing at random.** `IncompleteGCCA` leads MOFA+ (best of five
  random starts) by 0.16 at 1% shared and 0.04 at 2%, with a spread of 0.3 across
  datasets; the two tie at 0.5%. Masked `CCAEY` and the zero-filled and
  complete-case GCCA are far behind it (0.36-0.46 against 0.69 at 1%).
- **50% missing, or missingness tied to the biology.** MOFA+ is ahead or level.
- **Imputation.** `predict` of the CCA-family models is calibrated: it uses the
  latent of the other views out of fold with a cross-validated ridge, and `impute`
  rescales the scores of the samples the model was fitted to. Before, it
  made about seven times the error of predicting the mean at 1% shared variance
  (R squared of -10 to -70); now `IncompleteGCCA` and masked `CCAEY` never fall far
  below the mean (skill about 0 where there is little signal) and reach 0.3-0.7
  where there is, in the base regime. MOFA+ imputes better wherever the signal is
  real (skill 0.3-0.5 in the hard regime at 1-2% shared, against 0 to 0.21), but
  it can also fall far below the mean at heavy missingness (skill -0.5 to -2.5 at
  60% missing in the base regime), where the CCA family does not.
- **Time.** `IncompleteGCCA` takes 0.6 s; MOFA+ takes 2 s, or 10 s from five starts.

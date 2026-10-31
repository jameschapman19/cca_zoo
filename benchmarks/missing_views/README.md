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

## Hard regime: faint shared signal

`python run.py --regime hard` uses 300-400 features per view, 80% of the variance in
three private factors per view and 0.5-2% in the two shared factors, to test
whether a correlation-based method does better when the shared signal is faint
next to private variance. Results are in `output/results_hard.json`.

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
- **Imputation.** The CCA-family fill-in of a missing view is far worse than
  predicting zero when the signal is faint (R squared of -10 to -70 for
  `IncompleteGCCA`). `predict` needs shrinking before it is usable here.
- **Time.** `IncompleteGCCA` takes 0.6 s; MOFA+ takes 2 s, or 10 s from five starts.

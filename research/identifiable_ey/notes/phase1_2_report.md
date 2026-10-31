# Phase 1–2 report: gate failed

## What was run
`scripts/phase2_gate.py`: k=5 shared sources, p=10 per view (5 private Gaussian dims), n in {500, 1000, 2000, 5000}, 10 seeds (100–109), three configs:
mixed kurtosis (+2/−1) with ρ gap 0.05, all super-Gaussian (+2) with gap 0.05, mixed kurtosis with gap 0.01. Metric: Amari distance to the true sources, averaged over views.
Figures: `figures/phase2_amari.png` (mean and min–max), `figures/phase2_paired.png` (per-seed difference to CCA→FastICA).

Methods: CCA; CCA→FastICA (logcosh, on the shared average); CCA→MultiViewICA; CCA→CF rotation (rotation-only stage on top of CCA with the CF non-Gaussianity term); joint EY+NG at μ=1 and μ=10 (full batch, L-BFGS, best of 4 rotated CCA starts by final objective).

## What it shows
The gate fails clearly. In all 12 (config, n) cells the joint objective's worst seed is not better than CCA→FastICA's best seed, and means are equal or slightly worse (e.g. mixed, gap 0.05, n=500: FastICA 0.082±0.029, joint 0.096). From n≥2000 all three (FastICA, CF rotation, joint) agree to three decimals. The CF rotation stage alone matches the joint objective, so the joint term adds nothing over the rotation. In the pilot (seeds 0–2), μ=100 moved the subspace angle from ~4.5° to 5–14°, so a large μ hurts subspace recovery without improving Amari.

Reading: CCA's subspace error dominates the Amari distance in this regime, and the NG term does not repair it. Once the subspace is fixed, CF-based and FastICA-based rotation estimators perform alike.

CCA→MultiViewICA is the one competitor that differs: it is best of all at all-super-Gaussian (n=500: 0.083 vs FastICA 0.101; n=5000: 0.020 vs 0.022), but poor on mixed-sign kurtosis (0.16–0.29), because its logcosh model assumes super-Gaussian sources.

## Established vs suggestive
Established (for this generator): no gain from the joint objective over CCA→FastICA at k=5, p=10, ρ gaps ≤0.05, n 500–5000, equal or small μ.
Suggestive only: that the gain is absent for larger k, larger p, smaller n, or sources with higher kurtosis; not tested.

## Caveats
- Full-batch only. The minibatch/unbiasedness claim is tested at the estimator level (`tests/test_loss.py`: CF U-statistic and two-batch EY penalty unbiased, single-batch penalty biased, gradcheck, scale invariance), not yet in training.
- ShICA is not installed: the PyPI `multiviewica` 0.0.1 has no ShICA, its SRM reduction is broken against the installed fastsrm, and installing ShICA from GitHub was blocked. MultiViewICA is therefore run after CCA reduction, not with its own reduction.
- μ was piloted on seeds 0–2 only (2 values); FastICA used logcosh only, no oracle choice of contrast.
- The research code lives in `research/identifiable_ey/` inside the cca_zoo branch, because that is the only pushable repo here. It has its own `pyproject.toml` and can be moved out.

## Recommended next step
Stop, per the gate. Two cheap options: (1) ship the rotation step as a cca_zoo feature (CCA→CF rotation matches FastICA, so the case is weak; its only advantage is a minibatch-capable, FastICA-free path), or (2) one more probe before closing: n=100–300 and k=10–20 with larger p, where subspace noise is worst, to see whether any regime favours joint. I would do (2) only if you want the paper to survive; I would not expect it to.

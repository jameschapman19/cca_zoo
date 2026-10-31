# Batch size x embedding dimension sweep (synthetic, linear)

`scripts/sweep_batch_dim.py`; figure `figures/sweep_batch_dim.png`. p=256, 8 shared Gaussian sources (rho 0.5–0.2), d in {16, 64, 128}, batch in {8..128}, 4 seeds (400–403), 3000 Adam steps, linear encoders. Arms: EY single-batch; EY split-batch (batch halved into a, b, penalty tr(V_a V_b), unbiased, same samples); LeJEPA (repo SIGReg); LeJEPA with U-statistic SIGReg. U-stat SIGReg unbiased under the Gaussian null and the repo's V-stat biased (`tests/test_loss.py`).

## Results
- **Bias is real and shows up in scale.** Single-batch EY shrinks the embedding variance to 0.1–0.45 of the EY optimum at batch 8 and still 0.75 at batch 128 with d=128; split-batch EY sits at 1.0–1.15 everywhere. The effect grows with d/N, as predicted.
- **It does not show up in subspace or probe accuracy.** Subspace capture and probe R² (standardised features, so scale-blind) are equal within seed spread at batch ≥ 64, and the unbiased split arm is **worse** at small batch (capture 0.15 vs 0.44 at d=16, batch 8; 0.71 vs 0.78 at batch 32): halving the batch leaves covariances of 4–16 samples, and the variance of the unbiased gradient outweighs the bias.
- **LeJEPA ≈ single-batch EY** on capture and probe in every cell (differences ≤ 0.02 at batch ≥ 32). U-stat vs V-stat SIGReg: also near-identical (capture 0.516 vs 0.500 worst case).
- Kill criterion ("unbiased is no better at any d/N") is met for capture and probe at equal samples.

## Not tested (the obvious missing control)
Two independent full batches of N (2N samples per step), as `DCCAEY(independent_views=...)` does, so unbiasedness is not paid for with halved batches. Also: non-Gaussian sources, real data, other lrs, SIGReg lambda.

## Control: two independent full batches of N (2N samples) vs single batch of N
From the same grid (split arm at batch 2N vs single at N). Capture at d=16: +0.04 (N=32: 0.817 vs 0.776), +0.04 (N=16), ~0 (N=8); at d≥64 equal or worse (d=64, N=8: 0.507 vs 0.709). Probe R² differences ≤ 0.02 except small N at d≥64 (worse). Unbiasedness does not buy accuracy even with twice the samples.

## Heavy-tailed sources (`scripts/sweep_nongaussian.py`, figure `figures/sweep_nongaussian.png`)
Same toy with excess kurtosis 0 vs 6, d in {16, 64}, batch in {32, 128}, 4 seeds. Hypothesis: SIGReg's Gaussian-marginal pull degrades subspace recovery when sources are non-Gaussian.
- **Not supported.** EY and LeJEPA (λ=0.05) lose at most 0.02 capture going from kurtosis 0 to 6 (e.g. d=16, batch 32: LeJEPA 0.798 → 0.778, EY 0.776 → 0.758), within seed spread (sd 0.03–0.05), and equally for both. Probe R² is unchanged.
- **λ=0.5 is destructive, with or without heavy tails.** Capture falls to 0.11–0.35 in both the Gaussian and kurtosis-6 cases, with no dependence on kurtosis. So this is SIGReg overpowering the alignment term, not a non-Gaussianity effect. It also shows the toy's sensitivity to λ, which the paper reports as stable; whether that is a toy artefact (the loss scales here are unnormalised) is untested.
- Correction to an earlier argument: uniform-slice SIGReg is rotation-invariant, so it cannot push sources to mix with each other; it could only bias which directions enter the span, and the data show no sign of that.

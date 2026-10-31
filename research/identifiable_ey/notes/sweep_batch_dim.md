# Batch size x embedding dimension sweep (synthetic, linear)

`scripts/sweep_batch_dim.py`; figure `figures/sweep_batch_dim.png`. p=256, 8 shared Gaussian sources (rho 0.5–0.2), d in {16, 64, 128}, batch in {8..128}, 4 seeds (400–403), 3000 Adam steps, linear encoders. Arms: EY single-batch; EY split-batch (batch halved into a, b, penalty tr(V_a V_b), unbiased, same samples); LeJEPA (repo SIGReg); LeJEPA with U-statistic SIGReg. U-stat SIGReg unbiased under the Gaussian null and the repo's V-stat biased (`tests/test_loss.py`).

## Results
- **Bias is real and shows up in scale.** Single-batch EY shrinks the embedding variance to 0.1–0.45 of the EY optimum at batch 8 and still 0.75 at batch 128 with d=128; split-batch EY sits at 1.0–1.15 everywhere. The effect grows with d/N, as predicted.
- **It does not show up in subspace or probe accuracy.** Subspace capture and probe R² (standardised features, so scale-blind) are equal within seed spread at batch ≥ 64, and the unbiased split arm is **worse** at small batch (capture 0.15 vs 0.44 at d=16, batch 8; 0.71 vs 0.78 at batch 32): halving the batch leaves covariances of 4–16 samples, and the variance of the unbiased gradient outweighs the bias.
- **LeJEPA ≈ single-batch EY** on capture and probe in every cell (differences ≤ 0.02 at batch ≥ 32). U-stat vs V-stat SIGReg: also near-identical (capture 0.516 vs 0.500 worst case).
- Kill criterion ("unbiased is no better at any d/N") is met for capture and probe at equal samples.

## Not tested (the obvious missing control)
Two independent full batches of N (2N samples per step), as `DCCAEY(independent_views=...)` does, so unbiasedness is not paid for with halved batches. Also: non-Gaussian sources, real data, other lrs, SIGReg lambda.

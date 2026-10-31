# Idea 2: noise-aware contrast, low-ρ probe (negative)

`scripts/probe_lowrho.py`: k=5, p=30, ρ in 0.5–0.3 and 0.35–0.15, mixed and super-Gaussian sources, n in {500, 2000, 5000}, 8 seeds (200–207). Figure: `figures/probe_lowrho.png`.

Contrast: the CF distance with each component's Gaussian view noise divided out (integrand times exp(noise_fraction · t²), noise fraction from fixed moments as 1 − w₁'S₁₂w₂ / var(z̄)). It is unbiased for the noise-free signal's distance (tested) and gradchecked.

## Result
- Denoising does not help: CF rotation, plain vs denoised, are within seed noise of each other in almost every cell (e.g. 0.213 vs 0.164 at n=2000 mixed ρ 0.5–0.3 is the largest gain; 0.358 vs 0.352 at n=500). Neither beats CCA→FastICA, which is better in most cells (n=2000 mixed: FastICA 0.123, CF 0.213, denoised 0.164).
- The joint objective never leaves CCA's subspace: the subspace angle of joint EY+NG equals CCA's to 0.1° in every cell (e.g. 22.7° vs 22.7°; 59.7° vs 59.7° at ρ 0.35–0.15), so the premise "non-Gaussianity pulls in directions CCA ranks too low" was **not tested**. At μ=1 the EY gradient dominates; at larger μ the pilot subspace got worse.
- Subspace error is the bottleneck: CCA's angle is 14–85°, and FastICA on the oracle subspace is far better (0.034–0.36 vs 0.036–0.42).
- The CF contrast with Gauss–Hermite normal weighting is a less efficient rotation estimator than FastICA's logcosh at small n. Bandwidth and node count were not tuned.

## Established vs suggestive
Established: no benefit from noise-aware weighting in this model and regime. Not established: whether non-Gaussianity can improve the subspace at all (needs a design that can move it, e.g. fitting more components than k).

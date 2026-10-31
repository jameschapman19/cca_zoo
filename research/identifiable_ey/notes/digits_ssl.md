# Nonlinear encoders on real data: sklearn digits (`scripts/digits_ssl.py`, figure `figures/digits_ssl.png`)

Shared MLP encoder (64-256-256-32), two augmented views (±1 pixel shift, noise, 20% pixel dropout), 1000 train / 797 held-out digits, Adam lr 1e-3, batch 64 and 256, 3 seeds (500–502). Probes on un-augmented embeddings, scored at 250/1000/3000/8000 steps. Untrained-encoder reference: linear 0.907, kNN 0.928.

## A mistake caught on the way
The first run trained `150 epochs x (1000 // batch)` = 450 steps at batch 256, and showed EY ~0.96 vs LeJEPA ~0.90 (no better than untrained). A logged run showed LeJEPA starting from tiny embeddings (std 0.02, SIGReg 105), dipping (kNN 0.82, effective rank 3.5 at step 100) and still improving at step 1600 (0.945). That comparison measured convergence speed; λ in {0.01..0.8} made no difference. The script now scores several step counts.

## Results (mean of 3 seeds, linear / kNN at 8000 steps)
- Batch 256: EY single 0.964 / 0.961, EY split 0.965 / 0.955, LeJEPA (V) 0.961 / 0.951, LeJEPA (U) 0.962 / 0.951.
- Batch 64: EY single 0.969 / 0.959, EY split 0.969 / 0.954, LeJEPA (V) 0.944 / 0.933, LeJEPA (U) 0.944 / 0.934.
- EY is within 1 point of its final accuracy by 250 steps; LeJEPA needs 3000–8000 and, at batch 64, has not caught up by 8000.
- Unbiased (split) EY and U-stat SIGReg match their biased versions at every checkpoint (differences ≤ 0.006).

## What this supports, and what it does not
Supported (tiny dataset, 3 seeds): EY is much faster to converge than LeJEPA here, and ends equal or ahead, by ~0–2.5 points depending on batch and probe. Unbiasedness is irrelevant to accuracy here too.
Not supported: any claim about images at scale, or that the ordering holds with tuned learning rates, a projector head, or SIGReg-friendly initialisation (the slow start might be a scale-initialisation effect). The kNN theorem LeJEPA relies on is not shown to matter or not.

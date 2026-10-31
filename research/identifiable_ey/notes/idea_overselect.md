# Overselection: can non-Gaussianity choose the subspace? (negative)

`scripts/probe_overselect.py`: fit the top m=3k=15 CCA components, whiten their shared average, then select k=5 directions by non-Gaussianity (FastICA in m dims keeping the k with the largest logcosh negentropy proxy, or CF projection pursuit on the orthonormal m×k frame, 8 restarts, one at the top-k CCA frame). Same four generators as the low-ρ probe, p=30, n in {1000, 2000, 5000}, 8 seeds (300–307). Figure: `figures/probe_overselect.png`.

## Result
Worse than plain CCA top-k→FastICA in all 12 (config, n) cells for both selection rules, usually by a factor of 1.5–3 (e.g. ρ 0.5–0.3 mixed, n=2000: top-k 0.107, FastICA-select 0.241, CF pursuit 0.369; the oracle subspace gets 0.074). The only near-tie is ρ 0.5–0.3 super-Gaussian at n=5000 (0.037 vs 0.040). The largest principal angle of the selected subspace is 77–87° in nearly every cell, so at least one selected direction is a spurious one.

This is the direct test of "non-Gaussianity carries subspace information that CCA's second-order ranking discards". In this model it does not, at these n and ρ.

## Likely reason (not tested)
A shared component's non-Gaussianity is attenuated by its signal fraction (excess kurtosis scales with the square of it: about 0.2 of the source's at ρ=0.3 in the shared average), while the sample non-Gaussianity of the many null directions is of order 1/√n and is maximised over a 10-dimensional family of candidates. Selecting by it picks spurious directions before weakly non-Gaussian true ones.

## Established vs suggestive
Established, for this generator: the two selection rules tried do not improve on top-k CCA; the joint objective's lack of effect on the subspace (previous probe) is not an optimiser artefact of the design.
Suggestive: that the information is too weak; a selection with a proper significance threshold per component was not tried.

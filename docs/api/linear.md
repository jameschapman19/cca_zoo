# cca_zoo.linear

Linear CCA methods. All classes are `sklearn.base.BaseEstimator` subclasses.
Sparse/regularised iterative methods live in [`cca_zoo.sparse`](sparse.md);
mini-batch methods live in [`cca_zoo.stochastic`](stochastic.md).

---

## Base class

::: cca_zoo._base.BaseModel
    options:
      show_source: false
      members:
        - fit
        - transform
        - inverse_transform
        - predict
        - fit_transform
        - score
        - feature_importances_

---

## Two-view exact methods

::: cca_zoo.linear.CCA

---

::: cca_zoo.linear.rCCA

---

::: cca_zoo.linear.PLS

---

## Multiview methods

::: cca_zoo.linear.MCCA

---

::: cca_zoo.linear.GCCA

---

::: cca_zoo.linear.TCCA

---

## Confound-adjusted / structured methods

::: cca_zoo.linear.PartialCCA

---

::: cca_zoo.linear.GRCCA

---

## Reduced-rank regression methods

::: cca_zoo.linear.CCAR3

---

::: cca_zoo.linear.ECCA

---

## Sparse-precision covariance methods

::: cca_zoo.linear.GraphicalLassoCCA

---

## Robust methods

::: cca_zoo.linear.RANSACCCA

---

::: cca_zoo.linear.TrimmedCCA

---

::: cca_zoo.linear.ProjectionPursuitCCA

---

## EY-loss methods

::: cca_zoo.linear.PLSEY

---

::: cca_zoo.linear.CCAEY

---

::: cca_zoo.linear.HuberCCA

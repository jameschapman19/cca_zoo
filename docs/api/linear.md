# cca_zoo.linear

Linear CCA methods. All classes are `sklearn.base.BaseEstimator` subclasses.

---

## Base class

::: cca_zoo._base.BaseModel
    options:
      show_source: false
      members:
        - fit
        - transform
        - fit_transform
        - score
        - pairwise_correlations
        - average_pairwise_correlations
        - weights
        - get_factor_loadings

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

## Gradient-descent methods

::: cca_zoo.linear.PLSEY

---

::: cca_zoo.linear.CCAEY

---

::: cca_zoo.linear.MCCAEY

---

## Sparse / iterative methods

::: cca_zoo.linear.PLSALS

---

::: cca_zoo.linear.SCCAPMD

---

::: cca_zoo.linear.SCCAADMM

---

::: cca_zoo.linear.SCCAIPLS

---

::: cca_zoo.linear.SCCASpan

---

::: cca_zoo.linear.ElasticCCA

---

::: cca_zoo.linear.ParkhomenkoCCA

---

::: cca_zoo.linear.SAR

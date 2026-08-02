# cca_zoo.deep

Deep CCA variants. Requires `pip install cca-zoo[deep]`.

---

## Base class

::: cca_zoo.deep.BaseDeep
    options:
      members:
        - forward
        - transform
        - score
        - training_step
        - validation_step
        - configure_optimizers

---

## Data

::: cca_zoo.deep.MultiviewDataset

---

## Models

::: cca_zoo.deep.DCCA

---

::: cca_zoo.deep.DCCA_EY

---

::: cca_zoo.deep.DCCA_NOI

---

::: cca_zoo.deep.DCCA_SDL

---

::: cca_zoo.deep.DCCAE

---

::: cca_zoo.deep.DVCCA

---

::: cca_zoo.deep.DTCCA

---

::: cca_zoo.deep.DMCCA

---

::: cca_zoo.deep.DGCCA

---

::: cca_zoo.deep.SplitAE

---

::: cca_zoo.deep.BarlowTwins

---

::: cca_zoo.deep.VICReg

---

## Objectives

::: cca_zoo.deep.objectives
    options:
      members:
        - CCALoss
        - MCCALoss
        - GCCALoss
        - TCCALoss

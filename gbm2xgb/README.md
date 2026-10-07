# gbm2xgb

Convert LightGBM and CatBoost models to XGBoost. Predictions match the source model to float32 precision.

```python
import gbm2xgb, xgboost as xgb

bst = gbm2xgb.convert(model)      # any fitted LightGBM or CatBoost model
bst.predict(xgb.DMatrix(X))
bst.save_model("model.ubj")       # XGBoost binary (UBJSON); .json also works
```

CLI: `gbm2xgb lightgbm model.txt model.ubj` or `gbm2xgb catboost model.cbm model.ubj`.

## Estimators restricted to convertible parameters

`gbm2xgb.lightgbm.{LGBMRegressor, LGBMClassifier}` and `gbm2xgb.catboost.{CatBoostRegressor, CatBoostClassifier}`
subclass the originals, raise `ValueError` at construction, `set_params` and `fit` for anything the conversion
can't represent, and add `.to_xgboost()`, which returns an `xgboost.XGBRegressor` / `XGBClassifier`
(`fit` still returns `self`). Classifier labels must be `0..k-1`, since `XGBClassifier` predicts indices.

```python
from gbm2xgb.lightgbm import LGBMRegressor

reg = LGBMRegressor(objective="huber").fit(X, y)
reg.to_xgboost()
LGBMRegressor(zero_as_missing=True)   # ValueError
```

## LightGBM

Supported: numerical and categorical splits, NaN handling, `regression`/`regression_l1`/`huber`/`fair`/`quantile`/`mape`
(identity link), `binary` (any `sigmoid`), `multiclass`, `poisson`, `gamma`, `tweedie`, gbdt/dart/goss.
Models with early stopping convert up to the best iteration, as `predict` uses.

Unsupported (`NotImplementedError` from `convert`, `ValueError` from the estimators): `zero_as_missing`,
linear trees, `boosting_type="rf"`, `multiclassova`, ranking, `cross_entropy`, `reg_sqrt`, custom objectives,
and `init_score` at fit time.

## CatBoost

Supported: numerical features, symmetric trees (each depth-d tree becomes a full binary tree with 2^d leaves),
`nan_mode` Min/Max, `boost_from_average` bias, `RMSE`/`MAE`/`Quantile`/`Huber`/`Expectile`/`LogCosh`/`MAPE`
(identity link), `Logloss`/`CrossEntropy`, `MultiClass`.

Unsupported: categorical, text and embedding features, `grow_policy` other than `SymmetricTree`, other losses.

## Notes

- Converted `regression_l1`/`huber`/etc. models load as `reg:squarederror`; inference is identical, but
  continuing training in XGBoost would use the wrong loss.
- LightGBM categorical features must be passed to XGBoost as integer codes with `feature_types` containing `"c"`
  (check `bst.feature_types`).
- LightGBM's placeholder `Column_i` feature names are dropped; real names are kept.
- LightGBM's own model file is text, not binary; the binary format is XGBoost's `.ubj`.

# lgb2xgb

Convert LightGBM models to XGBoost models. Predictions match LightGBM to float32 precision.

```python
import lgb2xgb, xgboost as xgb

bst = lgb2xgb.convert(lgb_booster)          # Booster, LGBMRegressor/Classifier, or model file path
bst.predict(xgb.DMatrix(X))

lgb2xgb.convert_file("model.txt", "model.ubj")   # XGBoost binary (UBJSON); .json also works
```

CLI: `lgb2xgb model.txt model.ubj`

LightGBM's own model file is text, not binary; the binary format here is XGBoost's `.ubj`.

## Supported

Numerical and categorical splits, NaN handling, `regression`/`regression_l1`/`huber`/`fair`/`quantile`/`mape`
(identity link), `binary` (any `sigmoid`), `multiclass`, `poisson`, `gamma`, `tweedie`, gbdt/dart/goss.

Everything else raises `NotImplementedError`: `zero_as_missing`, linear trees, random forest mode,
`multiclassova`, ranking, `cross_entropy`, `reg_sqrt`.

## Notes

- Converted `regression_l1`/`huber`/etc. models load as `reg:squarederror`; inference is identical, but
  continuing training in XGBoost would use the wrong loss.
- Categorical features must be passed to XGBoost as integer codes with `feature_types` containing `"c"`
  (or a pandas `category` dtype whose codes match LightGBM's). Check `bst.feature_types`.
- LightGBM's placeholder `Column_i` feature names are dropped; real names are kept.

import xgboost as xgb


def convert(model) -> xgb.Booster:
    """Convert a fitted LightGBM or CatBoost model to an ``xgboost.Booster``."""
    framework = type(model).__module__.split(".")[0]
    if framework == "lightgbm":
        from gbm2xgb import lightgbm as backend
    elif framework == "catboost":
        from gbm2xgb import catboost as backend
    else:
        raise TypeError(f"cannot convert {type(model).__name__}; expected a LightGBM or CatBoost model")
    return backend.convert(model)


__all__ = ["convert"]

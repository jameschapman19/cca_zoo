import argparse
import importlib

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LightGBM or CatBoost model file to an XGBoost model file."
    )
    parser.add_argument("framework", choices=["lightgbm", "catboost"])
    parser.add_argument("src", help="model file: LightGBM text format or CatBoost .cbm")
    parser.add_argument("dst", help="output path; .json or .ubj (binary)")
    args = parser.parse_args()
    if not args.dst.endswith((".json", ".ubj")):
        parser.error("dst must end in .json or .ubj")
    importlib.import_module(f"gbm2xgb.{args.framework}").convert(args.src).save_model(args.dst)

if __name__ == "__main__":
    main()

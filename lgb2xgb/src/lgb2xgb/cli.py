import argparse

from lgb2xgb import convert_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LightGBM model file to an XGBoost model file."
    )
    parser.add_argument("src", help="LightGBM model file (text format)")
    parser.add_argument("dst", help="output path; .json or .ubj (binary)")
    args = parser.parse_args()
    convert_file(args.src, args.dst)


if __name__ == "__main__":
    main()

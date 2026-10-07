#!/usr/bin/env python3
"""
Entry point: radio signal (RSSI) prediction with linear regression from scratch.

Pipeline (each step lives in its own module under src/):
    1. data_generation : generate (or load) link-budget data
    2. eda             : exploratory data analysis and plots
    3. features        : feature engineering and train/test split
    4. linear_regression : normal equation and gradient descent solvers
    5. metrics         : metrics written from scratch
    6. evaluation      : reports, physical interpretation, result plots

Usage:
    python run_linear_regression.py
    python run_linear_regression.py --n-samples 5000 --seed 7
    python run_linear_regression.py --no-generate --out data/my_measurements.csv
"""
import argparse
from pathlib import Path

from src.config import DEFAULT_DATA_PATH, FIG_DIR, TARGET
from src.data_generation import generate_data, load_data, save_data
from src.eda import explore
from src.evaluation import (
    evaluate_classification,
    evaluate_regression,
    plot_results,
    print_coefficients,
    print_physical_interpretation,
    sanity_check_sklearn,
    save_predictions,
)
from src.features import build_features, train_test_split
from src.linear_regression import LinearRegressionScratch


def parse_args():
    parser = argparse.ArgumentParser(description="Linear regression from scratch for RSSI prediction")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", default=str(DEFAULT_DATA_PATH),
                        help="CSV path: where generated data is saved, or the file to load with --no-generate")
    parser.add_argument("--no-generate", action="store_true",
                        help="load an existing CSV from --out instead of generating (and overwriting) data")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)

    # Step 1: data
    if args.no_generate:
        df = load_data(out_path)
        print(f"Loaded {len(df)} samples from {out_path}\n")
    else:
        df = generate_data(args.n_samples, args.seed)
        save_data(df, out_path)
        print(f"Generated {len(df)} samples and saved them to {out_path}\n")

    # Step 2: exploratory analysis
    explore(df)

    # Step 3: features and split
    env_levels = sorted(df["environment"].unique())
    X, names = build_features(df, env_levels)
    y = df[TARGET].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te, _, test_idx = train_test_split(X, y, args.test_size, args.seed)
    print(f"Train: {len(y_tr)} samples | Test: {len(y_te)} samples | Features: {len(names)}\n")

    # Step 4: train both solvers
    normal = LinearRegressionScratch(method="normal").fit(X_tr, y_tr)
    gd = LinearRegressionScratch(method="gd").fit(X_tr, y_tr)
    print_coefficients(normal, gd, names)
    print_physical_interpretation(normal, names, env_levels)

    # Steps 5-6: evaluate
    pred_rssi = evaluate_regression(normal, X_tr, y_tr, X_te, y_te, df["environment"])
    pred_cls, cm = evaluate_classification(y_te, pred_rssi)

    pred_path = save_predictions(df, test_idx, pred_rssi, pred_cls,
                                 out_path.parent / "linear_regression_predictions.csv")
    plot_results(y_te, pred_rssi, gd.loss_history_, cm)
    sanity_check_sklearn(normal, X_tr, y_tr)

    print(f"Predictions saved to {pred_path}")
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

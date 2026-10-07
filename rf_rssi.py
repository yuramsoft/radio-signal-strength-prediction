#!/usr/bin/env python3
"""
Random forest prediction and classification of measured_rssi_dbm.

Two approaches are trained and compared on the same train/test split:

  1. Regression -> binning : RandomForestRegressor predicts RSSI (dBm), then the
                             prediction is binned into signal classes.
  2. Direct classification : RandomForestClassifier predicts the class directly.

Classes (same thresholds as the rest of the project):
    no_signal : < -100 dBm
    poor      : -100 to -85 dBm
    fair      : -85 to -70 dBm
    good      : > -70 dBm

Usage:
    python rf_rssi.py
    python rf_rssi.py --data data/signal_data.csv --target measured_rssi_dbm
    python rf_rssi.py --features distance_m frequency_mhz tx_power_dbm
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

BINS = [-np.inf, -100, -85, -70, np.inf]
LABELS = ["no_signal", "poor", "fair", "good"]


def rssi_to_class(rssi):
    """Bin RSSI values (dBm) into the four signal classes."""
    return pd.cut(rssi, bins=BINS, labels=LABELS, right=False).astype(str)


def build_preprocessor(X):
    """Pass numeric columns through, one-hot encode categorical columns."""
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def feature_importances(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    return pd.Series(imp, index=names).sort_values(ascending=False)


def within_one_class(y_true, y_pred):
    """Share of predictions that are correct or off by one adjacent class."""
    idx = {label: i for i, label in enumerate(LABELS)}
    diff = np.abs(y_true.map(idx).to_numpy() - pd.Series(y_pred).map(idx).to_numpy())
    return float(np.mean(diff <= 1))


def main():
    parser = argparse.ArgumentParser(description="Random forest RSSI prediction and classification")
    parser.add_argument("--data", default=str(DATA_DIR / "signal_data.csv"), help="input CSV path")
    parser.add_argument("--target", default="measured_rssi_dbm", help="RSSI column name")
    parser.add_argument("--features", nargs="*", default=None,
                        help="feature columns (default: every column except the target)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---------------------------------------------------------------- data
    df = pd.read_csv(args.data).dropna(subset=[args.target])
    drop_cols = {args.target, "signal_class", "class"}
    feature_cols = args.features or [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y_rssi = df[args.target]
    y_class = rssi_to_class(y_rssi)

    print(f"Rows: {len(df)} | Features: {feature_cols}")
    print("Class distribution:\n", y_class.value_counts().reindex(LABELS).fillna(0).astype(int), "\n")

    # Stratify by class when every class has at least 2 samples
    stratify = y_class if y_class.value_counts().min() >= 2 else None
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_rssi, y_class,
        test_size=args.test_size, random_state=args.seed, stratify=stratify,
    )

    rf_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
    )

    # ------------------------------------- 1) regression, then binning
    reg = Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", RandomForestRegressor(**rf_params)),
    ])
    reg.fit(X_train, yr_train)
    pred_rssi = reg.predict(X_test)
    pred_class_from_reg = rssi_to_class(pd.Series(pred_rssi))

    print("=" * 60)
    print("1) RANDOM FOREST REGRESSION (RSSI in dBm)")
    print("=" * 60)
    print(f"MAE  : {mean_absolute_error(yr_test, pred_rssi):.2f} dB")
    print(f"RMSE : {np.sqrt(mean_squared_error(yr_test, pred_rssi)):.2f} dB")
    print(f"R^2  : {r2_score(yr_test, pred_rssi):.3f}")
    print("\nFeature importances (regressor):")
    print(feature_importances(reg).round(4).to_string())

    print("\nClasses obtained by binning the regression output:")
    print(classification_report(yc_test, pred_class_from_reg, labels=LABELS, zero_division=0))

    # ------------------------------------------ 2) direct classification
    clf = Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", RandomForestClassifier(class_weight="balanced", **rf_params)),
    ])
    clf.fit(X_train, yc_train)
    pred_class = clf.predict(X_test)

    print("=" * 60)
    print("2) RANDOM FOREST CLASSIFICATION (direct)")
    print("=" * 60)
    print(classification_report(yc_test, pred_class, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(yc_test, pred_class, labels=LABELS),
                       index=LABELS, columns=LABELS))
    print("\nFeature importances (classifier):")
    print(feature_importances(clf).round(4).to_string())

    # ---------------------------------------------------------- comparison
    print("\n" + "=" * 60)
    print("COMPARISON ON THE TEST SET")
    print("=" * 60)
    summary = pd.DataFrame({
        "accuracy": [
            accuracy_score(yc_test, pred_class_from_reg),
            accuracy_score(yc_test, pred_class),
        ],
        "macro_f1": [
            f1_score(yc_test, pred_class_from_reg, average="macro", labels=LABELS, zero_division=0),
            f1_score(yc_test, pred_class, average="macro", labels=LABELS, zero_division=0),
        ],
        "within_one_class": [
            within_one_class(yc_test, pred_class_from_reg),
            within_one_class(yc_test, pred_class),
        ],
    }, index=["regression + binning", "direct classifier"])
    print(summary.round(3))

    # --------------------------------------------------------------- save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, MODEL_DIR / "rf_regressor.joblib")
    joblib.dump(clf, MODEL_DIR / "rf_classifier.joblib")

    out = X_test.copy()
    out["true_rssi_dbm"] = yr_test.to_numpy()
    out["pred_rssi_dbm"] = pred_rssi
    out["true_class"] = yc_test.to_numpy()
    out["pred_class_from_regression"] = pred_class_from_reg.to_numpy()
    out["pred_class_direct"] = pred_class
    out.to_csv(DATA_DIR / "rf_predictions.csv", index=False)

    print(f"\nModels saved to      : {MODEL_DIR}")
    print(f"Predictions saved to : {DATA_DIR / 'rf_predictions.csv'}")


if __name__ == "__main__":
    main()

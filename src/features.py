"""
Step 3: feature engineering and train/test split.

Features are chosen so that the link budget is linear in them:
  log_distance            -> coefficient = -10 * n  (path-loss exponent)
  log_frequency           -> coefficient ~ -20
  tx_power, tx/rx gain    -> coefficient ~ +1
  env dummies             -> shift in intercept per environment (first level is baseline)
  log_distance x env      -> change in path-loss exponent per environment
"""
import numpy as np


def build_features(df, env_levels):
    """Return the design matrix X (NumPy array) and the list of feature names."""
    log_d = np.log10(df["distance_m"].to_numpy(dtype=float))
    cols = {
        "log_distance": log_d,
        "log_frequency": np.log10(df["frequency_mhz"].to_numpy(dtype=float)),
        "tx_power_dbm": df["tx_power_dbm"].to_numpy(dtype=float),
        "tx_gain_dbi": df["tx_gain_dbi"].to_numpy(dtype=float),
        "rx_gain_dbi": df["rx_gain_dbi"].to_numpy(dtype=float),
    }
    for lvl in env_levels[1:]:
        cols[f"env_{lvl}"] = (df["environment"].to_numpy() == lvl).astype(float)
    for lvl in env_levels[1:]:
        cols[f"log_distance_x_env_{lvl}"] = log_d * cols[f"env_{lvl}"]

    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    return X, names


def train_test_split(X, y, test_size=0.2, seed=42):
    """Random split using NumPy only. Returns the splits and the index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(round(len(y) * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx

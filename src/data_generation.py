"""
Step 1: data generation and loading.

The target is produced by a link budget with log-distance path loss, shadowing
and Rayleigh fading:

    RSSI = Pt + Gt + Gr - L_other - PL(d) - X_sigma + F
    PL(d) = 20*log10(d0) + 20*log10(f_MHz) - 27.56 + 10*n*log10(d/d0)
    X_sigma ~ Normal(0, sigma^2)                 (shadowing, dB)
    F = 10*log10(Exponential(1))                 (Rayleigh small-scale fading, dB)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import D0_M, ENVIRONMENTS, LABELS, OTHER_LOSS_DB, TARGET, THRESHOLDS


def rssi_to_class(rssi):
    """Bin RSSI values (dBm) into class names using the project thresholds."""
    idx = np.digitize(np.asarray(rssi, dtype=float), THRESHOLDS)
    return np.array(LABELS)[idx]


def generate_data(n_samples, seed):
    """Simulate link measurements and return them as a DataFrame."""
    rng = np.random.default_rng(seed)

    env = rng.choice(list(ENVIRONMENTS), n_samples, p=[0.30, 0.35, 0.35])
    n_exp = np.array([ENVIRONMENTS[e]["n"] for e in env])
    sigma = np.array([ENVIRONMENTS[e]["sigma"] for e in env])

    distance = 10 ** rng.uniform(np.log10(10), np.log10(1500), n_samples)  # m
    freq = rng.choice([433, 868, 915, 2400], n_samples)                   # MHz
    tx_power = rng.choice([10, 14, 17, 20], n_samples)                    # dBm
    tx_gain = rng.choice([0, 2, 3], n_samples)                            # dBi
    rx_gain = rng.choice([0, 2], n_samples)                               # dBi

    pl_d0 = 20 * np.log10(D0_M) + 20 * np.log10(freq) - 27.56
    path_loss = pl_d0 + 10 * n_exp * np.log10(distance / D0_M)
    shadowing = rng.normal(0.0, sigma)
    fading = 10 * np.log10(rng.exponential(1.0, n_samples))

    rssi = tx_power + tx_gain + rx_gain - OTHER_LOSS_DB - path_loss - shadowing + fading

    df = pd.DataFrame({
        "distance_m": distance.round(2),
        "frequency_mhz": freq,
        "tx_power_dbm": tx_power,
        "tx_gain_dbi": tx_gain,
        "rx_gain_dbi": rx_gain,
        "environment": env,
        TARGET: rssi.round(2),
    })
    df["signal_class"] = rssi_to_class(df[TARGET])
    return df


def save_data(df, path):
    """Write the dataset to CSV, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_data(path):
    """Load an existing CSV and (re)compute the signal_class column."""
    df = pd.read_csv(path)
    df["signal_class"] = rssi_to_class(df[TARGET])
    return df

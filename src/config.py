"""Shared paths and constants for the RSSI project."""
from pathlib import Path

# ---- paths (relative to the project root, which is the parent of src/) ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
DEFAULT_DATA_PATH = DATA_DIR / "signal_data.csv"

# ---- target and classes ----
TARGET = "measured_rssi_dbm"
THRESHOLDS = [-100, -85, -70]
LABELS = ["no_signal", "poor", "fair", "good"]
# no_signal: < -100 dBm | poor: -100 to -85 | fair: -85 to -70 | good: > -70

# ---- physical model used to generate the data ----
# Path-loss exponent (n) and shadowing standard deviation (sigma, dB) per environment
ENVIRONMENTS = {
    "rural": {"n": 2.3, "sigma": 4.0},
    "suburban": {"n": 3.0, "sigma": 6.0},
    "urban": {"n": 3.5, "sigma": 8.0},
}
D0_M = 1.0            # reference distance (m)
OTHER_LOSS_DB = 2.0   # cable and connector losses (dB)

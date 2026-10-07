# Radio Signal Strength (RSSI) Prediction and Classification
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Models](https://img.shields.io/badge/models-linear%20regression%20%7C%20random%20forest-green)

A machine-learning project for predicting and classifying **received radio signal strength** from wireless propagation parameters using Python. The project combines fundamental radio-propagation theory with machine-learning techniques. 

Received signal strength is a fundamental parameter in the design, analysis, and optimisation of wireless communication systems.

In an ideal free-space environment, received power can be estimated using the Friis transmission equation:

$$
P_{RX}=P_{TX}G_{TX}G_{RX}
\left(\frac{\lambda}{4\pi d}\right)^2
$$

where:

- $P_{TX}$ — transmitted power
- $G_{TX}$ — transmitter antenna gain
- $G_{RX}$ — receiver antenna gain
- $\lambda$ — wavelength
- $d$ — transmitter-receiver distance

Real wireless environments are more complex because of reflection, diffraction, scattering, multipath propagation, shadowing, terrain, buildings, and other environmental factors.


The project is designed as a practical bridge between **wireless communications engineering, mathematical modelling, Python programming, data analysis, and machine learning**.

---

Predict the received signal strength (`measured_rssi_dbm`) of a radio link and classify link quality as `no_signal`, `poor`, `fair` or `good`.
---
The project has two parts:

1. **Linear regression implemented from scratch** (NumPy only), on data generated from a physical link-budget model, with a full exploratory analysis.
2. **Random forest** regression and classification, used as a nonlinear comparison.

## Table of Contents

- [Problem Definition](#problem-definition)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [How the Data Is Generated](#how-the-data-is-generated)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Method: Linear Regression Followed by Binning](#method-linear-regression-followed-by-binning)
- [Method: Random Forest](#method-random-forest)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Using Your Own Measurements](#using-your-own-measurements)

## Problem Definition

The target variable `measured_rssi_dbm` is continuous. For classification tasks it is binned into four ordered classes:

| Class       | RSSI range (dBm) |
|-------------|------------------|
| `no_signal` | < -100           |
| `poor`      | -100 to -85      |
| `fair`      | -85 to -70       |
| `good`      | > -70            |

Two tasks are solved:

- **Prediction (regression):** estimate the RSSI in dBm from link parameters.
- **Classification:** assign the link to one of the four classes, either by binning the predicted RSSI or with a classifier that predicts the class directly.

## Project Structure

```
.
├── run_linear_regression.py       # entry point: runs the linear regression pipeline
├── rf_rssi.py                     # random forest regression and classification
├── requirements.txt
├── README.md
├── src/                           # one module per pipeline step
│   ├── __init__.py
│   ├── config.py                  # paths, class thresholds, environment parameters
│   ├── data_generation.py         # step 1: link-budget data generation and loading
│   ├── eda.py                     # step 2: exploratory data analysis and plots
│   ├── features.py                # step 3: feature engineering and train/test split
│   ├── linear_regression.py       # step 4: linear regression from scratch
│   ├── metrics.py                 # step 5: metrics written from scratch
│   └── evaluation.py              # step 6: reports, interpretation, result plots
├── data/                          # created on first run
│   ├── signal_data.csv
│   ├── linear_regression_predictions.csv
│   └── rf_predictions.csv
├── figures/
│   ├── eda.png
│   └── results.png
└── models/                        # created by rf_rssi.py
    ├── rf_regressor.joblib
    └── rf_classifier.joblib
```

### Pipeline modules

| Module                  | Responsibility                                                              |
|-------------------------|-----------------------------------------------------------------------------|
| `src/config.py`         | Paths, the four class thresholds and labels, per-environment path-loss parameters |
| `src/data_generation.py`| `generate_data`, `save_data`, `load_data`, `rssi_to_class`                  |
| `src/eda.py`            | `summarize` (printed statistics), `plot_eda` (figure), `explore` (both)     |
| `src/features.py`       | `build_features` (log-distance, log-frequency, dummies, interactions), `train_test_split` |
| `src/linear_regression.py` | `LinearRegressionScratch` with normal equation and gradient descent      |
| `src/metrics.py`        | MAE, RMSE, R², skewness, confusion matrix, precision/recall/F1, within-one-class |
| `src/evaluation.py`     | Coefficient and path-loss interpretation, regression and classification reports, plots |

`run_linear_regression.py` only wires these modules together, so each step can be imported and reused on its own, for example:

```python
from src.data_generation import generate_data
from src.features import build_features, train_test_split
from src.linear_regression import LinearRegressionScratch
```

## Getting Started
### 1. Clone the repository
```bash
git clone https://github.com/yuramsoft/radio-signal-strength-prediction.git
cd radio-signal-strength-prediction
```
### 2. Create a virtual environment
Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`. The linear regression model itself uses NumPy only. scikit-learn is used by the random forest script and for an optional sanity check.

## Quick Start

```bash
# 1. Generate data, run the exploratory analysis, train linear regression from scratch
python run_linear_regression.py

# 2. Train and compare random forest regression and classification
python rf_rssi.py
```

Useful options:

```bash
python run_linear_regression.py --n-samples 5000 --seed 7
python run_linear_regression.py --no-generate --out data/my_measurements.csv
python rf_rssi.py --data data/signal_data.csv --target measured_rssi_dbm
python rf_rssi.py --features distance_m frequency_mhz tx_power_dbm
```

All outputs go into `data/`, `figures/` and `models/` next to the scripts, regardless of the directory you run them from.

## Dataset

`data/signal_data.csv` is generated by `run_linear_regression.py` (via `src/data_generation.py`) (3,000 samples by default).

| Column              | Description                                         |
|---------------------|-----------------------------------------------------|
| `distance_m`        | Transmitter-receiver distance, 10 to 1,500 m (log-uniform) |
| `frequency_mhz`     | Carrier frequency: 433, 868, 915 or 2400            |
| `tx_power_dbm`      | Transmit power: 10, 14, 17 or 20                    |
| `tx_gain_dbi`       | Transmit antenna gain: 0, 2 or 3                    |
| `rx_gain_dbi`       | Receive antenna gain: 0 or 2                        |
| `environment`       | `rural`, `suburban` or `urban`                      |
| `measured_rssi_dbm` | Target: received signal strength                    |
| `signal_class`      | Class obtained by binning `measured_rssi_dbm`       |

Example class distribution with the default seed:

| Class       | Count | Share |
|-------------|-------|-------|
| `no_signal` | 656   | 21.9% |
| `poor`      | 543   | 18.1% |
| `fair`      | 660   | 22.0% |
| `good`      | 1,141 | 38.0% |

## How the Data Is Generated

The target comes from a **link budget** with **log-distance path loss**, **shadowing** and **fading**. All terms are in dB or dBm, so they add and subtract directly.

```
RSSI = Pt + Gt + Gr - L_other - PL(d) - X_sigma + F
```

| Symbol    | Meaning                                                    |
|-----------|------------------------------------------------------------|
| `Pt`      | Transmit power (dBm)                                       |
| `Gt`, `Gr`| Transmit and receive antenna gains (dBi)                   |
| `L_other` | Fixed cable and connector losses (2 dB)                    |
| `PL(d)`   | Mean path loss at distance `d`                             |
| `X_sigma` | Shadowing, large-scale random variation                    |
| `F`       | Small-scale fading from multipath                          |

### Path loss

```
PL(d) = PL(d0) + 10 * n * log10(d / d0)
PL(d0) = 20*log10(d0_m) + 20*log10(f_MHz) - 27.56       (free-space loss at d0 = 1 m)
```

`n` is the path-loss exponent, which depends on the environment:

| Environment | Path-loss exponent `n` | Shadowing sigma (dB) |
|-------------|------------------------|----------------------|
| rural       | 2.3                    | 4                    |
| suburban    | 3.0                    | 6                    |
| urban       | 3.5                    | 8                    |

### Shadowing

Shadowing models slow variation caused by obstacles. It is log-normal, meaning Gaussian when expressed in dB:

```
X_sigma ~ Normal(0, sigma^2)
```

### Fading

Fast multipath fading is modeled as **Rayleigh** fading. The power gain is exponentially distributed, so in dB it is skewed toward occasional deep fades:

```
F = 10 * log10(Exponential(1))      (standard deviation about 5.6 dB)
```

## Exploratory Data Analysis

Running `run_linear_regression.py` prints summary statistics, missing values, class balance, RSSI per environment and correlations, and saves `figures/eda.png`.

![Exploratory data analysis](figures/eda.png)

Key observations:

- **RSSI is not linear in distance, but it is linear in `log10(distance)`.** The correlation with RSSI is -0.79 for `log10(distance)` and -0.69 for raw distance. This is the signature of the log-distance path loss model.
- **Environment matters.** Mean RSSI is about -65 dBm in rural, -81 dBm in suburban and -91 dBm in urban areas, and the spread grows with the shadowing level.
- **The scatter around the trend is large** (several dB) because of shadowing and fading, which sets a limit on prediction accuracy.
- **Higher frequency lowers RSSI** and higher transmit power raises it, as the link budget predicts. There are no missing values.

## Method: Linear Regression Followed by Binning

Linear regression predicts a number, not a class. The pipeline is therefore: **predict RSSI, then apply the class thresholds.**

```
features -> linear regression -> predicted RSSI (dBm) -> bin -> class
```

This keeps the natural ordering of the classes (`no_signal` < `poor` < `fair` < `good`), provides an estimate of the actual dBm value, and lets the class thresholds change without retraining.

### Why linear regression suits this problem

For fixed link parameters the link budget is **linear in the right features**:

```
RSSI = [Pt + Gt + Gr - L_other - PL(d0)] - 10*n*log10(d) - 20*log10(f) + noise
```

So the model is fitted on engineered features, not raw columns:

| Feature                         | Why                                                         | Expected coefficient |
|---------------------------------|-------------------------------------------------------------|----------------------|
| `log10(distance)`               | Path loss is linear in log-distance                         | `-10 * n`            |
| `log10(frequency)`              | Free-space loss grows with `20*log10(f)`                    | about -20            |
| `tx_power_dbm`                  | Enters the link budget one-to-one                           | about +1             |
| `tx_gain_dbi`, `rx_gain_dbi`    | Antenna gains add directly                                  | about +1             |
| environment dummies             | Shift the intercept per environment                         | -                    |
| `log10(distance)` x environment | Different path-loss exponent per environment                | -                    |

The path-loss exponent can be read directly from the model: `n = -(slope on log10(distance)) / 10`.

### Implementation from scratch

`LinearRegressionScratch` in `src/linear_regression.py` uses NumPy only and provides two solvers:

**1. Normal equation** (closed form):

```
theta = (X^T X)^-1 X^T y        (solved with np.linalg.solve; pseudo-inverse as fallback)
```

**2. Batch gradient descent** on the mean squared error:

```
loss  = mean( (X w + b - y)^2 )
w <- w - lr * (2/m) * X^T (X w + b - y)
b <- b - lr * (2/m) * sum(X w + b - y)
```

Features are standardized during gradient descent, and the weights are mapped back to the original units afterwards, so both solvers return directly comparable coefficients. The metrics (`src/metrics.py`: MAE, RMSE, R², confusion matrix, precision, recall, F1) and the train/test split (`src/features.py`) are also written from scratch.

### Fitted model

Coefficients from the default run (3,000 samples, 80/20 split, seed 42):

| Coefficient                    | Estimated | Expected |
|--------------------------------|-----------|----------|
| `log_frequency`                | -19.84    | about -20 |
| `tx_power_dbm`                 | 1.04      | about +1  |
| `tx_gain_dbi`                  | 0.78      | about +1  |
| `rx_gain_dbi`                  | 0.88      | about +1  |

Recovered path-loss exponents:

| Environment | Estimated `n` | True `n` |
|-------------|---------------|----------|
| rural       | 2.311         | 2.3      |
| suburban    | 3.036         | 3.0      |
| urban       | 3.437         | 3.5      |

The two solvers agree to about 1e-4, and the normal equation matches scikit-learn's `LinearRegression` to about 1e-11.

## Method: Random Forest

`rf_rssi.py` trains two models on the same split and compares them:

1. **`RandomForestRegressor`** predicts RSSI in dBm, and the prediction is binned into classes.
2. **`RandomForestClassifier`** predicts the class directly, with `class_weight="balanced"`.

Categorical columns are one-hot encoded inside a scikit-learn pipeline. Trees are unaffected by monotonic transforms, so raw `distance_m` and `frequency_mhz` can be used without taking logarithms. The script prints feature importances, a classification report, a confusion matrix and a comparison table, and saves both models to `models/` and the test predictions to `data/rf_predictions.csv`.

## Results

![Model results](figures/results.png)

Results on the default dataset (3,000 samples, 20% test set). The two scripts use different random splits, so small differences are within noise.

**Regression (RSSI in dBm):**

| Model                     | MAE (dB) | RMSE (dB) | R²    |
|---------------------------|----------|-----------|-------|
| Linear regression (scratch)| 6.80    | 8.62      | 0.873 |
| Random forest regressor   | 7.37     | 9.12      | 0.851 |

**Classification:**

| Model                       | Accuracy | Macro F1 | Within one class |
|-----------------------------|----------|----------|------------------|
| Linear regression + binning | 0.727    | 0.698    | 0.992            |
| Random forest + binning     | 0.722    | 0.681    | 0.990            |
| Random forest, direct       | 0.707    | 0.666    | 0.977            |

"Within one class" counts a prediction as acceptable if it is correct or lands in an adjacent class. It suits ordered classes, because confusing `fair` with `good` is far less serious than confusing `good` with `no_signal`.

Findings:

- **Linear regression performs best here**, because the data follows a model that is linear in log-distance. The model is correctly specified, so it also recovers the physical parameters.
- **There is an irreducible error floor.** Shadowing and Rayleigh fading are random, so no model can predict them. For this dataset the floor is roughly 8.4 dB RMSE, and linear regression is already close to it (8.6 dB).
- **Most classification errors are between neighboring classes** near the thresholds. A prediction of -84.5 dBm for a true value of -85.5 dBm is a 1 dB error but a different class, so class accuracy looks lower than the dB error suggests.
- The random forest is useful when the true relationship is nonlinear or the form is unknown, but it cannot extrapolate beyond the distances seen in training, while linear regression can.

## Limitations

- **Synthetic data.** Results reflect the generating model. Real measurements contain effects not modeled here, such as obstacles, terrain, interference and receiver sensitivity limits.
- **Single slope per environment.** The linear model assumes one path-loss exponent per environment. Mixed line-of-sight and non-line-of-sight conditions break this assumption.
- **Random noise.** Shadowing and fading cannot be predicted, which bounds the achievable accuracy.
- **Boundary sensitivity.** Class accuracy is limited by samples that fall close to the thresholds.

## Future Work

- Validate on real RSSI measurements.
- Add logistic regression and **ordinal regression** that respect class ordering.
- Add gradient boosting and neural network baselines.
- Use a dual-slope path-loss model with a breakpoint distance.
- Report prediction intervals, so samples near a class boundary can be flagged as uncertain.
- Add cross-validation and hyperparameter tuning for the random forest.

## Using Your Own Measurements

To use real data, put your CSV in `data/` and run the scripts on it. Without `--no-generate`, `run_linear_regression.py` generates new data and **overwrites** the file given by `--out`, so always pass the flag for your own files:

```bash
python run_linear_regression.py --no-generate --out data/my_measurements.csv
python rf_rssi.py --data data/my_measurements.csv
```

The scripts expect these columns:

```
distance_m, frequency_mhz, tx_power_dbm, tx_gain_dbi, rx_gain_dbi, environment, measured_rssi_dbm
```

`rf_rssi.py` uses every column except the target as a feature, or only the ones passed with `--features`. Adjust the feature list in `build_features()` in `src/features.py` if your dataset has different columns.
 
 ---

## Author

**Ibrahim Mustapha, PhD**

Department of Electrical & Electronic Engineering
University of Maiduguri
Communications Engineering / Wireless Communications
Python & Machine Learning

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This project is primarily an educational and research-oriented implementation. Predictions produced by the initial model should not be treated as a substitute for detailed radio-frequency planning, field measurements, or validated commercial propagation models.

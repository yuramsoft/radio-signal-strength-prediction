"""
Step 6: evaluation reports, physical interpretation and result plots.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ENVIRONMENTS, FIG_DIR, LABELS, THRESHOLDS
from .data_generation import rssi_to_class
from .metrics import (
    accuracy,
    classification_metrics,
    confusion_matrix,
    mae,
    r2,
    rmse,
    skewness,
    within_one_class,
)


def print_coefficients(ne_model, gd_model, names):
    """Compare the coefficients found by the two solvers."""
    print("=" * 70)
    print("MODEL COEFFICIENTS")
    print("=" * 70)
    table = pd.DataFrame({
        "normal_equation": np.r_[ne_model.intercept_, ne_model.coef_],
        "gradient_descent": np.r_[gd_model.intercept_, gd_model.coef_],
    }, index=["intercept"] + names)
    print(table.round(4).to_string())
    print(f"\nGradient descent stopped after {len(gd_model.loss_history_)} iterations")
    print(f"Max |difference| between solvers: {np.max(np.abs(ne_model.coef_ - gd_model.coef_)):.2e}\n")


def print_physical_interpretation(model, names, env_levels):
    """Relate the fitted coefficients to the link-budget parameters."""
    c = dict(zip(names, model.coef_))
    print("Expected from the link budget: log_frequency ~ -20, tx_power ~ +1, gains ~ +1")
    print(f"Estimated: log_frequency = {c['log_frequency']:.2f}, tx_power = {c['tx_power_dbm']:.2f}, "
          f"tx_gain = {c['tx_gain_dbi']:.2f}, rx_gain = {c['rx_gain_dbi']:.2f}\n")

    print("Path-loss exponent n = -(slope on log10(distance)) / 10")
    rows = []
    for k, lvl in enumerate(env_levels):
        slope = c["log_distance"] + (c[f"log_distance_x_env_{lvl}"] if k > 0 else 0.0)
        rows.append({"environment": lvl, "estimated_n": -slope / 10,
                     "true_n": ENVIRONMENTS[lvl]["n"]})
    print(pd.DataFrame(rows).set_index("environment").round(3).to_string(), "\n")


def evaluate_regression(model, X_tr, y_tr, X_te, y_te, environments):
    """Print regression metrics and residual diagnostics. Returns test predictions."""
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    print("=" * 70)
    print("REGRESSION PERFORMANCE (normal equation)")
    print("=" * 70)
    print(pd.DataFrame({
        "MAE (dB)": [mae(y_tr, pred_tr), mae(y_te, pred_te)],
        "RMSE (dB)": [rmse(y_tr, pred_tr), rmse(y_te, pred_te)],
        "R^2": [r2(y_tr, pred_tr), r2(y_te, pred_te)],
    }, index=["train", "test"]).round(3).to_string())

    resid = y_te - pred_te
    # Rayleigh fading in dB has variance of about 31 dB^2 (std about 5.6 dB)
    shadow_var = np.mean([ENVIRONMENTS[e]["sigma"] ** 2 for e in environments])
    floor = np.sqrt(shadow_var + 31.0)
    print(f"\nResidual mean = {resid.mean():.3f} dB, std = {resid.std():.2f} dB, "
          f"skewness = {skewness(resid):.2f}")
    print(f"Approximate error floor from shadowing + Rayleigh fading: ~{floor:.1f} dB RMSE")
    print("(Rayleigh fading in dB has a standard deviation of about 5.6 dB and is skewed toward deep fades.)\n")
    return pred_te


def evaluate_classification(y_te, pred_te):
    """Bin RSSI into classes and report classification metrics."""
    true_cls = rssi_to_class(y_te)
    pred_cls = rssi_to_class(pred_te)
    cm = confusion_matrix(true_cls, pred_cls)
    metrics = classification_metrics(cm)

    print("=" * 70)
    print("CLASSIFICATION BY BINNING THE PREDICTED RSSI")
    print("=" * 70)
    print(metrics.round(3).to_string())
    print(f"\nAccuracy = {accuracy(cm):.3f} | Macro F1 = {metrics['f1'].mean():.3f} | "
          f"Within one class = {within_one_class(true_cls, pred_cls):.3f}\n")
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string(), "\n")
    return pred_cls, cm


def save_predictions(df, test_idx, pred_rssi, pred_cls, path):
    """Save the test rows together with the model predictions."""
    out = df.iloc[test_idx].copy().reset_index(drop=True)
    out["pred_rssi_dbm"] = np.round(pred_rssi, 2)
    out["pred_class"] = pred_cls
    out.to_csv(path, index=False)
    return path


def plot_results(y_te, pred_te, loss_history, cm, path=None):
    """Four panels: predicted vs actual, residuals, gradient descent loss, confusion matrix."""
    path = path or FIG_DIR / "results.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    resid = y_te - pred_te

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))

    ax = axes[0]
    ax.scatter(y_te, pred_te, s=8, alpha=0.4)
    lims = [min(y_te.min(), pred_te.min()), max(y_te.max(), pred_te.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    for t in THRESHOLDS:
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(t, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Predicted vs actual RSSI (test)")
    ax.set_xlabel("Actual RSSI (dBm)")
    ax.set_ylabel("Predicted RSSI (dBm)")

    ax = axes[1]
    ax.hist(resid, bins=40, color="slateblue", edgecolor="white")
    ax.axvline(0, color="k", linewidth=1)
    ax.set_title("Residuals (actual - predicted)")
    ax.set_xlabel("Error (dB)")

    ax = axes[2]
    ax.plot(loss_history)
    ax.set_yscale("log")
    ax.set_title("Gradient descent: training MSE")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE (log scale)")

    ax = axes[3]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, rotation=30)
    ax.set_yticklabels(LABELS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def sanity_check_sklearn(model, X_tr, y_tr):
    """Optional: compare the from-scratch coefficients with scikit-learn."""
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("scikit-learn not installed: skipped the optional sanity check.")
        return None

    sk = LinearRegression().fit(X_tr, y_tr)
    diff = max(np.max(np.abs(sk.coef_ - model.coef_)), abs(sk.intercept_ - model.intercept_))
    print(f"Sanity check vs scikit-learn: max coefficient difference = {diff:.2e}")
    return diff

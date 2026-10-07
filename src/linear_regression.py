"""
Step 4: linear regression implemented from scratch (NumPy only).
"""
import numpy as np


class LinearRegressionScratch:
    """
    Ordinary least squares linear regression.

        y_hat = intercept + X @ coef

    method="normal": solves the normal equation (X^T X) theta = X^T y
    method="gd"    : minimizes the mean squared error with batch gradient descent
                     on standardized features, then maps the weights back to the
                     original feature scale so both methods give comparable coefficients.
    """

    def __init__(self, method="normal", lr=0.05, n_iter=30000, tol=1e-12):
        if method not in ("normal", "gd"):
            raise ValueError("method must be 'normal' or 'gd'")
        self.method = method
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    # ---- training -----------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.method == "normal":
            self._fit_normal(X, y)
        else:
            self._fit_gd(X, y)
        return self

    def _fit_normal(self, X, y):
        m = X.shape[0]
        Xb = np.hstack([np.ones((m, 1)), X])          # add intercept column
        A = Xb.T @ Xb                                  # X^T X
        b = Xb.T @ y                                   # X^T y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:                  # singular matrix: use pseudo-inverse
            theta = np.linalg.pinv(A) @ b
        self.intercept_, self.coef_ = theta[0], theta[1:]

    def _fit_gd(self, X, y):
        m, p = X.shape
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd

        w = np.zeros(p)
        b = y.mean()
        self.loss_history_ = []
        prev = np.inf
        for _ in range(self.n_iter):
            err = Xs @ w + b - y
            loss = np.mean(err ** 2)
            self.loss_history_.append(loss)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
            w -= self.lr * (2.0 / m) * (Xs.T @ err)
            b -= self.lr * (2.0 / m) * err.sum()

        # convert back to original feature units
        self.coef_ = w / sd
        self.intercept_ = b - np.sum(w * mu / sd)

    # ---- inference ----------------------------------------------------------
    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_

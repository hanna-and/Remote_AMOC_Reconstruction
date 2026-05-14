
"""
Bayesian Linear Regression with Automatic Relevance Determination (ARD).

This module provides a scikit-learn-compatible wrapper around a variational
Bayesian linear regression model. The key idea behind ARD is that each input
feature gets its own precision (inverse variance) hyperparameter. During
fitting, irrelevant features have their precision driven to infinity, which
shrinks the corresponding weight to zero. This gives you built-in feature
selection without needing a separate step.

Typical usage:

    model = BayesianLinearRegressionARD()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Inspect which features matter
    print(model.feature_relevances_)

The model internally normalises both X and y to zero mean and unit variance
before fitting. The public attributes (coef_, intercept_) are always reported
in the original (un-normalised) scale so you can interpret them directly.
"""

import sys
import warnings
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core numerical routines (unchanged from the original implementation)
# ---------------------------------------------------------------------------

def _meanvar(D):
    """Compute column-wise mean and variance of an array."""
    m = np.mean(D, axis=0)
    s = np.shape(D)
    if len(s) == 1:
        c = D.var()
        if c == 0:
            c = 1.0
    elif (s[0] == 1) + (s[1] == 1):
        c = D.var()
        if c == 0:
            c = 1.0
    else:
        c = np.diag(np.cov(D, rowvar=False)).copy()
        c[c == 0] = 1
    return m, c


def _logdet(a):
    """Log-determinant of a symmetric positive-definite matrix via Cholesky."""
    step1 = np.linalg.cholesky(a)
    step2 = np.diag(step1.T)
    return 2.0 * np.sum(np.log(step2), axis=0)


def _bayes_linear_fit_ard(X, y):
    """
    Variational Bayesian linear regression with ARD priors.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Normalised design matrix.
    y : array-like, shape (N, 1)
        Normalised target vector.

    Returns
    -------
    w        : posterior mean of weights, shape (D,)
    V        : posterior covariance of weights, shape (D, D)
    invV     : precision matrix (inverse of V)
    logdetV  : log-determinant of V
    an, bn   : posterior parameters of the noise precision Gamma distribution
    E_a      : expected precision for each weight (high = irrelevant feature)
    L        : variational lower bound (approximate log model evidence)
    """
    X = np.matrix(X)
    y = np.matrix(y)

    # Uninformative prior hyperparameters
    a0, b0 = 1e-2, 1e-4   # noise precision Gamma prior
    c0, d0 = 1e-2, 1e-4   # weight precision Gamma prior

    [N, D] = np.shape(X)
    X_corr = X.T * X
    Xy_corr = X.T * y
    an = a0 + N / 2.0
    gammaln_an = scipy.special.gammaln(an)
    cn = c0 + 0.5
    D_gammaln_cn = D * scipy.special.gammaln(cn)

    L_last = -sys.float_info.max
    max_iter = 500
    E_a = np.matrix(np.ones(D) * c0 / d0).T

    for iteration in range(max_iter):
        # Posterior covariance and mean of weights
        invV = np.matrix(np.diag(np.array(E_a)[:, 0])) + X_corr
        V = np.matrix(np.linalg.inv(invV))
        logdetV = -_logdet(invV)
        w = np.dot(V, Xy_corr)[:, 0]

        # Noise precision update
        sse = np.sum(np.power(X * w - y, 2), axis=0)
        if np.imag(sse) == 0:
            sse = np.real(sse)[0]
        else:
            warnings.warn("Complex SSE encountered – stopping early.")
            break
        bn = b0 + 0.5 * (sse + np.sum(
            (np.array(w)[:, 0] ** 2) * np.array(E_a)[:, 0], axis=0))
        E_t = an / bn

        # Weight precision update (ARD)
        dn = d0 + 0.5 * (E_t * (np.array(w)[:, 0] ** 2) + np.diag(V))
        E_a = np.matrix(cn / dn).T

        # Variational lower bound
        L = (-0.5 * (E_t * sse + np.sum(np.multiply(X, X * V)))
             + 0.5 * logdetV - b0 * E_t + gammaln_an
             - an * np.log(bn) + an + D_gammaln_cn
             - cn * np.sum(np.log(dn)))

        if L_last > L:
            warnings.warn("Variational bound decreased – possible numerical issues.")
            break
        if abs(L_last - L) < abs(1e-5 * L):
            break
        L_last = L

    if iteration == max_iter - 1:
        warnings.warn("ARD fit did not converge within %d iterations." % max_iter)

    # Add constant terms to the bound
    L = (L - 0.5 * (N * np.log(2 * np.pi) - D)
         - scipy.special.gammaln(a0) + a0 * np.log(b0)
         + D * (-scipy.special.gammaln(c0) + c0 * np.log(d0)))

    return w, V, invV, logdetV, an, bn, E_a, L


# ---------------------------------------------------------------------------
# Sklearn-style wrapper
# ---------------------------------------------------------------------------

class BayesianLinearRegressionARD:
    """
    Bayesian Linear Regression with Automatic Relevance Determination.

    This estimator follows the scikit-learn interface (fit / predict) and
    internally normalises the data before fitting.

    How it works (in plain English)
    -------------------------------
    Ordinary linear regression finds weights w that minimise squared error.
    Bayesian linear regression treats w as a random variable and computes a
    full posterior distribution over w, which tells you not just the best
    guess but also how uncertain you are about each weight.

    ARD adds one twist: every weight gets its own regularisation strength.
    The model learns these automatically. If a feature is irrelevant, its
    regularisation is cranked up, shrinking that weight towards zero. The
    result is a sparse-ish model that tells you which features matter.

    Attributes (available after calling fit)
    ----------------------------------------
    coef_ : np.ndarray, shape (n_features,)
        Regression coefficients in the **original scale** of X and y.
        Directly comparable to sklearn's LinearRegression.coef_.

    intercept_ : float
        Intercept in the **original scale**.

    coef_normalised_ : np.ndarray, shape (n_features,)
        Regression coefficients in the normalised (zero-mean, unit-variance)
        space. Useful for comparing relative feature importance when features
        have different units.

    feature_relevances_ : np.ndarray, shape (n_features,)
        Relevance score for each feature, defined as 1 / E[precision].
        Larger values indicate more relevant features. Features with very
        small relevance scores have been "switched off" by ARD.

    noise_variance_ : float
        Estimated noise variance (in normalised space).

    posterior_cov_ : np.ndarray, shape (n_features, n_features)
        Posterior covariance of the weights (normalised space).

    lower_bound_ : float
        Variational lower bound on the log marginal likelihood. Higher is
        better; can be used for model comparison.

    Examples
    --------
    >>> model = BayesianLinearRegressionARD()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print("Relevant features:", np.where(model.feature_relevances_ > 0.01)[0])
    """

    def __init__(self):
        # All state is set during fit().
        self._is_fitted = False

    # -- public API --------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input matrix.
        y : array-like, shape (n_samples,) or (n_samples, 1)
            Training target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(
                "X has %d samples but y has %d." % (n_samples, y.shape[0]))

        # ---- Normalise ----
        # We store the training statistics so we can:
        #   1. normalise new X at prediction time
        #   2. convert the fitted weights back to the original scale
        self._X_mean, self._X_var = _meanvar(X)
        self._y_mean, self._y_var = _meanvar(y)

        # Ensure arrays (not matrices) for clean arithmetic
        self._X_mean = np.asarray(self._X_mean).ravel()
        self._X_var = np.asarray(self._X_var).ravel()
        self._y_mean = float(np.asarray(self._y_mean).ravel()[0])
        self._y_var = float(np.asarray(self._y_var).ravel()[0])

        X_norm = (X - self._X_mean) / np.sqrt(self._X_var)
        y_norm = (y - self._y_mean) / np.sqrt(self._y_var)

        # ---- Fit in normalised space ----
        w, V, invV, logdetV, an, bn, E_a, L = _bayes_linear_fit_ard(
            X_norm, y_norm.reshape(-1, 1))

        # Store normalised-space results
        self.coef_normalised_ = np.asarray(w).ravel()
        self.posterior_cov_ = np.asarray(V)
        self.noise_variance_ = float(np.asarray(bn / an).ravel()[0])
        self.lower_bound_ = float(np.asarray(L).ravel()[0])

        # ARD relevance = 1 / expected_precision  (higher = more relevant)
        E_a_arr = np.asarray(E_a).ravel()
        self.feature_relevances_ = 1.0 / E_a_arr

        # ---- Convert weights back to original scale ----
        #
        # In normalised space the model is:
        #     y_norm = X_norm @ w_norm
        #
        # Substituting the normalisation definitions:
        #     (y - m_y)/s_y = sum_i  w_i * (X_i - m_x_i) / s_x_i
        #
        # Rearranging for y:
        #     y = sum_i  (s_y * w_i / s_x_i) * X_i
        #         - sum_i (s_y * w_i * m_x_i / s_x_i)
        #         + m_y
        #
        # So:
        #     coef_i    = s_y * w_i / s_x_i
        #     intercept = m_y - sum_i (coef_i * m_x_i)
        #
        sy = np.sqrt(self._y_var)
        sx = np.sqrt(self._X_var)
        self.coef_ = sy * self.coef_normalised_ / sx
        self.intercept_ = self._y_mean - np.dot(self.coef_, self._X_mean)

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict target values for new inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values in the original scale.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_

    def predict_with_uncertainty(self, X):
        """
        Predict target values together with predictive standard deviations.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted mean in the original scale.
        y_std  : np.ndarray, shape (n_samples,)
            Predictive standard deviation in the original scale.  This
            accounts for both weight uncertainty and estimated noise.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = X @ self.coef_ + self.intercept_

        # Uncertainty in normalised space, then scale back
        X_norm = (X - self._X_mean) / np.sqrt(self._X_var)
        var_norm = self.noise_variance_ + np.sum(
            (X_norm @ self.posterior_cov_) * X_norm, axis=1)
        y_std = np.sqrt(var_norm) * np.sqrt(self._y_var)

        return y_pred, y_std

    def score(self, X, y):
        """
        Return the R² score (coefficient of determination).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        r2 : float
        """
        self._check_fitted()
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot

    def coef_std(self):
        """
        Posterior standard deviation for each coefficient, in original scale.

        These are the Bayesian analogue of "standard errors" in frequentist
        regression.  A weight whose std is large relative to its magnitude
        is poorly determined by the data.

        The maths: in normalised space the posterior covariance is V, so
        std_norm_i = sqrt(V[i,i]).  Since coef_i = (s_y / s_x_i) * w_norm_i,
        the original-scale std is (s_y / s_x_i) * std_norm_i.

        Returns
        -------
        std : np.ndarray, shape (n_features,)
        """
        self._check_fitted()
        sy = np.sqrt(self._y_var)
        sx = np.sqrt(self._X_var)
        std_norm = np.sqrt(np.diag(self.posterior_cov_))
        return (sy / sx) * std_norm

    def coef_ci(self, level=0.95):
        """
        Credible intervals for each coefficient in original scale.

        Unlike frequentist confidence intervals, a Bayesian credible interval
        means: "given the data, there is a `level` probability that the true
        weight lies in this range."

        Parameters
        ----------
        level : float
            Probability mass (default 0.95 for a 95% credible interval).

        Returns
        -------
        lower : np.ndarray, shape (n_features,)
        upper : np.ndarray, shape (n_features,)
        """
        self._check_fitted()
        from scipy.stats import norm as normal_dist
        z = normal_dist.ppf(0.5 + level / 2.0)
        std = self.coef_std()
        return self.coef_ - z * std, self.coef_ + z * std

    def posterior_cov_original(self):
        """
        Full posterior covariance of the weights in the original scale.

        If V is the posterior covariance in normalised space and
        S = diag(s_y / s_x), then the original-scale covariance is S V S^T.

        Off-diagonal entries tell you how correlated two weight estimates
        are — useful for understanding whether two features are competing
        to explain the same variance in y.

        Returns
        -------
        cov : np.ndarray, shape (n_features, n_features)
        """
        self._check_fitted()
        sy = np.sqrt(self._y_var)
        sx = np.sqrt(self._X_var)
        scale = sy / sx  # shape (D,)
        # S V S^T  where S = diag(scale)
        return (scale[:, None] * self.posterior_cov_) * scale[None, :]

    def summary(self):
        """Print a human-readable summary of the fitted model."""
        self._check_fitted()
        n = len(self.coef_)
        std = self.coef_std()
        lo, hi = self.coef_ci(0.95)
        print("Bayesian Linear Regression with ARD")
        print("=" * 70)
        print(f"{'Feature':<10} {'Coef':>10} {'Std':>10} "
              f"{'95% CI':>22} {'Relevance':>12}")
        print("-" * 70)
        for i in range(n):
            ci_str = f"[{lo[i]:>8.4f}, {hi[i]:>8.4f}]"
            print(f"  x{i:<7} {self.coef_[i]:>10.4f} {std[i]:>10.4f} "
                  f"{ci_str:>22} {self.feature_relevances_[i]:>12.6f}")
        print("-" * 70)
        print(f"  Intercept:        {self.intercept_:.6f}")
        print(f"  Noise variance:   {self.noise_variance_:.6f}")
        print(f"  Lower bound:      {self.lower_bound_:.4f}")

    # -- internals ---------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

"""scikit-learn compatible estimators layered on the glmnet path.

Translating between glmnet's and scikit-learn's parameterizations is subtler
than the usual "swap the names" advice, and getting it wrong silently produces
an over-fit ridge component. The correct translation is derived here.

scikit-learn minimizes::

    (1/2n)*||y - Xb - b0||^2 + A*r*||b||_1 + (A/2)*(1-r)*||b||_2^2

where ``A = alpha`` (penalty strength) and ``r = l1_ratio`` (mixing).

glmnet appears to minimize the same thing with ``lambda <-> A`` and
``alpha <-> r``. It does not. For ``family="gaussian"`` glmnet first rescales
``y`` to unit variance (``ys = sd_w(y)``, the 1/n formula) and solves with
``lambda_tilde = lambda / ys``. The L1 penalty is homogeneous of degree 1 in
``b`` and rescales cleanly; the L2 penalty is degree 2 and does not. Unwinding
the substitution, what glmnet actually minimizes in the original units is::

    (1/2)*sum_i w_i (y_i - b0 - x_i'b)^2
        + lambda*alpha*||b||_1
        + (lambda*(1-alpha)/(2*ys))*||b||_2^2
                            ^^^^^^ note the stray 1/ys

This is the documented-but-easily-missed remark in ``?glmnet``: "for gaussian,
glmnet standardizes y to have unit variance before computing its lambda
sequence." It matters only when ``alpha < 1``; pure lasso is unaffected.

Equating coefficients gives the mapping used by :func:`_to_glmnet`::

    lambda = A*r + A*(1-r)*ys
    alpha  = A*r / lambda

Estimators here take **scikit-learn's** names and semantics; use
:func:`glmnet.glmnet` for R's. ``standardize`` defaults to False here (matching
sklearn) and True in R glmnet.
"""

from __future__ import annotations

import numpy as np

from ._path import glmnet

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "glmnet.sklearn requires scikit-learn: pip install 'glmnet-rs[sklearn]'"
    ) from exc

__all__ = ["ElasticNet", "Lasso"]


def _y_scale(y, weights, fit_intercept):
    """glmnet's `ys`: the weighted sd of y under the 1/n convention."""
    wn = weights / weights.sum()
    ybar = float(wn @ y) if fit_intercept else 0.0
    return float(np.sqrt(np.sum(wn * (y - ybar) ** 2)))


def _to_glmnet(alpha, l1_ratio, ys):
    """Map scikit-learn `(alpha, l1_ratio)` to glmnet `(lambda, alpha)`.

    See the module docstring: the `ys` factor on the L2 term is what makes this
    a genuine reparameterization rather than a rename.
    """
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio) * ys
    lam = l1 + l2
    if lam <= 0:
        raise ValueError("degenerate penalty: alpha > 0 required (and y must vary)")
    return lam, l1 / lam


class ElasticNet(RegressorMixin, BaseEstimator):
    """Elastic net with scikit-learn's parameterization, solved by glmnet's kernels.

    Parameters
    ----------
    alpha : float
        Penalty strength (glmnet calls this `lambda`).
    l1_ratio : float
        Mixing: 1.0 is lasso, 0.0 is ridge (glmnet calls this `alpha`).
    standardize : bool
        Scale columns to unit variance before fitting. False matches sklearn's
        `ElasticNet`; True matches R glmnet's default.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        *,
        fit_intercept: bool = True,
        standardize: bool = False,
        max_iter: int = 100_000,
        tol: float = 1e-7,
        positive: bool = False,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.alpha <= 0:
            raise ValueError("alpha must be > 0; use LinearRegression for alpha=0")

        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        ys = _y_scale(y, w, self.fit_intercept)
        lam, gl_alpha = _to_glmnet(self.alpha, self.l1_ratio, ys)

        path = glmnet(
            X,
            y,
            alpha=gl_alpha,
            lambda_=[lam],
            standardize=self.standardize,
            intercept=self.fit_intercept,
            thresh=self.tol,
            maxit=self.max_iter,
            lower_limits=0.0 if self.positive else None,
            weights=sample_weight,
        )
        self.path_ = path
        self.coef_ = path.beta[:, 0]
        self.intercept_ = float(path.a0[0])
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = path.npasses
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


class Lasso(ElasticNet):
    """Lasso: `ElasticNet` with `l1_ratio=1`."""

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        standardize: bool = False,
        max_iter: int = 100_000,
        tol: float = 1e-7,
        positive: bool = False,
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=1.0,
            fit_intercept=fit_intercept,
            standardize=standardize,
            max_iter=max_iter,
            tol=tol,
            positive=positive,
        )

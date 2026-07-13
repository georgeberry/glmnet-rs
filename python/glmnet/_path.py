"""The lambda path: glmnet's actual primitive.

`glmnet()` does not fit one model, it fits a whole regularization path and
returns coefficients at every lambda. Exposing that path directly (rather than
hiding it behind a single-lambda estimator) is what makes warm starts, `coef(s=)`
interpolation, and honest cross-validation possible.

Naming follows R glmnet: `alpha` is the elastic-net mixing parameter and
`lambda` is the penalty strength. This is the *opposite* of scikit-learn, where
`alpha` is the penalty strength. See `glmnet.sklearn` for the translation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core

__all__ = ["GlmnetPath", "glmnet", "lambda_interp"]


def lambda_interp(lambda_, s):
    """Locate `s` within the fitted `lambda_` grid.

    Returns `(left, right, frac)` such that a quantity at `s` is
    `frac * value[left] + (1 - frac) * value[right]`.

    Mirrors R's `lambda.interp`: interpolation is linear on the lambda scale
    (not log-lambda), after rescaling the grid to [0, 1]. Values of `s` outside
    the fitted range are clamped to its endpoints rather than extrapolated.
    """
    lambda_ = np.asarray(lambda_, dtype=float)
    s = np.atleast_1d(np.asarray(s, dtype=float))

    if lambda_.size == 1:
        n = s.size
        return np.zeros(n, int), np.zeros(n, int), np.ones(n)

    k = lambda_.size
    span = lambda_[0] - lambda_[k - 1]
    sfrac = (lambda_[0] - s) / span
    lam = (lambda_[0] - lambda_) / span

    sfrac = np.clip(sfrac, lam.min(), lam.max())

    # R uses approx(lam, seq_along(lam), sfrac): a linear index lookup.
    coord = np.interp(sfrac, lam, np.arange(k, dtype=float))
    left = np.floor(coord).astype(int)
    right = np.ceil(coord).astype(int)

    denom = lam[left] - lam[right]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (sfrac - lam[right]) / denom
    frac = np.where(left == right, 1.0, frac)
    frac = np.where(np.abs(denom) < np.finfo(float).eps, 1.0, frac)
    return left, right, frac


@dataclass
class GlmnetPath:
    """A fitted elastic-net regularization path."""

    lambda_: np.ndarray  # (lmu,) descending
    a0: np.ndarray  # (lmu,) intercepts
    beta: np.ndarray  # (p, lmu)
    dev_ratio: np.ndarray  # (lmu,) fraction of null deviance explained
    nulldev: float
    npasses: int
    alpha: float
    warning: tuple | None = None

    @property
    def lmu(self) -> int:
        """Number of lambdas actually fit.

        Usually smaller than `nlambda`: glmnet halts the path once the deviance
        stops improving (`fdev`) or saturates (`devmax`).
        """
        return self.lambda_.size

    @property
    def df(self) -> np.ndarray:
        """Number of nonzero coefficients at each lambda."""
        return np.count_nonzero(self.beta, axis=0)

    def coef(self, s=None):
        """Coefficients at lambda values `s`, interpolating off-grid values.

        Returns `(p + 1, len(s))` with the intercept in row 0. With `s=None`,
        returns the full fitted path.
        """
        if s is None:
            return np.vstack([self.a0, self.beta])
        left, right, frac = lambda_interp(self.lambda_, s)
        full = np.vstack([self.a0, self.beta])
        return full[:, left] * frac + full[:, right] * (1.0 - frac)

    def predict(self, X, s=None):
        """Linear predictor. Returns `(n, len(s))`, or `(n, lmu)` when `s is None`."""
        X = np.ascontiguousarray(X, dtype=float)
        c = self.coef(s)
        return X @ c[1:] + c[0]

    def __repr__(self) -> str:
        w = f", warning={self.warning[0]}" if self.warning else ""
        return (
            f"GlmnetPath(alpha={self.alpha}, lmu={self.lmu}, "
            f"lambda=[{self.lambda_[0]:.4g} .. {self.lambda_[-1]:.4g}], "
            f"dev_ratio<={self.dev_ratio[-1]:.4g}{w})"
        )


def glmnet(
    x,
    y,
    *,
    alpha: float = 1.0,
    nlambda: int = 100,
    lambda_min_ratio: float | None = None,
    lambda_=None,
    standardize: bool = True,
    intercept: bool = True,
    thresh: float = 1e-7,
    maxit: int = 100_000,
    dfmax: int | None = None,
    pmax: int | None = None,
    penalty_factor=None,
    lower_limits=None,
    upper_limits=None,
    weights=None,
    exclude=None,
) -> GlmnetPath:
    """Fit the Gaussian elastic-net path. Mirrors R's `glmnet(family="gaussian")`.

    `alpha=1` is the lasso, `alpha=0` is ridge. `lambda_` overrides the
    automatically-generated grid.
    """
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float).ravel()

    def _vec(v, name):
        if v is None:
            return None
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size == 1:
            arr = np.repeat(arr, x.shape[1])
        if arr.size != x.shape[1]:
            raise ValueError(f"{name} must have length 1 or {x.shape[1]}")
        return [float(t) for t in arr]

    res = _core.elnet_gaussian(
        x,
        y,
        alpha=float(alpha),
        nlambda=int(nlambda),
        lambda_min_ratio=lambda_min_ratio,
        user_lambda=None if lambda_ is None else [float(t) for t in np.atleast_1d(lambda_)],
        standardize=bool(standardize),
        intercept=bool(intercept),
        thresh=float(thresh),
        maxit=int(maxit),
        dfmax=dfmax,
        pmax=pmax,
        penalty_factor=_vec(penalty_factor, "penalty_factor"),
        lower_limits=_vec(lower_limits, "lower_limits"),
        upper_limits=_vec(upper_limits, "upper_limits"),
        weights=None if weights is None else [float(t) for t in np.atleast_1d(weights)],
        exclude=None if exclude is None else [int(t) for t in np.atleast_1d(exclude)],
    )

    return GlmnetPath(
        lambda_=res["lambda"],
        a0=res["a0"],
        beta=res["beta"],
        dev_ratio=res["dev_ratio"],
        nulldev=res["nulldev"],
        npasses=res["npasses"],
        alpha=float(alpha),
        warning=res["warning"],
    )

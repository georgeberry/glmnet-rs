"""The lambda path: glmnet's actual primitive.

`glmnet()` does not fit one model, it fits a whole regularization path and
returns coefficients at every lambda. Exposing that path directly (rather than
hiding it behind a single-lambda estimator) is what makes warm starts, `coef(s=)`
interpolation, and honest cross-validation possible.

Naming follows R glmnet: `alpha` is the elastic-net mixing parameter and
`lambda` is the penalty strength. This is the *opposite* of scikit-learn, where
`alpha` is the penalty strength. See `glmnetrs.sklearn` for the translation.
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
    family: str = "gaussian"
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

    def predict(self, X, s=None, type="link"):
        """Predictions. Returns `(n, len(s))`, or `(n, lmu)` when `s is None`.

        `type="link"` returns the linear predictor (all families).
        `type="response"` applies the inverse link: the class-1 probability for
        binomial, the mean count `exp(eta)` for Poisson (identity for gaussian).
        `type="class"` (binomial only) returns the 0/1 label at a 0.5 threshold.
        """
        X = np.ascontiguousarray(X, dtype=float)
        c = self.coef(s)
        eta = X @ c[1:] + c[0]
        if type == "link":
            return eta
        if type == "response":
            if self.family == "binomial":
                return 1.0 / (1.0 + np.exp(-eta))
            if self.family == "poisson":
                return np.exp(eta)
            return eta  # gaussian: identity link
        if type == "class":
            if self.family != "binomial":
                raise ValueError("type='class' is only defined for family='binomial'")
            return (eta > 0).astype(float)
        raise ValueError(f"unknown predict type {type!r}")

    def summary(self) -> str:
        """A per-lambda `Df / %Dev / Lambda` table, as R's ``print.glmnet``.

        ``Df`` is the number of nonzero coefficients and ``%Dev`` is the percent
        of null deviance explained. Returns the formatted string (also what
        ``print(path)`` shows).
        """
        rows = [f"{'':>4}  {'Df':>4}  {'%Dev':>6}  {'Lambda':>10}"]
        df = self.df
        for i in range(self.lmu):
            rows.append(
                f"{i + 1:>4}  {df[i]:>4}  {100 * self.dev_ratio[i]:>6.2f}  {self.lambda_[i]:>10.4g}"
            )
        return "\n".join(rows)

    def plot(self, xvar="norm", label=False, ax=None):
        """Plot the coefficient paths (matplotlib), as R's ``plot.glmnet``.

        `xvar` is ``"norm"`` (L1 norm, default), ``"lambda"`` (``-log(lambda)``)
        or ``"dev"`` (fraction of deviance explained). Returns the ``Axes``.
        """
        from .plot import plot_coef

        return plot_coef(self, xvar=xvar, label=label, ax=ax)

    def to_frame(self):
        """The summary as a pandas ``DataFrame`` (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(
            {
                "Df": self.df,
                "pct_dev": 100 * self.dev_ratio,
                "lambda": self.lambda_,
            },
            index=np.arange(1, self.lmu + 1),
        )

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        w = f", warning={self.warning[0]}" if self.warning else ""
        return (
            f"GlmnetPath(family={self.family!r}, alpha={self.alpha}, lmu={self.lmu}, "
            f"lambda=[{self.lambda_[0]:.4g} .. {self.lambda_[-1]:.4g}], "
            f"dev_ratio<={self.dev_ratio[-1]:.4g}{w})"
        )


def glmnet(
    x,
    y,
    *,
    family: str = "gaussian",
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
    """Fit an elastic-net path. Mirrors R's `glmnet`.

    `family` is `"gaussian"` (least squares), `"binomial"` (two-class logistic),
    or `"poisson"` (log-link counts). For binomial, `y` must be 0/1 (or observed
    proportions); for poisson, `y` must be non-negative. `alpha=1` is the lasso,
    `alpha=0` is ridge. `lambda_` overrides the automatically-generated grid.
    """
    if family not in ("gaussian", "binomial", "poisson"):
        raise ValueError(
            f"family {family!r} not supported (yet); use 'gaussian', 'binomial' or 'poisson'"
        )

    # Sparse input is detected by duck typing so scipy stays an optional dependency.
    is_sparse = hasattr(x, "tocsc") and hasattr(x, "shape")
    if is_sparse:
        if family != "gaussian":
            raise ValueError("sparse X is currently supported only for family='gaussian'")
        nvars = x.shape[1]
    else:
        x = np.ascontiguousarray(x, dtype=float)
        nvars = x.shape[1]

    y = np.ascontiguousarray(y, dtype=float).ravel()

    if family == "binomial":
        uniq = np.unique(y)
        if not np.all((y >= 0) & (y <= 1)):
            raise ValueError("binomial y must be 0/1 or in [0, 1]")
        if uniq.size < 2:
            raise ValueError("binomial y has only one class")
    elif family == "poisson":
        if np.any(y < 0):
            raise ValueError("poisson y must be non-negative")

    def _vec(v, name):
        if v is None:
            return None
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size == 1:
            arr = np.repeat(arr, nvars)
        if arr.size != nvars:
            raise ValueError(f"{name} must have length 1 or {nvars}")
        return [float(t) for t in arr]

    common = dict(
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

    if is_sparse:
        xc = x.tocsc()
        xc.sum_duplicates()  # our solver sums stored entries; collapse any dupes
        n, p = xc.shape
        res = _core.elnet_sparse(
            int(n),
            int(p),
            [int(v) for v in xc.indptr],
            [int(v) for v in xc.indices],
            np.ascontiguousarray(xc.data, dtype=float),
            y,
            **common,
        )
        return GlmnetPath(
            lambda_=res["lambda"],
            a0=res["a0"],
            beta=res["beta"],
            dev_ratio=res["dev_ratio"],
            nulldev=res["nulldev"],
            npasses=res["npasses"],
            alpha=float(alpha),
            family=family,
            warning=res["warning"],
        )

    res = _core.elnet_path(
        x,
        y,
        family=family,
        **common,
    )

    return GlmnetPath(
        lambda_=res["lambda"],
        a0=res["a0"],
        beta=res["beta"],
        dev_ratio=res["dev_ratio"],
        nulldev=res["nulldev"],
        npasses=res["npasses"],
        alpha=float(alpha),
        family=family,
        warning=res["warning"],
    )

"""Cross-validation over the lambda path, matching R's ``cv.glmnet``.

The lambda sequence is fixed to the *full-data* fit, and every fold is refit on
that same grid, so all folds evaluate the same lambdas. Per-observation losses
are aggregated exactly as R does (``cvcompute`` -> ``cvstats`` -> ``getOptcv``):

* the loss is averaged within each fold (weighted), giving a ``nfolds x nlambda``
  matrix ``outmat``;
* ``cvm`` is the fold-weight-weighted mean of ``outmat`` across folds;
* ``cvsd`` is the between-fold standard error,
  ``sqrt( wmean((outmat - cvm)^2) / (nfolds - 1) )``;
* ``lambda.min`` minimizes ``cvm`` (largest lambda achieving the min), and
  ``lambda.1se`` is the largest lambda whose ``cvm`` is within one ``cvsd`` of it.

The loss formulas per family are transliterated from ``cv.elnet`` / ``cv.lognet``
/ ``cv.fishnet``. ``deviance``/``mse``/``mae``/``class`` reproduce R exactly;
``auc`` uses a standard weighted estimator that is close to but not bit-identical
with R's ``survival::concordance``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._path import GlmnetPath, glmnet

__all__ = ["CVGlmnet", "cv_glmnet"]

# Allowed measures and the default per family, from glmnet's cvtype().
_MEASURES = {
    "gaussian": (["mse", "deviance", "mae"], "mse"),
    "binomial": (["deviance", "class", "auc", "mse", "mae"], "deviance"),
    "poisson": (["deviance", "mse", "mae"], "deviance"),
}
# Measures where a larger value is better (so lambda.min maximizes cvm).
_MAXIMIZE = {"auc"}


def _weighted_auc(y01, prob, w):
    """Weighted AUC = P(score(pos) > score(neg)) with 0.5 for ties.

    Standard rank estimator; not bit-matched to survival::concordance.
    """
    pos = y01 > 0
    neg = ~pos
    wp, sp = w[pos], prob[pos]
    wn, sn = w[neg], prob[neg]
    if wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    num = 0.0
    for s_i, w_i in zip(sp, wp):
        num += w_i * (np.sum(wn[sn < s_i]) + 0.5 * np.sum(wn[sn == s_i]))
    return num / (wp.sum() * wn.sum())


def _cvraw_per_obs(family, measure, y, link, weights):
    """Per-observation loss `(n, nlambda)` for the non-AUC measures.

    Returns `(cvraw, eff_weights)`; `eff_weights` may differ from `weights`
    (R scales binomial weights by the row sums, a no-op for 0/1 y).
    """
    if family == "gaussian":
        pred = link
        if measure == "mae":
            return np.abs(y[:, None] - pred), weights
        return (y[:, None] - pred) ** 2, weights  # mse == deviance

    if family == "poisson":
        if measure == "mse":
            return (y[:, None] - np.exp(link)) ** 2, weights
        if measure == "mae":
            return np.abs(y[:, None] - np.exp(link)), weights
        # deviance: 2 * ((y log y - y) - (y*eta - exp eta))
        devy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0)) - y, 0.0)
        deveta = y[:, None] * link - np.exp(link)
        return 2.0 * (devy[:, None] - deveta), weights

    # binomial: p = sigmoid(eta), y in {0,1}
    p = 1.0 / (1.0 + np.exp(-link))
    if measure == "mse":
        return 2.0 * (y[:, None] - p) ** 2, weights
    if measure == "mae":
        return 2.0 * np.abs(y[:, None] - p), weights
    if measure == "class":
        return y[:, None] * (p <= 0.5) + (1.0 - y[:, None]) * (p > 0.5), weights
    # deviance
    pmin_, pmax_ = 1e-5, 1 - 1e-5
    pc = np.clip(p, pmin_, pmax_)
    lp = (1.0 - y[:, None]) * np.log(1.0 - pc) + y[:, None] * np.log(pc)
    return -2.0 * lp, weights


@dataclass
class CVGlmnet:
    """Result of :func:`cv_glmnet`. Mirrors R's ``cv.glmnet`` object."""

    lambda_: np.ndarray
    cvm: np.ndarray
    cvsd: np.ndarray
    cvup: np.ndarray
    cvlo: np.ndarray
    nzero: np.ndarray
    lambda_min: float
    lambda_1se: float
    index_min: int
    index_1se: int
    glmnet_fit: GlmnetPath
    family: str
    type_measure: str
    foldid: np.ndarray

    def _resolve_s(self, s):
        if s == "lambda.min":
            return self.lambda_min
        if s == "lambda.1se":
            return self.lambda_1se
        return s  # numeric lambda passed through

    def coef(self, s="lambda.1se"):
        """Coefficients at ``lambda.min``/``lambda.1se`` (or a numeric lambda)."""
        return self.glmnet_fit.coef(s=self._resolve_s(s))

    def predict(self, X, s="lambda.1se", type="link"):
        """Predictions at ``lambda.min``/``lambda.1se`` (or a numeric lambda)."""
        return self.glmnet_fit.predict(X, s=self._resolve_s(s), type=type)

    _MEASURE_NAMES = {
        "mse": "Mean-Squared Error",
        "mae": "Mean Absolute Error",
        "deviance": "Deviance",
        "class": "Misclassification Error",
        "auc": "AUC",
    }

    def summary(self) -> str:
        """The ``min``/``1se`` selection table, as R's ``print.cv.glmnet``.

        Columns: selected lambda, its path index, the CV measure, its standard
        error, and the number of nonzero coefficients.
        """
        name = self._MEASURE_NAMES.get(self.type_measure, self.type_measure)
        head = f"Measure: {name}\n"
        cols = f"{'':>3}  {'Lambda':>9}  {'Index':>5}  {'Measure':>9}  {'SE':>7}  {'Nonzero':>7}"
        rows = [head + cols]
        for label, idx, lam in (
            ("min", self.index_min, self.lambda_min),
            ("1se", self.index_1se, self.lambda_1se),
        ):
            rows.append(
                f"{label:>3}  {lam:>9.4g}  {idx + 1:>5}  {self.cvm[idx]:>9.4g}  "
                f"{self.cvsd[idx]:>7.3g}  {self.nzero[idx]:>7}"
            )
        return "\n".join(rows)

    def plot(self, ax=None):
        """Plot the CV curve with error bars and min/1se lines (matplotlib), as
        R's ``plot.cv.glmnet``. Returns the ``Axes``."""
        from .plot import plot_cv

        return plot_cv(self, ax=ax)

    def to_frame(self):
        """Per-lambda CV curve as a pandas ``DataFrame`` (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(
            {
                "lambda": self.lambda_,
                "cvm": self.cvm,
                "cvsd": self.cvsd,
                "cvup": self.cvup,
                "cvlo": self.cvlo,
                "nzero": self.nzero,
            },
            index=np.arange(1, self.lambda_.size + 1),
        )

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return (
            f"CVGlmnet(family={self.family!r}, measure={self.type_measure!r}, "
            f"lambda.min={self.lambda_min:.4g} (cvm={self.cvm[self.index_min]:.4g}), "
            f"lambda.1se={self.lambda_1se:.4g})"
        )


def _make_folds(n, nfolds, foldid, seed):
    if foldid is not None:
        foldid = np.asarray(foldid, dtype=int).ravel()
        if foldid.size != n:
            raise ValueError(f"foldid must have length {n}")
        return foldid, int(foldid.max()) + 1
    rng = np.random.default_rng(seed)
    # R: sample(rep(1:nfolds, length=n)); 0-based here.
    base = np.tile(np.arange(nfolds), n // nfolds + 1)[:n]
    rng.shuffle(base)
    return base, nfolds


def cv_glmnet(
    x,
    y,
    *,
    family: str = "gaussian",
    type_measure: str = "default",
    nfolds: int = 10,
    foldid=None,
    seed=None,
    weights=None,
    **glmnet_kwargs,
) -> CVGlmnet:
    """Cross-validate the elastic-net path. See module docstring for the method.

    Extra keyword arguments are forwarded to :func:`glmnet.glmnet` (``alpha``,
    ``standardize``, ``penalty_factor``, ...). Returns a :class:`CVGlmnet`.
    """
    if family not in _MEASURES:
        raise ValueError(f"family {family!r} not supported for CV")
    allowed, default = _MEASURES[family]
    measure = default if type_measure == "default" else type_measure
    if measure not in allowed:
        raise ValueError(
            f"type_measure {measure!r} not available for {family}; choose from {allowed}"
        )

    y = np.ascontiguousarray(y, dtype=float).ravel()
    n = y.shape[0]
    is_sparse = hasattr(x, "tocsc") and hasattr(x, "shape")
    if not is_sparse:
        x = np.ascontiguousarray(x, dtype=float)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()

    # Full-data fit fixes the lambda grid shared by every fold.
    full = glmnet(x, y, family=family, weights=weights, **glmnet_kwargs)
    lam = full.lambda_
    nlam = lam.size

    foldid, nfolds = _make_folds(n, nfolds, foldid, seed)

    def rows(mat, idx):
        return mat[idx] if not is_sparse else mat.tocsr()[idx]

    # predmat[i, j] = held-out link prediction for obs i at the full-data
    # lambda[j]. Matching R's cv.glmnet: each fold is fit on its OWN lambda path
    # (not the full grid), then its coefficients are interpolated onto the full
    # grid (alignment="lambda"). `predict(..., s=lam)` does that interpolation,
    # clamping s outside the fold's range to the endpoints, exactly as R does.
    predmat = np.empty((n, nlam))
    for k in range(nfolds):
        test = foldid == k
        train = ~test
        fit_k = glmnet(
            rows(x, train),
            y[train],
            family=family,
            weights=w[train].tolist(),
            **glmnet_kwargs,
        )
        x_test = np.asarray(rows(x, test).todense()) if is_sparse else x[test]
        predmat[test] = fit_k.predict(x_test, s=lam, type="link")

    # --- aggregate: per-fold means -> cvm / cvsd (grouped, as R defaults) ----
    if measure == "auc":
        p = 1.0 / (1.0 + np.exp(-predmat))
        outmat = np.full((nfolds, nlam), np.nan)
        fold_w = np.zeros(nfolds)
        for k in range(nfolds):
            test = foldid == k
            fold_w[k] = w[test].sum()
            for j in range(nlam):
                outmat[k, j] = _weighted_auc(y[test], p[test, j], w[test])
    else:
        cvraw, eff_w = _cvraw_per_obs(family, measure, y, predmat, w)
        outmat = np.full((nfolds, nlam), np.nan)
        fold_w = np.zeros(nfolds)
        for k in range(nfolds):
            test = foldid == k
            wi = eff_w[test]
            fold_w[k] = wi.sum()
            outmat[k] = np.average(cvraw[test], axis=0, weights=wi)

    cvm = np.average(outmat, axis=0, weights=fold_w)
    # cvsd is R's between-fold standard error: the fold-weighted mean squared
    # deviation from cvm, divided by (nfolds - 1).
    var = np.average((outmat - cvm) ** 2, axis=0, weights=fold_w)
    cvsd = np.sqrt(var / (nfolds - 1)) if nfolds > 1 else np.zeros(nlam)

    nzero = full.df

    # --- lambda.min / lambda.1se (getOptcv) --------------------------------
    # R takes max(lambda[...]) -- the largest lambda *value* -- not the largest
    # index. lambda descends, so that is the earliest index among candidates.
    score = -cvm if measure in _MAXIMIZE else cvm
    cvmin = np.nanmin(score)
    lambda_min = float(lam[score <= cvmin].max())
    idmin = int(np.where(lam == lambda_min)[0][0])
    semin = (score + cvsd)[idmin]
    lambda_1se = float(lam[score <= semin].max())
    id1se = int(np.where(lam == lambda_1se)[0][0])

    return CVGlmnet(
        lambda_=lam,
        cvm=cvm,
        cvsd=cvsd,
        cvup=cvm + cvsd,
        cvlo=cvm - cvsd,
        nzero=nzero,
        lambda_min=float(lambda_min),
        lambda_1se=float(lambda_1se),
        index_min=idmin,
        index_1se=id1se,
        glmnet_fit=full,
        family=family,
        type_measure=measure,
        foldid=foldid,
    )

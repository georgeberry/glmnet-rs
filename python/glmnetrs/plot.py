"""Coefficient-path and cross-validation plots, matching R's ``plot.glmnet`` and
``plot.cv.glmnet``.

``plot.glmnet`` draws one line per (ever-nonzero) coefficient against a choice of
x-axis -- the L1 norm of the coefficient vector (default), ``-log(lambda)``, or
the fraction of deviance explained -- with a secondary top axis showing the
number of nonzero coefficients (``Df``).

``plot.cv.glmnet`` draws the CV curve (``cvm``) against ``-log(lambda)`` with
one-standard-error bars, a ``Df`` top axis, and vertical dashed lines at
``lambda.min`` and ``lambda.1se``.

matplotlib is an optional dependency; import it only when plotting.
"""

from __future__ import annotations

import numpy as np


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting requires matplotlib: pip install 'glmnet-rs[plot]'") from exc
    return plt


def _df_top_axis(ax, index, df, ticklocs, xlabel):
    """Add R's top ``Df`` axis: at each tick x-position, label the number of
    nonzero coefficients, via constant (step) interpolation of `df` over `index`."""
    order = np.argsort(index)
    xs, ds = index[order], df[order]
    # method="constant", rule=2 (clamp): the df at the largest index <= tick.
    labels = []
    for t in ticklocs:
        j = np.searchsorted(xs, t, side="right") - 1
        j = min(max(j, 0), len(ds) - 1)
        labels.append(int(ds[j]))
    top = ax.secondary_xaxis("top")
    top.set_xticks(ticklocs)
    top.set_xticklabels(labels)
    top.set_xlabel(xlabel)


def plot_coef(fit, xvar="norm", label=False, ax=None, sign_lambda=-1):
    """Plot the coefficient paths. `xvar` is ``"norm"``, ``"lambda"`` or ``"dev"``.

    Returns the matplotlib ``Axes``.
    """
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots()

    beta = fit.beta  # (p, lmu)
    which = np.flatnonzero(np.any(beta != 0.0, axis=1))
    if which.size == 0:
        raise ValueError("no plot produced: all coefficients are zero")
    b = beta[which]

    if xvar == "norm":
        index = np.abs(b).sum(axis=0)
        xlabel = "L1 Norm"
    elif xvar == "lambda":
        index = sign_lambda * np.log(fit.lambda_)
        xlabel = "-Log(Lambda)" if sign_lambda < 0 else "Log(Lambda)"
    elif xvar == "dev":
        index = fit.dev_ratio
        xlabel = "Fraction Deviance Explained"
    else:
        raise ValueError(f"xvar must be 'norm', 'lambda' or 'dev', got {xvar!r}")

    for row in b:
        ax.plot(index, row, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coefficients")

    _df_top_axis(ax, index, fit.df, ax.get_xticks(), "Df")

    if label:
        xpos = index.max() if sign_lambda < 0 or xvar != "lambda" else index.min()
        for w, row in zip(which, b):
            ax.annotate(str(w), (xpos, row[-1]), fontsize=6, va="center")

    return ax


def plot_cv(cv, ax=None, sign_lambda=-1):
    """Plot the cross-validation curve with error bars and min/1se lines.

    Returns the matplotlib ``Axes``.
    """
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots()

    x = sign_lambda * np.log(cv.lambda_)
    ax.errorbar(
        x,
        cv.cvm,
        yerr=cv.cvsd,
        fmt="o",
        ms=3,
        color="red",
        ecolor="darkgrey",
        elinewidth=1,
        capsize=2,
    )
    from .cv import CVGlmnet

    name = CVGlmnet._MEASURE_NAMES.get(cv.type_measure, cv.type_measure)
    ax.set_xlabel("-Log(Lambda)" if sign_lambda < 0 else "Log(Lambda)")
    ax.set_ylabel(name)

    for lam in (cv.lambda_min, cv.lambda_1se):
        ax.axvline(sign_lambda * np.log(lam), ls=":", color="k", lw=1)

    _df_top_axis(ax, x, cv.nzero, ax.get_xticks(), "Df")
    return ax

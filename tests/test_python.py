"""End-to-end checks for the Python layer.

`crates/glmnet-core/tests/parity.rs` already pins the kernels to R glmnet. These
tests cover what sits above the kernels: array marshalling, the path object, and
the scikit-learn parameterization.
"""

import json
import pathlib

import numpy as np
import pytest

from glmnet import glmnet, lambda_interp

FIXTURES = sorted((pathlib.Path(__file__).parent / "fixtures").glob("*.json"))


def _load(path):
    with open(path) as fh:
        d = json.load(fh)
    for k in ("lower_limits", "upper_limits"):
        d[k] = [float(v) for v in d[k]]  # "Inf" / "-Inf" strings from jsonlite
    return d


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_matches_r_glmnet(path):
    """The full Python -> Rust round trip reproduces R glmnet."""
    d = _load(path)
    n, p = d["n"], d["p"]
    X = np.asarray(d["x"], dtype=float).reshape((n, p), order="F")
    y = np.asarray(d["y"], dtype=float)

    fit = glmnet(
        X,
        y,
        alpha=d["alpha"],
        nlambda=d["nlambda"],
        lambda_min_ratio=d["lambda_min_ratio"],
        lambda_=d.get("user_lambda"),
        standardize=d["standardize"],
        intercept=d["intercept"],
        thresh=d["thresh"],
        maxit=int(d["maxit"]),
        dfmax=int(d["dfmax"]),
        pmax=int(d["pmax"]),
        penalty_factor=d["penalty_factor"],
        lower_limits=d["lower_limits"],
        upper_limits=d["upper_limits"],
        weights=d["weights"],
    )

    assert fit.lmu == d["lmu"]
    assert fit.npasses == d["npasses"]

    r_beta = np.asarray(d["beta"], dtype=float).reshape((p, d["lmu"]), order="F")
    np.testing.assert_allclose(fit.lambda_, d["lambda"], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(fit.a0, d["a0"], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(fit.beta, r_beta, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(fit.dev_ratio, d["dev_ratio"], rtol=1e-11, atol=1e-13)


def test_coef_on_grid_is_exact():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 8))
    y = X[:, 0] * 2 - X[:, 3] + rng.standard_normal(60) * 0.5
    fit = glmnet(X, y)
    k = 7
    got = fit.coef(s=fit.lambda_[k]).ravel()
    want = np.concatenate([[fit.a0[k]], fit.beta[:, k]])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_coef_interpolates_and_clamps():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 5))
    y = X[:, 1] * 3 + rng.standard_normal(60) * 0.3
    fit = glmnet(X, y)

    mid = 0.5 * (fit.lambda_[3] + fit.lambda_[4])
    c = fit.coef(s=mid).ravel()
    lo = np.minimum(fit.beta[:, 3], fit.beta[:, 4])
    hi = np.maximum(fit.beta[:, 3], fit.beta[:, 4])
    assert np.all(c[1:] >= lo - 1e-12) and np.all(c[1:] <= hi + 1e-12)

    # s beyond the fitted range clamps to the endpoints; it must not extrapolate.
    big = fit.coef(s=fit.lambda_[0] * 10).ravel()
    np.testing.assert_allclose(big, fit.coef(s=fit.lambda_[0]).ravel(), atol=1e-12)


def test_lambda_interp_single_lambda():
    left, right, frac = lambda_interp(np.array([0.3]), [0.1, 5.0])
    assert np.all(left == 0) and np.all(right == 0) and np.all(frac == 1.0)


def test_predict_matches_manual():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 4))
    y = X @ np.array([1.0, 0, -2.0, 0]) + rng.standard_normal(40) * 0.1
    fit = glmnet(X, y)
    pred = fit.predict(X)
    manual = X @ fit.beta + fit.a0
    np.testing.assert_allclose(pred, manual, rtol=1e-12)


def test_df_and_dev_ratio_monotone():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 10))
    y = X[:, :3] @ np.array([2.0, -1.0, 1.5]) + rng.standard_normal(80) * 0.4
    fit = glmnet(X, y)
    assert fit.df[0] == 0  # first lambda is lambda_max: everything shrunk to zero
    assert np.all(np.diff(fit.dev_ratio) >= -1e-12)  # deviance explained never decreases


def test_lambda_max_zeroes_all_coefficients():
    """lambda[0] is recovered from a sentinel by log-linear extrapolation;
    the coefficients there must be exactly zero."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 6))
    y = rng.standard_normal(50)
    fit = glmnet(X, y)
    assert np.all(fit.beta[:, 0] == 0.0)


@pytest.mark.parametrize("l1_ratio", [1.0, 0.9, 0.7, 0.5, 0.2, 0.05])
@pytest.mark.parametrize("alpha", [0.5, 0.1, 0.01])
def test_sklearn_parameterization_matches_sklearn(alpha, l1_ratio):
    """Our (alpha, l1_ratio) must mean exactly what sklearn's do."""
    sklm = pytest.importorskip("sklearn.linear_model")
    from glmnet.sklearn import ElasticNet

    rng = np.random.default_rng(5)
    X = rng.standard_normal((120, 10))
    y = X[:, :4] @ np.array([3.0, -2.0, 1.0, 0.5]) + rng.standard_normal(120)

    ours = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-12).fit(X, y)
    theirs = sklm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-14, max_iter=1000000).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, rtol=1e-4, atol=1e-6)


def test_sklearn_mapping_is_identity_for_lasso():
    """With l1_ratio=1 the L2 term vanishes, so lambda == alpha and glmnet alpha == 1.

    This is exactly why a wrong mapping goes unnoticed: the lasso case hides it.
    """
    from glmnet.sklearn import _to_glmnet

    lam, a = _to_glmnet(0.3, 1.0, ys=7.5)
    assert lam == pytest.approx(0.3)
    assert a == pytest.approx(1.0)


def test_sklearn_mapping_absorbs_y_scale_for_ridge():
    """Pure ridge: all penalty is L2, so lambda picks up the full ys factor."""
    from glmnet.sklearn import _to_glmnet

    lam, a = _to_glmnet(0.3, 0.0, ys=7.5)
    assert lam == pytest.approx(0.3 * 7.5)
    assert a == pytest.approx(0.0)


def test_glmnet_l2_penalty_carries_inverse_y_scale():
    """Pin the quirk itself: recovered L2 is lambda*(1-alpha)/ys, not lambda*(1-alpha).

    Recovered from the stationarity condition at active coordinates:
        -(1/n) x_j'r + lambda*alpha*sign(b_j) + L2*b_j = 0
    """
    rng = np.random.default_rng(5)
    X = rng.standard_normal((120, 10))
    y = X[:, :4] @ np.array([3.0, -2.0, 1.0, 0.5]) + rng.standard_normal(120)
    ys = np.sqrt(np.mean((y - y.mean()) ** 2))

    lam, a = 0.1, 0.7
    fit = glmnet(X, y, alpha=a, lambda_=[lam], standardize=False, thresh=1e-14)
    b, b0 = fit.beta[:, 0], fit.a0[0]
    r = y - b0 - X @ b

    active = np.flatnonzero(b)
    assert active.size >= 3
    l2_hat = [(X[:, j] @ r / len(y) - lam * a * np.sign(b[j])) / b[j] for j in active]

    np.testing.assert_allclose(l2_hat, lam * (1 - a) / ys, rtol=1e-6)
    assert not np.allclose(l2_hat, lam * (1 - a), rtol=1e-2)


def test_positive_constraint():
    from glmnet.sklearn import Lasso

    rng = np.random.default_rng(6)
    X = rng.standard_normal((80, 6))
    y = X @ np.array([2.0, -3.0, 1.0, 0, 0, -1.0]) + rng.standard_normal(80) * 0.2
    m = Lasso(alpha=0.05, positive=True).fit(X, y)
    assert np.all(m.coef_ >= 0.0)


def test_rejects_positive_lower_limit():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)
    with pytest.raises(ValueError, match="non-positive"):
        glmnet(X, y, lower_limits=0.5)

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

# `sp*` (sparse CSC schema: sp_/spb_/spp_) and `cv_*` (cross-validation schema)
# fixtures are covered by their own tests, not this dense-input parametrization.
FIXTURES = sorted(
    p for p in (pathlib.Path(__file__).parent / "fixtures").glob("*.json")
    if not p.stem.startswith(("sp", "cv_"))
)


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
    if path.stem.startswith("bin_"):
        family = "binomial"
    elif path.stem.startswith("pois_"):
        family = "poisson"
    else:
        family = "gaussian"

    fit = glmnet(
        X,
        y,
        family=family,
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


# --- binomial-specific behaviour ------------------------------------------


def _logistic_data(seed, n=200, p=10, k=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = X[:, :k] @ np.array([1.5, -1.0, 0.8][:k])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


def test_binomial_predict_types_agree():
    X, y = _logistic_data(10)
    fit = glmnet(X, y, family="binomial")
    s = fit.lambda_[20]
    link = fit.predict(X, s=s, type="link").ravel()
    resp = fit.predict(X, s=s, type="response").ravel()
    cls = fit.predict(X, s=s, type="class").ravel()

    np.testing.assert_allclose(resp, 1.0 / (1.0 + np.exp(-link)), rtol=1e-12)
    np.testing.assert_array_equal(cls, (link > 0).astype(float))
    assert np.all((resp >= 0) & (resp <= 1))


def test_gaussian_response_is_identity_link():
    """For gaussian the response link is the identity, so response == link."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((40, 4))
    y = rng.standard_normal(40)
    fit = glmnet(X, y)
    link = fit.predict(X, type="link")
    resp = fit.predict(X, type="response")
    np.testing.assert_array_equal(link, resp)
    # type="class" remains binomial-only.
    with pytest.raises(ValueError, match="binomial"):
        fit.predict(X, type="class")


def test_binomial_rejects_non_binary_y():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((30, 3))
    with pytest.raises(ValueError, match="0/1"):
        glmnet(X, rng.standard_normal(30), family="binomial")


def test_binomial_rejects_single_class():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((30, 3))
    with pytest.raises(ValueError, match="one class"):
        glmnet(X, np.ones(30), family="binomial")


def test_binomial_dev_ratio_monotone_and_bounded():
    X, y = _logistic_data(14)
    fit = glmnet(X, y, family="binomial")
    assert fit.df[0] == 0
    assert fit.dev_ratio[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(fit.dev_ratio) >= -1e-9)
    assert fit.dev_ratio[-1] < 1.0


# --- poisson-specific behaviour -------------------------------------------


def _poisson_data(seed, n=200, p=10, k=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = 0.2 + X[:, :k] @ np.array([0.5, -0.4, 0.3][:k])
    y = rng.poisson(np.exp(eta)).astype(float)
    return X, y


def test_poisson_predict_response_is_exp_link():
    X, y = _poisson_data(20)
    fit = glmnet(X, y, family="poisson")
    s = fit.lambda_[20]
    link = fit.predict(X, s=s, type="link").ravel()
    resp = fit.predict(X, s=s, type="response").ravel()
    np.testing.assert_allclose(resp, np.exp(link), rtol=1e-12)
    assert np.all(resp > 0)


def test_poisson_rejects_negative_y():
    rng = np.random.default_rng(21)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)  # has negatives
    with pytest.raises(ValueError, match="non-negative"):
        glmnet(X, y, family="poisson")


def test_poisson_dev_ratio_monotone_and_bounded():
    X, y = _poisson_data(22)
    fit = glmnet(X, y, family="poisson")
    assert fit.df[0] == 0
    assert fit.dev_ratio[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(fit.dev_ratio) >= -1e-9)
    assert fit.dev_ratio[-1] < 1.0


def test_poisson_class_type_rejected():
    X, y = _poisson_data(23)
    fit = glmnet(X, y, family="poisson")
    with pytest.raises(ValueError, match="binomial"):
        fit.predict(X, type="class")


def test_sklearn_logistic_matches_sklearn():
    """Our LogisticRegression must match sklearn's liblinear/saga solution."""
    sklm = pytest.importorskip("sklearn.linear_model")
    from glmnet.sklearn import LogisticRegression

    X, y = _logistic_data(15, n=300, p=8)
    C = 2.0  # sklearn penalizes 1/C; ours maps it internally
    ours = LogisticRegression(C=C, penalty="l2", tol=1e-10).fit(X, y)
    # sklearn's default penalty is already l2; naming it triggers a FutureWarning
    # on newer versions, so rely on the default here.
    theirs = sklm.LogisticRegression(C=C, solver="lbfgs", tol=1e-10, max_iter=100000).fit(X, y)

    np.testing.assert_allclose(ours.coef_.ravel(), theirs.coef_.ravel(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_.ravel(), rtol=2e-3, atol=2e-3)


# --- sparse X --------------------------------------------------------------


def test_sparse_matches_dense():
    """A scipy CSC matrix must give the same fit as its dense counterpart."""
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(30)
    n, p = 150, 40
    Xd = rng.standard_normal((n, p))
    Xd[rng.random((n, p)) < 0.7] = 0.0  # ~70% zeros
    y = Xd[:, :5] @ np.array([1.5, -1.0, 0.8, 0.0, 0.6]) + rng.standard_normal(n) * 0.4

    dense = glmnet(Xd, y)
    sparse = glmnet(sp.csc_matrix(Xd), y)

    assert sparse.lmu == dense.lmu
    assert sparse.npasses == dense.npasses
    np.testing.assert_allclose(sparse.lambda_, dense.lambda_, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(sparse.a0, dense.a0, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(sparse.beta, dense.beta, rtol=1e-8, atol=1e-9)


def test_sparse_csr_input_accepted():
    """CSR input is converted to CSC internally, so it must also work."""
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(31)
    Xd = rng.standard_normal((80, 20))
    Xd[rng.random((80, 20)) < 0.6] = 0.0
    y = Xd[:, 0] * 2 - Xd[:, 3] + rng.standard_normal(80) * 0.3
    csr = glmnet(sp.csr_matrix(Xd), y)
    dense = glmnet(Xd, y)
    np.testing.assert_allclose(csr.beta, dense.beta, rtol=1e-8, atol=1e-9)


def test_sparse_predict_works():
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(32)
    Xd = rng.standard_normal((60, 10))
    Xd[Xd < 0.5] = 0.0
    y = Xd[:, 1] * 1.5 + rng.standard_normal(60) * 0.3
    fit = glmnet(sp.csc_matrix(Xd), y)
    # predict takes dense X; check it matches the manual linear predictor.
    pred = fit.predict(Xd, s=fit.lambda_[10]).ravel()
    c = fit.coef(s=fit.lambda_[10]).ravel()
    np.testing.assert_allclose(pred, Xd @ c[1:] + c[0], rtol=1e-10)


def test_sparse_rejects_nongaussian():
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(33)
    Xd = rng.standard_normal((50, 6))
    y = (rng.random(50) < 0.5).astype(float)
    with pytest.raises(ValueError, match="gaussian"):
        glmnet(sp.csc_matrix(Xd), y, family="binomial")


# --- cross-validation (cv_glmnet) ------------------------------------------

import glob as _glob

CV_FIXTURES = sorted(_glob.glob(str(pathlib.Path(__file__).parent / "fixtures" / "cv_*.json")))


@pytest.mark.parametrize("path", CV_FIXTURES, ids=lambda p: pathlib.Path(p).stem)
def test_cv_matches_r(path):
    """cv_glmnet reproduces R's cv.glmnet cvm/cvsd and lambda.min/1se given the
    same folds."""
    from glmnet import cv_glmnet

    with open(path) as fh:
        d = json.load(fh)
    n, p = d["n"], d["p"]
    X = np.asarray(d["x"], dtype=float).reshape((n, p), order="F")
    y = np.asarray(d["y"], dtype=float)

    cv = cv_glmnet(
        X,
        y,
        family=d["family"],
        type_measure=d["measure"],
        foldid=np.asarray(d["foldid0"], dtype=int),
        alpha=d["alpha"],
    )

    assert cv.lambda_.size == len(d["lambda"])
    np.testing.assert_allclose(cv.lambda_, d["lambda"], rtol=1e-11, atol=1e-13)
    # cvm/cvsd depend on the fold fits (bit-matched to R), the interpolation onto
    # the full grid, and the exact loss + aggregation formulas -- all reproduced,
    # so this matches to ~1e-13.
    np.testing.assert_allclose(cv.cvm, d["cvm"], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(cv.cvsd, d["cvsd"], rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(cv.nzero, d["nzero"])
    assert cv.lambda_min == pytest.approx(d["lambda_min"], rel=1e-10)
    assert cv.lambda_1se == pytest.approx(d["lambda_1se"], rel=1e-10)


def test_cv_predict_and_coef():
    from glmnet import cv_glmnet

    rng = np.random.default_rng(40)
    X = rng.standard_normal((150, 12))
    y = X[:, :3] @ np.array([2.0, -1.0, 1.5]) + rng.standard_normal(150)
    cv = cv_glmnet(X, y, seed=0)

    # coef/predict at the two named lambdas resolve to the underlying path.
    c_min = cv.coef(s="lambda.min").ravel()
    c_1se = cv.coef(s="lambda.1se").ravel()
    assert c_min.shape == (X.shape[1] + 1,)
    # 1se is a larger lambda -> at least as sparse as min.
    assert np.count_nonzero(c_1se[1:]) <= np.count_nonzero(c_min[1:])
    np.testing.assert_allclose(
        cv.predict(X, s="lambda.min").ravel(),
        X @ c_min[1:] + c_min[0],
        rtol=1e-10,
    )
    assert cv.lambda_1se >= cv.lambda_min


def test_cv_auc_is_sane():
    """AUC isn't bit-matched to R, but must be a valid [0,1] score that peaks at
    a sensible lambda."""
    from glmnet import cv_glmnet

    rng = np.random.default_rng(41)
    X = rng.standard_normal((300, 8))
    y = (rng.random(300) < 1.0 / (1.0 + np.exp(-(X[:, :3] @ [1.5, -1.0, 1.2])))).astype(float)
    cv = cv_glmnet(X, y, family="binomial", type_measure="auc", seed=1)
    assert np.all((cv.cvm >= 0) & (cv.cvm <= 1))
    assert cv.cvm[cv.index_min] > 0.7  # signal is learnable


def test_cv_rejects_bad_measure():
    from glmnet import cv_glmnet

    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 4))
    y = rng.standard_normal(40)
    with pytest.raises(ValueError, match="not available"):
        cv_glmnet(X, y, family="gaussian", type_measure="auc")


# --- summaries -------------------------------------------------------------


def test_path_summary_shape_and_content():
    rng = np.random.default_rng(50)
    X = rng.standard_normal((100, 6))
    y = X[:, 0] * 2 + rng.standard_normal(100)
    fit = glmnet(X, y)
    s = fit.summary()
    lines = s.splitlines()
    assert lines[0].split() == ["Df", "%Dev", "Lambda"]
    assert len(lines) == fit.lmu + 1  # header + one row per lambda
    # First lambda: all coefficients zero -> Df 0, %Dev 0.
    first = lines[1].split()
    assert first[0] == "1" and first[1] == "0" and float(first[2]) == 0.0
    assert str(fit) == s  # __str__ delegates to summary


def test_cv_summary_reports_min_and_1se():
    from glmnet import cv_glmnet

    rng = np.random.default_rng(51)
    X = rng.standard_normal((150, 8))
    y = X[:, :3] @ np.array([2.0, -1.0, 1.5]) + rng.standard_normal(150)
    cv = cv_glmnet(X, y, seed=0)
    s = cv.summary()
    assert "Mean-Squared Error" in s
    lines = s.splitlines()
    assert lines[-2].startswith("min") and lines[-1].startswith("1se")
    # The reported index/measure must match the stored fields.
    assert f"{cv.index_min + 1}" in lines[-2]
    assert f"{cv.index_1se + 1}" in lines[-1]


def test_to_frame_optional_pandas():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(52)
    X = rng.standard_normal((80, 5))
    y = X[:, 1] * 1.5 + rng.standard_normal(80)
    fit = glmnet(X, y)
    frame = fit.to_frame()
    assert list(frame.columns) == ["Df", "pct_dev", "lambda"]
    assert len(frame) == fit.lmu


# --- plotting --------------------------------------------------------------


def _mpl_agg():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")


@pytest.mark.parametrize("xvar", ["norm", "lambda", "dev"])
def test_path_plot_line_count(xvar):
    _mpl_agg()
    rng = np.random.default_rng(60)
    X = rng.standard_normal((120, 8))
    y = X[:, :3] @ np.array([2.0, -1.0, 1.5]) + rng.standard_normal(120)
    fit = glmnet(X, y)
    ax = fit.plot(xvar=xvar)
    # one line per ever-nonzero coefficient.
    n_nonzero = int(np.any(fit.beta != 0, axis=1).sum())
    assert len(ax.get_lines()) == n_nonzero
    assert ax.get_ylabel() == "Coefficients"


def test_path_plot_xlabels():
    _mpl_agg()
    rng = np.random.default_rng(61)
    X = rng.standard_normal((80, 5))
    y = X[:, 0] * 2 + rng.standard_normal(80)
    fit = glmnet(X, y)
    assert fit.plot(xvar="norm").get_xlabel() == "L1 Norm"
    assert fit.plot(xvar="lambda").get_xlabel() == "-Log(Lambda)"
    assert fit.plot(xvar="dev").get_xlabel() == "Fraction Deviance Explained"
    with pytest.raises(ValueError):
        fit.plot(xvar="bogus")


def test_cv_plot_has_min_1se_lines():
    _mpl_agg()
    from glmnet import cv_glmnet

    rng = np.random.default_rng(62)
    X = rng.standard_normal((150, 8))
    y = X[:, :3] @ np.array([2.0, -1.0, 1.5]) + rng.standard_normal(150)
    cv = cv_glmnet(X, y, seed=0)
    ax = cv.plot()
    # two vertical dashed lines (lambda.min, lambda.1se).
    vlines = [ln for ln in ax.get_lines() if ln.get_linestyle() in (":", "dotted")]
    assert len(vlines) == 2
    assert ax.get_ylabel() == "Mean-Squared Error"


def test_cv_plot_binomial_measure_label():
    _mpl_agg()
    from glmnet import cv_glmnet

    rng = np.random.default_rng(63)
    X = rng.standard_normal((200, 6))
    y = (rng.random(200) < 1.0 / (1.0 + np.exp(-(X[:, :3] @ [1.5, -1, 1.2])))).astype(float)
    cv = cv_glmnet(X, y, family="binomial", type_measure="deviance", seed=0)
    assert cv.plot().get_ylabel() == "Deviance"

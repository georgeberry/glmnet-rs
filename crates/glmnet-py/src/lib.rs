//! PyO3 bindings. Deliberately thin: this layer converts numpy arrays to slices
//! and hands back plain arrays. All user-facing ergonomics (the path object,
//! `coef(s=...)`, the scikit-learn estimators) live in the Python package, where
//! they are far easier to iterate on.

use glmnet_core::{elnet_naive, lognet, Control, FitConfig};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The common shape of every family's fit, so packing is written once.
struct CommonFit {
    lmu: usize,
    lambda: Vec<f64>,
    a0: Vec<f64>,
    beta: Vec<f64>,
    dev_ratio: Vec<f64>,
    nulldev: f64,
    npasses: usize,
    warning: Option<(String, i32)>,
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    n: usize,
    p: usize,
    alpha: f64,
    nlambda: usize,
    lambda_min_ratio: Option<f64>,
    user_lambda: Option<Vec<f64>>,
    standardize: bool,
    intercept: bool,
    thresh: f64,
    maxit: usize,
    dfmax: Option<usize>,
    pmax: Option<usize>,
    penalty_factor: Option<Vec<f64>>,
    lower_limits: Option<Vec<f64>>,
    upper_limits: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    exclude: Option<Vec<usize>>,
) -> FitConfig {
    let mut cfg = FitConfig::new(n, p);
    cfg.alpha = alpha;
    cfg.nlambda = nlambda;
    if let Some(r) = lambda_min_ratio {
        cfg.lambda_min_ratio = r;
    }
    cfg.user_lambda = user_lambda;
    cfg.standardize = standardize;
    cfg.intercept = intercept;
    cfg.thresh = thresh;
    cfg.maxit = maxit;
    if let Some(d) = dfmax {
        cfg.dfmax = d;
    }
    // R derives pmax from dfmax, so recompute it whenever dfmax was overridden.
    cfg.pmax = pmax.unwrap_or_else(|| (cfg.dfmax * 2 + 20).min(p));
    cfg.penalty_factor = penalty_factor;
    cfg.lower_limits = lower_limits;
    cfg.upper_limits = upper_limits;
    cfg.weights = weights;
    cfg.exclude = exclude.unwrap_or_default();
    cfg.control = Control::default();
    cfg
}

/// numpy hands us C-order (row-major); the core wants column-major.
fn to_col_major(x: &PyReadonlyArray2<'_, f64>) -> (Vec<f64>, usize, usize) {
    let xr = x.as_array();
    let (n, p) = (xr.nrows(), xr.ncols());
    let mut xcm = Vec::with_capacity(n * p);
    for j in 0..p {
        for i in 0..n {
            xcm.push(xr[[i, j]]);
        }
    }
    (xcm, n, p)
}

fn pack<'py>(py: Python<'py>, p: usize, fit: CommonFit) -> PyResult<Bound<'py, PyDict>> {
    // beta comes back p x lmu column-major; hand numpy a (p, lmu) array.
    let rows: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..fit.lmu).map(|k| fit.beta[k * p + j]).collect())
        .collect();
    let beta = PyArray2::from_vec2(py, &rows)?;

    let out = PyDict::new(py);
    out.set_item("lmu", fit.lmu)?;
    out.set_item("lambda", fit.lambda.into_pyarray(py))?;
    out.set_item("a0", fit.a0.into_pyarray(py))?;
    out.set_item("beta", beta)?;
    out.set_item("dev_ratio", fit.dev_ratio.into_pyarray(py))?;
    out.set_item("nulldev", fit.nulldev)?;
    out.set_item("npasses", fit.npasses)?;
    out.set_item("warning", fit.warning)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x, y, *, family="gaussian", alpha=1.0, nlambda=100, lambda_min_ratio=None,
    user_lambda=None, standardize=true, intercept=true, thresh=1e-7, maxit=100_000,
    dfmax=None, pmax=None, penalty_factor=None, lower_limits=None,
    upper_limits=None, weights=None, exclude=None,
))]
fn elnet_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    family: &str,
    alpha: f64,
    nlambda: usize,
    lambda_min_ratio: Option<f64>,
    user_lambda: Option<Vec<f64>>,
    standardize: bool,
    intercept: bool,
    thresh: f64,
    maxit: usize,
    dfmax: Option<usize>,
    pmax: Option<usize>,
    penalty_factor: Option<Vec<f64>>,
    lower_limits: Option<Vec<f64>>,
    upper_limits: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    exclude: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (xcm, n, p) = to_col_major(&x);
    if y.len()? != n {
        return Err(PyValueError::new_err(format!(
            "x has {n} rows but y has length {}",
            y.len()?
        )));
    }
    let yv = y.as_slice()?.to_vec();

    let cfg = build_config(
        n,
        p,
        alpha,
        nlambda,
        lambda_min_ratio,
        user_lambda,
        standardize,
        intercept,
        thresh,
        maxit,
        dfmax,
        pmax,
        penalty_factor,
        lower_limits,
        upper_limits,
        weights,
        exclude,
    );

    let fit = match family {
        "gaussian" => elnet_naive(&xcm, &yv, n, p, &cfg)
            .map(|f| CommonFit {
                lmu: f.lmu,
                lambda: f.lambda,
                a0: f.a0,
                beta: f.beta,
                dev_ratio: f.dev_ratio,
                nulldev: f.nulldev,
                npasses: f.npasses,
                warning: f.warning.map(|w| (format!("{w:?}"), w.jerr())),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        "binomial" => lognet(&xcm, &yv, n, p, &cfg)
            .map(|f| CommonFit {
                lmu: f.lmu,
                lambda: f.lambda,
                a0: f.a0,
                beta: f.beta,
                dev_ratio: f.dev_ratio,
                nulldev: f.nulldev,
                npasses: f.npasses,
                warning: f.warning.map(|w| (format!("{w:?}"), w.jerr())),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown family {other:?}; expected 'gaussian' or 'binomial'"
            )))
        }
    };

    pack(py, p, fit)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(elnet_path, m)?)?;
    Ok(())
}

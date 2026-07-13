//! PyO3 bindings. Deliberately thin: this layer converts numpy arrays to slices
//! and hands back plain arrays. All user-facing ergonomics (the path object,
//! `coef(s=...)`, the scikit-learn estimators) live in the Python package, where
//! they are far easier to iterate on.

use glmnet_core::{elnet_naive, Control, FitConfig};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x, y, *, alpha=1.0, nlambda=100, lambda_min_ratio=None, user_lambda=None,
    standardize=true, intercept=true, thresh=1e-7, maxit=100_000,
    dfmax=None, pmax=None, penalty_factor=None, lower_limits=None,
    upper_limits=None, weights=None, exclude=None,
))]
fn elnet_gaussian<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
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
    let xr = x.as_array();
    let (n, p) = (xr.nrows(), xr.ncols());
    if y.len()? != n {
        return Err(PyValueError::new_err(format!(
            "x has {n} rows but y has length {}",
            y.len()?
        )));
    }

    // The core wants column-major; numpy hands us C order by default.
    let mut xcm = Vec::with_capacity(n * p);
    for j in 0..p {
        for i in 0..n {
            xcm.push(xr[[i, j]]);
        }
    }

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

    let yv = y.as_slice()?.to_vec();
    let fit =
        elnet_naive(&xcm, &yv, n, p, &cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;

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
    out.set_item("warning", fit.warning.map(|w| (format!("{w:?}"), w.jerr())))?;
    Ok(out)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(elnet_gaussian, m)?)?;
    Ok(())
}

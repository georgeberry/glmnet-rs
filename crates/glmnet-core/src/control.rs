//! Global tuning constants, mirroring R's `glmnet.control()`.
//!
//! These are not cosmetic. `fdev` and `devmax` terminate the lambda path early,
//! so they determine how many lambdas a fit actually returns. Changing them
//! changes output shapes, not just precision.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control {
    /// `fdev`: minimum fractional deviance change before the path stops.
    pub fdev: f64,
    /// `eps`: floor on `lambda.min.ratio` when building the lambda grid.
    pub eps: f64,
    /// `big`: sentinel used as lambda[0] so the first fit yields beta == 0.
    pub big: f64,
    /// `mnlam`: minimum number of lambdas computed before early stopping is allowed.
    pub mnlam: usize,
    /// `devmax`: maximum fraction of deviance explained before the path stops.
    pub devmax: f64,
    /// `pmin`: probability floor (binomial/poisson).
    pub pmin: f64,
    /// `exmx`: max exponent (binomial/poisson).
    pub exmx: f64,
}

impl Default for Control {
    fn default() -> Self {
        Control {
            fdev: 1e-5,
            eps: 1e-6,
            big: 9.9e35,
            mnlam: 5,
            devmax: 0.999,
            pmin: 1e-9,
            exmx: 250.0,
        }
    }
}

/// Per-fit configuration (the arguments a user actually passes to `glmnet()`).
#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Elastic-net mixing parameter. `1.0` = lasso, `0.0` = ridge.
    ///
    /// Note this is glmnet's `alpha`, NOT scikit-learn's `alpha` (which is the
    /// penalty strength, called `lambda` here).
    pub alpha: f64,
    pub nlambda: usize,
    pub lambda_min_ratio: f64,
    /// User-supplied lambda grid (descending). When set, the internal `flmin >= 1` branch runs.
    pub user_lambda: Option<Vec<f64>>,
    pub standardize: bool,
    pub intercept: bool,
    pub thresh: f64,
    pub maxit: usize,
    /// `dfmax`: stop once this many coefficients are nonzero.
    pub dfmax: usize,
    /// `pmax`: hard cap on how many coefficients may ever enter the active set.
    pub pmax: usize,
    pub penalty_factor: Option<Vec<f64>>,
    pub lower_limits: Option<Vec<f64>>,
    pub upper_limits: Option<Vec<f64>>,
    pub weights: Option<Vec<f64>>,
    /// Column indices to force out of the model.
    pub exclude: Vec<usize>,
    pub control: Control,
}

impl FitConfig {
    pub fn new(nobs: usize, nvars: usize) -> Self {
        FitConfig {
            alpha: 1.0,
            nlambda: 100,
            lambda_min_ratio: if nobs < nvars { 1e-2 } else { 1e-4 },
            user_lambda: None,
            standardize: true,
            intercept: true,
            thresh: 1e-7,
            maxit: 100_000,
            dfmax: nvars + 1,
            pmax: ((nvars + 1) * 2 + 20).min(nvars),
            penalty_factor: None,
            lower_limits: None,
            upper_limits: None,
            weights: None,
            exclude: Vec::new(),
            control: Control::default(),
        }
    }
}

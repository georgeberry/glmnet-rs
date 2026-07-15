//! Gaussian elastic-net path, "naive" solver.
//!
//! A transliteration of glmnetpp's `ElnetDriver<gaussian>` / `ElnetPath<gaussian,naive>`
//! / `ElnetPoint<gaussian,naive>`. The C++ splits these across CRTP layers to
//! share code with eleven other solvers; here they are one module because Rust
//! traits will let the other families share code without the indirection.
//!
//! Three behaviours are quirks of glmnet, not of the elastic net, and are
//! reproduced deliberately:
//!
//! * `lambda[0]` is fit at a sentinel `big = 9.9e35` rather than at lambda_max,
//!   which forces `beta == 0`. lambda_max is only computed at `m == 1`. The
//!   reported `lambda[0]` is then recovered by log-linear extrapolation in
//!   [`fix_lam`], which returns exactly lambda_max in exact arithmetic.
//! * The path stops early when the fractional deviance change drops below
//!   `fdev`, so the number of lambdas returned is data-dependent.
//! * The full pass runs only over the *strong set*, screened by the sequential
//!   strong rule `tlam = alpha * (2*lambda_m - lambda_{m-1})`, with a KKT check
//!   afterwards to readmit any variable the rule wrongly discarded.

use crate::control::{Control, FitConfig};
use crate::error::{FitError, PathWarning};
use crate::matrix::{chkvars, chkvars_sparse, Dense, DesignMatrix, Sparse};
use crate::standardize::{standardize_naive, standardize_naive_sparse, Standardization};

#[derive(Clone, Debug)]
pub struct GaussianFit {
    /// Number of lambdas actually computed (may be < `nlambda` due to early stopping).
    pub lmu: usize,
    /// Lambdas, descending, on the original `y` scale. Length `lmu`.
    pub lambda: Vec<f64>,
    /// Intercepts. Length `lmu`.
    pub a0: Vec<f64>,
    /// Coefficients, `p x lmu`, column-major, on the original scale.
    pub beta: Vec<f64>,
    /// Fraction of null deviance explained at each lambda. Length `lmu`.
    pub dev_ratio: Vec<f64>,
    /// Null deviance, `sum(w * (y - ybar)^2)` with raw weights.
    pub nulldev: f64,
    /// Total coordinate-descent passes over the whole path.
    pub npasses: usize,
    /// Set when the path was truncated. Mirrors a negative `jerr` in R.
    pub warning: Option<PathWarning>,
}

#[derive(Debug)]
enum PointErr {
    MaxIter,
    MaxActive,
}

/// Restores `lambda[0]`, which the solver evaluated at the `big` sentinel.
///
/// Since `lambda[1] = lmax*alf` and `lambda[2] = lmax*alf^2`,
/// `exp(2*ln(l1) - ln(l2)) = lmax` exactly.
fn fix_lam(lam: &mut [f64]) {
    if lam.len() > 2 {
        lam[0] = (2.0 * lam[1].ln() - lam[2].ln()).exp();
    }
}

struct Point<'a, M: DesignMatrix> {
    x: &'a M,
    /// Residual, in standardized coordinates. Starts equal to standardized `y`.
    r: Vec<f64>,
    /// Uncompressed coefficients, length `p`.
    a: Vec<f64>,
    /// Cwise-absolute gradient. Only refreshed for variables outside the strong set.
    g: Vec<f64>,
    /// Active variables, in order of entry.
    ia: Vec<usize>,
    in_active: Vec<bool>,
    /// Strong-set indicators. The full pass iterates only over these.
    ix: Vec<bool>,

    xv: &'a [f64],
    vp: &'a [f64],
    cl_lo: &'a [f64],
    cl_hi: &'a [f64],
    ju: &'a [bool],

    thr: f64,
    maxit: usize,
    nx: usize,

    nlp: usize,
    rsq: f64,
    rsq_prev: f64,
    dlx: f64,
    /// True once a partial (active-set-only) fit has been done at some lambda.
    iz: bool,
    /// Gradient at `k`, captured during `update_beta` so `update_rsq` can reuse it.
    gk_cache: f64,
    /// Matrix correction state (unit for dense, mean-shift scalar for sparse).
    corr: M::Corr,
}

impl<'a, M: DesignMatrix> Point<'a, M> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: &'a M,
        y: Vec<f64>,
        xv: &'a [f64],
        vp: &'a [f64],
        cl_lo: &'a [f64],
        cl_hi: &'a [f64],
        ju: &'a [bool],
        thr: f64,
        maxit: usize,
        nx: usize,
    ) -> Self {
        let p = x.ncols();
        let mut pt = Point {
            x,
            r: y,
            a: vec![0.0; p],
            g: vec![0.0; p],
            ia: Vec::with_capacity(nx),
            in_active: vec![false; p],
            ix: vec![false; p],
            xv,
            vp,
            cl_lo,
            cl_hi,
            ju,
            thr,
            maxit,
            nx,
            nlp: 0,
            rsq: 0.0,
            rsq_prev: 0.0,
            dlx: 0.0,
            iz: false,
            gk_cache: 0.0,
            corr: M::Corr::default(),
        };
        // glmnetpp `construct`: seed |grad| for every usable column.
        for k in 0..p {
            if pt.ju[k] {
                pt.g[k] = pt.x.grad(k, &pt.r, pt.corr).abs();
            }
        }
        pt
    }

    #[inline]
    fn update_beta(&mut self, k: usize, ab: f64, dem: f64, gk: f64) {
        self.gk_cache = gk;
        self.a[k] = crate::kernel::soft_threshold(
            self.a[k],
            gk,
            self.xv[k],
            self.vp[k],
            self.cl_lo[k],
            self.cl_hi[k],
            ab,
            dem,
        );
    }

    fn push_active(&mut self, k: usize) -> Result<(), PointErr> {
        if self.ia.len() >= self.nx {
            return Err(PointErr::MaxActive);
        }
        self.ia.push(k);
        self.in_active[k] = true;
        Ok(())
    }

    #[inline]
    fn update_one(&mut self, k: usize, full: bool, ab: f64, dem: f64) -> Result<(), PointErr> {
        let gk = self.x.grad(k, &self.r, self.corr);
        let a_old = self.a[k];
        self.update_beta(k, ab, dem, gk);
        if self.a[k] == a_old {
            return Ok(());
        }
        if full && !self.in_active[k] {
            self.push_active(k)?;
        }
        let diff = self.a[k] - a_old;
        self.dlx = self.dlx.max(self.xv[k] * diff * diff);
        self.rsq += diff * (2.0 * self.gk_cache - diff * self.xv[k]);
        self.x.update_resid(k, diff, &mut self.r, &mut self.corr);
        Ok(())
    }

    /// One coordinate-descent sweep. Returns `(converged, kkt_passed)`.
    fn fit_pass(
        &mut self,
        full: bool,
        do_kkt: bool,
        ab: f64,
        dem: f64,
    ) -> Result<(bool, bool), PointErr> {
        self.nlp += 1;
        self.dlx = 0.0;

        if full {
            for k in 0..self.x.ncols() {
                if !self.ix[k] {
                    continue;
                }
                self.update_one(k, true, ab, dem)?;
            }
        } else {
            for idx in 0..self.ia.len() {
                let k = self.ia[idx];
                self.update_one(k, false, ab, dem)?;
            }
        }
        // (gaussian has no intercept update: y is already centered)

        if self.dlx < self.thr {
            return Ok((true, if do_kkt { self.check_kkt(ab) } else { true }));
        }
        if self.nlp > self.maxit {
            return Err(PointErr::MaxIter);
        }
        Ok((false, false))
    }

    /// Recompute |grad| outside the strong set; readmit any KKT violator.
    /// Returns true when no variable had to be readmitted.
    fn check_kkt(&mut self, ab: f64) -> bool {
        for k in 0..self.x.ncols() {
            if self.ix[k] || !self.ju[k] {
                continue;
            }
            self.g[k] = self.x.grad(k, &self.r, self.corr).abs();
        }
        let mut updated = false;
        for k in 0..self.x.ncols() {
            if self.ix[k] || !self.ju[k] {
                continue;
            }
            if self.g[k] > ab * self.vp[k] {
                self.ix[k] = true;
                updated = true;
            }
        }
        !updated
    }

    /// Sequential strong rule (Tibshirani et al. 2012).
    fn initialize_strong_set(&mut self, alpha: f64, alm: f64, alm0: f64) {
        let tlam = alpha * (2.0 * alm - alm0);
        for k in 0..self.x.ncols() {
            if self.ix[k] || !self.ju[k] {
                continue;
            }
            if self.g[k] > tlam * self.vp[k] {
                self.ix[k] = true;
            }
        }
    }

    fn partial_fit(&mut self, ab: f64, dem: f64) -> Result<(), PointErr> {
        self.iz = true;
        loop {
            let (converged, _) = self.fit_pass(false, false, ab, dem)?;
            if converged {
                return Ok(());
            }
        }
    }

    /// Full passes until either non-convergence, or convergence with KKT satisfied.
    fn initial_fit(&mut self, ab: f64, dem: f64) -> Result<bool, PointErr> {
        loop {
            if self.nlp > self.maxit {
                return Err(PointErr::MaxIter);
            }
            let (converged, kkt) = self.fit_pass(true, true, ab, dem)?;
            if !converged {
                return Ok(false);
            }
            if kkt {
                return Ok(true);
            }
        }
    }

    fn solve_at(
        &mut self,
        alpha: f64,
        alm: f64,
        alm0: f64,
        ab: f64,
        dem: f64,
    ) -> Result<(), PointErr> {
        self.rsq_prev = self.rsq;
        self.initialize_strong_set(alpha, alm, alm0);

        if self.iz {
            self.partial_fit(ab, dem)?;
        }
        loop {
            if self.initial_fit(ab, dem)? {
                return Ok(());
            }
            self.partial_fit(ab, dem)?;
        }
    }
}

/// Fit the Gaussian elastic-net path with the naive solver.
///
/// `x_col_major` is `n * p`, column-major. `y` has length `n`.
pub fn elnet_naive(
    x_col_major: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    cfg: &FitConfig,
) -> Result<GaussianFit, FitError> {
    assert_eq!(x_col_major.len(), n * p);
    assert_eq!(y.len(), n);

    let mut ctl = cfg.control;

    // --- inclusion set -----------------------------------------------------
    let mut x = Dense::from_col_major(x_col_major.to_vec(), n, p);
    let mut ju = chkvars(&x);
    for &j in &cfg.exclude {
        if j < p {
            ju[j] = false;
        }
    }
    if !ju.iter().any(|&b| b) {
        return Err(FitError::AllExcluded);
    }

    // --- penalty / box / nulldev (shared with the sparse path) -------------
    let vp = normalize_penalty(cfg, p)?;
    let (cl_lo, cl_hi) = box_limits(cfg, p, &mut ctl)?;

    let w_raw = cfg.weights.clone().unwrap_or_else(|| vec![1.0; n]);
    let nulldev = null_deviance(y, &w_raw, cfg.intercept);

    // --- standardize (dense: modifies x and y in place) --------------------
    let mut yv = y.to_vec();
    let mut w = w_raw.clone();
    let st = standardize_naive(&mut x, &mut yv, &mut w, cfg.standardize, cfg.intercept, &ju);

    Ok(run_path(
        x, yv, &st, &ju, &vp, cl_lo, cl_hi, nulldev, cfg, ctl,
    ))
}

/// Fit the Gaussian elastic-net path for a **sparse** design matrix (CSC).
///
/// `col_ptr` (length `p+1`), `row_idx` and `values` are the CSC arrays; `y` has
/// length `n`. The matrix is never centered -- the standardization correction is
/// folded into the solver (see [`crate::matrix::Sparse`]).
pub fn elnet_naive_sparse(
    n: usize,
    p: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
    y: &[f64],
    cfg: &FitConfig,
) -> Result<GaussianFit, FitError> {
    assert_eq!(col_ptr.len(), p + 1);
    assert_eq!(row_idx.len(), values.len());
    assert_eq!(y.len(), n);

    let mut ctl = cfg.control;

    // --- inclusion set -----------------------------------------------------
    let mut ju = chkvars_sparse(n, p, col_ptr, values);
    for &j in &cfg.exclude {
        if j < p {
            ju[j] = false;
        }
    }
    if !ju.iter().any(|&b| b) {
        return Err(FitError::AllExcluded);
    }

    // --- penalty / box / nulldev (identical to the dense path) -------------
    let vp = normalize_penalty(cfg, p)?;
    let (cl_lo, cl_hi) = box_limits(cfg, p, &mut ctl)?;

    let w_raw = cfg.weights.clone().unwrap_or_else(|| vec![1.0; n]);
    let nulldev = null_deviance(y, &w_raw, cfg.intercept);

    // --- sparse standardize (leaves X untouched; y centered/scaled) --------
    let mut yv = y.to_vec();
    let mut w = w_raw.clone();
    let st = standardize_naive_sparse(
        col_ptr,
        row_idx,
        values,
        &mut yv,
        &mut w,
        p,
        cfg.standardize,
        cfg.intercept,
        &ju,
    );

    let x = Sparse::new(
        n,
        p,
        col_ptr.to_vec(),
        row_idx.to_vec(),
        values.to_vec(),
        w,
        st.xm.clone(),
        st.xs.clone(),
    );

    Ok(run_path(
        x, yv, &st, &ju, &vp, cl_lo, cl_hi, nulldev, cfg, ctl,
    ))
}

/// Normalize penalty factors to sum to `p` over all columns (glmnetpp).
fn normalize_penalty(cfg: &FitConfig, p: usize) -> Result<Vec<f64>, FitError> {
    let mut vp = cfg.penalty_factor.clone().unwrap_or_else(|| vec![1.0; p]);
    if vp.iter().cloned().fold(f64::NEG_INFINITY, f64::max) <= 0.0 {
        return Err(FitError::NonPositivePenalty);
    }
    vp.iter_mut().for_each(|v| *v = v.max(0.0));
    let vsum: f64 = vp.iter().sum();
    vp.iter_mut().for_each(|v| *v *= p as f64 / vsum);
    Ok(vp)
}

/// Substitute `+-big` for `+-Inf`, validate signs, and disable `fdev` when a
/// bound is exactly zero (glmnet.R:510). Returns the substituted limits.
fn box_limits(
    cfg: &FitConfig,
    p: usize,
    ctl: &mut Control,
) -> Result<(Vec<f64>, Vec<f64>), FitError> {
    let big = ctl.big;
    let subst = |v: f64| {
        if v == f64::NEG_INFINITY {
            -big
        } else if v == f64::INFINITY {
            big
        } else {
            v
        }
    };
    let cl_lo: Vec<f64> = cfg
        .lower_limits
        .clone()
        .unwrap_or_else(|| vec![f64::NEG_INFINITY; p])
        .into_iter()
        .map(subst)
        .collect();
    let cl_hi: Vec<f64> = cfg
        .upper_limits
        .clone()
        .unwrap_or_else(|| vec![f64::INFINITY; p])
        .into_iter()
        .map(subst)
        .collect();
    if cl_lo.iter().any(|&v| v > 0.0) {
        return Err(FitError::PositiveLowerLimit);
    }
    if cl_hi.iter().any(|&v| v < 0.0) {
        return Err(FitError::NegativeUpperLimit);
    }
    if cl_lo.iter().chain(cl_hi.iter()).any(|&v| v == 0.0) {
        ctl.fdev = 0.0;
    }
    Ok((cl_lo, cl_hi))
}

/// Null deviance `sum(w * (y - ybar)^2)` with raw (un-normalized) weights.
fn null_deviance(y: &[f64], w_raw: &[f64], intercept: bool) -> f64 {
    let wsum_raw: f64 = w_raw.iter().sum();
    let ybar = if intercept {
        y.iter().zip(w_raw).map(|(yi, wi)| yi * wi).sum::<f64>() / wsum_raw
    } else {
        0.0
    };
    y.iter()
        .zip(w_raw)
        .map(|(yi, wi)| wi * (yi - ybar).powi(2))
        .sum()
}

/// The shared lambda path: coordinate descent at each lambda, early stopping,
/// and un-standardization. Generic over dense/sparse via [`DesignMatrix`].
///
/// `cl_lo`/`cl_hi` are the substituted (not yet ys/xs-scaled) box limits.
#[allow(clippy::too_many_arguments)]
fn run_path<M: DesignMatrix>(
    x: M,
    r_init: Vec<f64>,
    st: &Standardization,
    ju: &[bool],
    vp: &[f64],
    mut cl_lo: Vec<f64>,
    mut cl_hi: Vec<f64>,
    nulldev: f64,
    cfg: &FitConfig,
    ctl: Control,
) -> GaussianFit {
    let p = x.ncols();

    // Rescale box limits into standardized coordinates (cl /= ys; *= xs if isd).
    for j in 0..p {
        cl_lo[j] /= st.ys;
        cl_hi[j] /= st.ys;
        if cfg.standardize {
            cl_lo[j] *= st.xs[j];
            cl_hi[j] *= st.xs[j];
        }
    }

    // --- lambda grid setup -------------------------------------------------
    let (flmin, nlam, ulam): (f64, usize, Vec<f64>) = match &cfg.user_lambda {
        Some(l) => {
            let mut l = l.clone();
            l.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
            let scaled = l.iter().map(|v| v / st.ys).collect();
            (1.0, l.len(), scaled)
        }
        None => (cfg.lambda_min_ratio, cfg.nlambda, Vec::new()),
    };

    let omb = 1.0 - cfg.alpha;
    let alf = if flmin < 1.0 {
        let eqs = ctl.eps.max(flmin);
        eqs.powf(1.0 / (nlam as f64 - 1.0))
    } else {
        1.0
    };
    let mnl = ctl.mnlam.min(nlam);

    let nx = cfg.pmax.min(p);
    let mut pt = Point::new(
        &x, r_init, &st.xv, vp, &cl_lo, &cl_hi, ju, cfg.thresh, cfg.maxit, nx,
    );

    // --- path --------------------------------------------------------------
    let mut ca = vec![0.0; nx * nlam]; // compressed betas, nx x nlam column-major
    let mut nin = vec![0usize; nlam];
    let mut almo = vec![0.0; nlam];
    let mut rsqo = vec![0.0; nlam];
    let mut lmu = 0usize;
    let mut warning = None;

    let mut lmda_curr = 0.0f64;

    for m in 0..nlam {
        let mut alm0 = lmda_curr;
        let mut alm = alm0;
        if flmin >= 1.0 {
            alm = ulam[m];
        } else if m > 1 {
            alm *= alf;
        } else if m == 0 {
            alm = ctl.big;
        } else {
            // m == 1: this is where lambda_max is actually computed.
            alm0 = 0.0;
            for j in 0..p {
                if !ju[j] || vp[j] <= 0.0 {
                    continue;
                }
                alm0 = alm0.max(pt.g[j].abs() / vp[j]);
            }
            alm0 /= cfg.alpha.max(1e-3); // guard so alpha = 0 (ridge) still yields a finite start
            alm = alm0 * alf;
        }
        lmda_curr = alm;
        let dem = alm * omb;
        let ab = alm * cfg.alpha;

        match pt.solve_at(cfg.alpha, alm, alm0, ab, dem) {
            Ok(()) => {}
            Err(PointErr::MaxIter) => {
                // glmnetpp: `return` -- the current lambda is discarded entirely.
                warning = Some(PathWarning::MaxIterReached { lambda_index: m });
                break;
            }
            Err(PointErr::MaxActive) => {
                warning = Some(PathWarning::MaxActiveReached { lambda_index: m });
                break;
            }
        }

        // process_point_fit
        for (l, &k) in pt.ia.iter().enumerate() {
            ca[m * nx + l] = pt.a[k];
        }
        nin[m] = pt.ia.len();
        rsqo[m] = pt.rsq;
        almo[m] = alm;
        lmu = m + 1;

        let me = pt.ia.iter().filter(|&&k| pt.a[k] != 0.0).count();
        let prop_dev_change = if pt.rsq == 0.0 {
            f64::INFINITY
        } else {
            (pt.rsq - pt.rsq_prev) / pt.rsq
        };

        if lmu < mnl || flmin >= 1.0 {
            continue;
        }
        if me > cfg.dfmax || prop_dev_change < ctl.fdev || pt.rsq > ctl.devmax {
            break;
        }
    }

    // --- unstandardize -----------------------------------------------------
    let mut lambda = vec![0.0; lmu];
    let mut a0 = vec![0.0; lmu];
    let mut beta = vec![0.0; p * lmu];

    for k in 0..lmu {
        lambda[k] = almo[k] * st.ys;
        let mut intercept = 0.0;
        for l in 0..nin[k] {
            let j = pt.ia[l];
            let b = ca[k * nx + l] * st.ys / st.xs[j];
            beta[k * p + j] = b;
            intercept -= b * st.xm[j];
        }
        a0[k] = if cfg.intercept {
            intercept + st.ym
        } else {
            0.0
        };
    }

    if cfg.user_lambda.is_none() {
        fix_lam(&mut lambda);
    }

    GaussianFit {
        lmu,
        lambda,
        a0,
        beta,
        dev_ratio: rsqo[..lmu].to_vec(),
        nulldev,
        npasses: pt.nlp,
        warning,
    }
}

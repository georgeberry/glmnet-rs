//! Binomial (two-class logistic) elastic-net path.
//!
//! Transliterated from glmnetpp `ElnetDriver<binomial>` /
//! `ElnetPath<binomial,two_class>` / `ElnetPointInternal<binomial,two_class>`.
//!
//! The new machinery relative to [`crate::gaussian`] is the **IRLS-over-WLS**
//! loop, which every non-Gaussian family reuses:
//!
//! ```text
//! for each lambda:
//!     initialize()                       # grow the strong set
//!     loop (IRLS):
//!         setup_wls()                    # freeze working weights v; save b -> bs
//!         wls()                          # coordinate descent at fixed v
//!         update_irls()                  # recompute q, v, r; test convergence + KKT
//! ```
//!
//! State (`b`, `q`, `v`, `r`, the strong set) persists across lambdas: each new
//! lambda warm-starts from the previous fit. That warm start is not an
//! optimization here, it is required for parity -- glmnet's iterate counts and
//! its `fdev` early-stop both depend on it.
//!
//! Differences from Gaussian worth flagging:
//! * `X` is centered but not `sqrt(w)`-scaled; weights live in `v = w*q*(1-q)`.
//! * `y` is not rescaled, so lambda is already on the natural scale.
//! * The `fdev` test compares the *absolute* change in the deviance ratio,
//!   whereas Gaussian compares the *relative* change in R^2.
//! * Probabilities are clamped to `[pmin, 1-pmin]` (via `fmin`/`fmax` on the
//!   linear predictor) so a separating hyperplane cannot send coefficients to
//!   infinity. When the total working variance collapses below `vmin`, the path
//!   stops -- this is the logistic analogue of Gaussian saturation.
//!
//! Scope: dense `X`, exact-Hessian Newton (`kopt = 0`, R's default
//! `type.logistic = "Newton"`), no offset. The modified-Newton upper bound
//! (`kopt = 1`) and offsets are not implemented.

use crate::control::FitConfig;
use crate::error::{FitError, PathWarning};
use crate::glm_matrix::{new_sparse_glm, DenseGlm, GlmMatrix};
use crate::kernel::soft_threshold;
use crate::matrix::{chkvars, chkvars_sparse, Dense};
use crate::standardize::{standardize_lognet, standardize_lognet_sparse};

#[derive(Clone, Debug)]
pub struct BinomialFit {
    /// Number of lambdas actually computed.
    pub lmu: usize,
    /// Lambdas, descending. Length `lmu`.
    pub lambda: Vec<f64>,
    /// Intercepts. Length `lmu`.
    pub a0: Vec<f64>,
    /// Coefficients, `p x lmu`, column-major, on the original scale.
    pub beta: Vec<f64>,
    /// Fraction of null deviance explained at each lambda. Length `lmu`.
    pub dev_ratio: Vec<f64>,
    /// Null deviance, `2 * sum(w) * dev0` on the original weight scale.
    pub nulldev: f64,
    /// Total coordinate-descent passes over the whole path.
    pub npasses: usize,
    pub warning: Option<PathWarning>,
}

#[derive(Debug)]
enum PointErr {
    MaxIter,
    MaxActive,
}

fn fix_lam(lam: &mut [f64]) {
    if lam.len() > 2 {
        lam[0] = (2.0 * lam[1].ln() - lam[2].ln()).exp();
    }
}

/// Persistent two-class logistic point solver. Lives for the whole path so that
/// `b`, `q`, `v`, `r`, and the strong set carry between lambdas. Generic over the
/// design matrix via [`GlmMatrix`], so dense and sparse share this solver; the
/// matrix owns any standardization-correction state (see `glm_matrix.rs`).
struct Point<'a, M: GlmMatrix> {
    mat: M,
    y: &'a [f64],
    w: &'a [f64], // normalized to sum 1

    b0: f64,
    b: Vec<f64>,
    /// Saved (pre-WLS) intercept and coefficients, for the IRLS convergence test.
    bs0: f64,
    bs: Vec<f64>,

    q: Vec<f64>,   // probability predictions
    v: Vec<f64>,   // IRLS working weights, w*q*(1-q)
    r: Vec<f64>,   // working residual, w*(y-q)
    eta: Vec<f64>, // scratch for the linear predictor, reused each IRLS step
    xmz: f64,      // sum of v

    xv: Vec<f64>, // weighted column variance, recomputed each IRLS step (kopt=0)
    ga: Vec<f64>, // abs gradient
    ix: Vec<bool>,

    ia: Vec<usize>,
    in_active: Vec<bool>,

    vp: &'a [f64],
    cl_lo: &'a [f64],
    cl_hi: &'a [f64],
    ju: &'a [bool],

    intr: bool,
    thr_scaled: f64, // thresh * dev0
    maxit: usize,
    nx: usize,
    fmax: f64, // clamp on the linear predictor, log(1/pmin - 1)
    vmin: f64,

    nlp: usize,
    dlx: f64,
    dev0: f64, // null deviance term (internal scale)
    dev1: f64, // null-model deviance, constant for two-class
}

impl<'a, M: GlmMatrix> Point<'a, M> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mat: M,
        y: &'a [f64],
        w: &'a [f64],
        vp: &'a [f64],
        cl_lo: &'a [f64],
        cl_hi: &'a [f64],
        ju: &'a [bool],
        intr: bool,
        thresh: f64,
        maxit: usize,
        nx: usize,
        pmin: f64,
    ) -> Result<Self, FitError> {
        let n = mat.nrows();
        let p = mat.ncols();

        // Weighted mean of y (weights already sum to 1).
        let q0: f64 = w.iter().zip(y).map(|(wi, yi)| wi * yi).sum();
        if q0 <= pmin {
            return Err(FitError::ProbMinReached);
        }
        if q0 >= 1.0 - pmin {
            return Err(FitError::ProbMaxReached);
        }

        // Null model. Offset is zero, so this is the closed form.
        let (b0, qc) = if intr {
            (((q0 / (1.0 - q0)).ln()), q0)
        } else {
            (0.0, 0.5)
        };
        let vi = qc * (1.0 - qc);
        let q = vec![qc; n];
        let v: Vec<f64> = w.iter().map(|wi| vi * wi).collect();
        let r: Vec<f64> = w.iter().zip(y).map(|(wi, yi)| wi * (yi - qc)).collect();
        let xmz = vi; // sum_i v_i = vi * sum_i w_i = vi

        // dev = -(b0*qc + log(1-qc)); dev0 adds the saturated correction, which
        // is zero for 0/1 responses but nonzero for fractional y.
        let dev_null = -(b0 * qc + (1.0 - qc).ln());
        let mut dev0 = dev_null;
        for (wi, &yi) in w.iter().zip(y) {
            if yi > 0.0 {
                dev0 += wi * yi * yi.ln();
            }
            if yi < 1.0 {
                dev0 += wi * (1.0 - yi) * (1.0 - yi).ln();
            }
        }

        let fmax = (1.0 / pmin - 1.0).ln();
        let vmin = (1.0 + pmin) * pmin * (1.0 - pmin);

        let mut pt = Point {
            mat,
            y,
            w,
            b0,
            b: vec![0.0; p],
            bs0: 0.0,
            bs: vec![0.0; p],
            q,
            v,
            r,
            eta: vec![0.0; n],
            xmz,
            xv: vec![0.0; p],
            ga: vec![0.0; p],
            ix: vec![false; p],
            ia: Vec::with_capacity(nx),
            in_active: vec![false; p],
            vp,
            cl_lo,
            cl_hi,
            ju,
            intr,
            thr_scaled: thresh * dev0,
            maxit,
            nx,
            fmax,
            vmin,
            nlp: 0,
            dlx: 0.0,
            dev0,
            dev1: dev_null,
        };

        // Initialize the matrix shift accumulators from the initial residual
        // (sparse: svr = sum r, o = 0), then seed |grad_j| for the strong rule.
        pt.mat.reset_shifts(&pt.r);
        for j in 0..p {
            if pt.ju[j] {
                pt.ga[j] = pt.mat.grad(j, &pt.r, &pt.v).abs();
            }
        }
        Ok(pt)
    }

    #[inline]
    fn is_excluded(&self, k: usize) -> bool {
        !self.ix[k]
    }

    /// Add column `k` to the active set (and let the matrix record its state).
    fn push_active(&mut self, k: usize) -> Result<(), PointErr> {
        if self.ia.len() >= self.nx {
            return Err(PointErr::MaxActive);
        }
        self.ia.push(k);
        self.in_active[k] = true;
        self.mat.on_activate(k, &self.v);
        Ok(())
    }

    /// One coordinate update within a WLS pass (weights `v`, `xv` held fixed).
    #[inline]
    fn update_one(&mut self, k: usize, full: bool, l1: f64, l2: f64) -> Result<(), PointErr> {
        let gk = self.mat.grad(k, &self.r, &self.v);
        let a_old = self.b[k];
        self.b[k] = soft_threshold(
            a_old,
            gk,
            self.xv[k],
            self.vp[k],
            self.cl_lo[k],
            self.cl_hi[k],
            l1,
            l2,
        );
        if self.b[k] == a_old {
            return Ok(());
        }
        if full && !self.in_active[k] {
            self.push_active(k)?;
        }
        let diff = self.b[k] - a_old;
        self.dlx = self.dlx.max(self.xv[k] * diff * diff);
        self.mat
            .update_resid(k, diff, &mut self.r, &self.v, self.xmz);
        Ok(())
    }

    /// Intercept update, run at the end of every WLS pass.
    #[inline]
    fn update_intercept(&mut self) {
        if !self.intr {
            return;
        }
        let r_sum = self.mat.resid_sum(&self.r);
        let d = r_sum / self.xmz;
        if d != 0.0 {
            self.b0 += d;
            self.dlx = self.dlx.max(self.xmz * d * d);
            self.mat.update_intercept(d, &mut self.r, &self.v, self.xmz);
        }
    }

    /// One WLS sweep. `full` iterates the strong set, else the active set.
    fn wls_pass(&mut self, full: bool, l1: f64, l2: f64) -> Result<bool, PointErr> {
        self.nlp += 1;
        self.dlx = 0.0;

        if full {
            for k in 0..self.mat.ncols() {
                if self.is_excluded(k) {
                    continue;
                }
                self.update_one(k, true, l1, l2)?;
            }
        } else {
            for idx in 0..self.ia.len() {
                let k = self.ia[idx];
                self.update_one(k, false, l1, l2)?;
            }
        }
        self.update_intercept();

        if self.dlx < self.thr_scaled {
            return Ok(true);
        }
        if self.nlp > self.maxit {
            return Err(PointErr::MaxIter);
        }
        Ok(false)
    }

    /// Coordinate descent to convergence at fixed working weights.
    fn wls(&mut self, l1: f64, l2: f64) -> Result<(), PointErr> {
        loop {
            if self.wls_pass(true, l1, l2)? {
                break;
            }
            loop {
                if self.wls_pass(false, l1, l2)? {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Save current coefficients and refresh `xv` from the frozen weights.
    fn setup_wls(&mut self) {
        self.bs0 = self.b0;
        for &k in &self.ia {
            self.bs[k] = self.b[k];
        }
        // kopt == 0: exact Hessian, so xv tracks the current weights.
        for k in 0..self.mat.ncols() {
            if self.ix[k] {
                self.xv[k] = self.mat.xv(k, &self.v, self.xmz);
            }
        }
    }

    /// Recompute probabilities, working weights and residual after a WLS solve.
    /// Returns false if the working variance collapsed (stop the path).
    fn update_irls_invariants(&mut self) -> bool {
        let n = self.mat.nrows();
        // Linear predictor eta = b0 + sum_{k active} beta_k z_.k. The matrix
        // handles dense vs sparse accumulation. Swap the scratch buffer out so we
        // can write it while reading self.mat; swapped back after (no allocation).
        let mut eta = std::mem::take(&mut self.eta);
        self.mat
            .accumulate_eta(&self.ia, &self.b, self.b0, &mut eta);
        for (qi, &fi) in self.q.iter_mut().zip(eta.iter()) {
            *qi = if fi < -self.fmax {
                0.0
            } else if fi > self.fmax {
                1.0
            } else {
                1.0 / (1.0 + (-fi).exp())
            };
        }
        self.eta = eta;
        self.xmz = 0.0;
        for i in 0..n {
            self.v[i] = self.w[i] * self.q[i] * (1.0 - self.q[i]);
            self.xmz += self.v[i];
        }
        if self.xmz <= self.vmin {
            return false;
        }
        for i in 0..n {
            self.r[i] = self.w[i] * (self.y[i] - self.q[i]);
        }
        // Sparse: reset o = 0 and svr = sum(r) now that r has been recomputed.
        self.mat.reset_shifts(&self.r);
        true
    }

    /// IRLS convergence: coefficients stable since `setup_wls`.
    fn has_converged_irls(&self) -> bool {
        let d0 = self.b0 - self.bs0;
        if self.xmz * d0 * d0 > self.thr_scaled {
            return false;
        }
        for &k in &self.ia {
            let d = self.b[k] - self.bs[k];
            if self.xv[k] * d * d > self.thr_scaled {
                return false;
            }
        }
        true
    }

    /// KKT check over variables outside the strong set; readmit violators.
    /// Returns true when none were readmitted (the strong set is complete).
    fn kkt_complete(&mut self, l1: f64) -> bool {
        for k in 0..self.mat.ncols() {
            if !self.is_excluded(k) || !self.ju[k] {
                continue;
            }
            self.ga[k] = self.mat.grad(k, &self.r, &self.v).abs();
        }
        let mut ok = true;
        for k in 0..self.mat.ncols() {
            if !self.is_excluded(k) || !self.ju[k] {
                continue;
            }
            if self.ga[k] > l1 * self.vp[k] {
                self.ix[k] = true;
                ok = false;
            }
        }
        ok
    }

    /// Grow the strong set for a new lambda (sequential strong rule).
    fn initialize_strong_set(&mut self, alpha: f64, alm: f64, alm0: f64) {
        let tlam = alpha * (2.0 * alm - alm0);
        for k in 0..self.mat.ncols() {
            if self.ix[k] || !self.ju[k] {
                continue;
            }
            if self.ga[k] > tlam * self.vp[k] {
                self.ix[k] = true;
            }
        }
    }

    /// Full IRLS solve at one lambda. Returns false if the path should stop
    /// (working variance collapsed).
    fn solve_at(
        &mut self,
        alpha: f64,
        alm: f64,
        alm0: f64,
        l1: f64,
        l2: f64,
    ) -> Result<bool, PointErr> {
        self.initialize_strong_set(alpha, alm, alm0);
        loop {
            if self.nlp > self.maxit {
                return Err(PointErr::MaxIter);
            }
            self.setup_wls();
            self.wls(l1, l2)?;
            if !self.update_irls_invariants() {
                return Ok(false);
            }
            if self.has_converged_irls() && self.kkt_complete(l1) {
                return Ok(true);
            }
        }
    }

    /// Current model deviance with probabilities clamped to `[pmin, 1-pmin]`
    /// (glmnetpp `dev2`).
    fn dev2(&self, pmin: f64) -> f64 {
        let pmax = 1.0 - pmin;
        let mut s = 0.0;
        for i in 0..self.mat.nrows() {
            let pi = self.q[i].max(pmin).min(pmax);
            s -= self.w[i] * (self.y[i] * pi.ln() + (1.0 - self.y[i]) * (1.0 - pi).ln());
        }
        s
    }
}

/// Fit the two-class logistic elastic-net path.
///
/// `x_col_major` is `n * p`, column-major. `y` is 0/1 (fractional values are
/// permitted and treated as observed proportions, matching glmnet).
pub fn lognet(
    x_col_major: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    cfg: &FitConfig,
) -> Result<BinomialFit, FitError> {
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

    // --- penalty factors ---------------------------------------------------
    let mut vp = cfg.penalty_factor.clone().unwrap_or_else(|| vec![1.0; p]);
    if vp.iter().cloned().fold(f64::NEG_INFINITY, f64::max) <= 0.0 {
        return Err(FitError::NonPositivePenalty);
    }
    vp.iter_mut().for_each(|v| *v = v.max(0.0));
    let vsum: f64 = vp.iter().sum();
    vp.iter_mut().for_each(|v| *v *= p as f64 / vsum);

    // --- weights: normalize to sum 1, remember the raw total for nulldev ----
    let w_raw = cfg.weights.clone().unwrap_or_else(|| vec![1.0; n]);
    let sw: f64 = w_raw.iter().sum();
    let w: Vec<f64> = w_raw.iter().map(|wi| wi / sw).collect();

    // --- box constraints ---------------------------------------------------
    let subst = |v: f64| {
        if v == f64::NEG_INFINITY {
            -ctl.big
        } else if v == f64::INFINITY {
            ctl.big
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
    // A bound of exactly zero pins its coefficient and disables `fdev` for the
    // whole fit (glmnet.R:510). See docs/PORTING.md; same rule as the Gaussian path.
    if cl_lo.iter().chain(cl_hi.iter()).any(|&v| v == 0.0) {
        ctl.fdev = 0.0;
    }

    // --- standardize X (weighted, no y-scaling, no sqrt(w) baked into X) ----
    let (xm, xs) = standardize_lognet(&mut x, &w, cfg.standardize, cfg.intercept, &ju);
    let mat = DenseGlm::new(x);

    run_path(mat, y, &w, sw, &xm, &xs, &vp, cl_lo, cl_hi, &ju, cfg, ctl)
}

/// Fit the two-class logistic path for a **sparse** design matrix (CSC).
///
/// `col_ptr`/`row_idx`/`values` are the CSC arrays; `y` is 0/1. The matrix is
/// never centered -- the standardization correction is folded into the solver
/// (see [`crate::glm_matrix`]).
pub fn lognet_sparse(
    n: usize,
    p: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
    y: &[f64],
    cfg: &FitConfig,
) -> Result<BinomialFit, FitError> {
    assert_eq!(col_ptr.len(), p + 1);
    assert_eq!(row_idx.len(), values.len());
    assert_eq!(y.len(), n);

    let mut ctl = cfg.control;

    let mut ju = chkvars_sparse(n, p, col_ptr, values);
    for &j in &cfg.exclude {
        if j < p {
            ju[j] = false;
        }
    }
    if !ju.iter().any(|&b| b) {
        return Err(FitError::AllExcluded);
    }

    let mut vp = cfg.penalty_factor.clone().unwrap_or_else(|| vec![1.0; p]);
    if vp.iter().cloned().fold(f64::NEG_INFINITY, f64::max) <= 0.0 {
        return Err(FitError::NonPositivePenalty);
    }
    vp.iter_mut().for_each(|v| *v = v.max(0.0));
    let vsum: f64 = vp.iter().sum();
    vp.iter_mut().for_each(|v| *v *= p as f64 / vsum);

    let w_raw = cfg.weights.clone().unwrap_or_else(|| vec![1.0; n]);
    let sw: f64 = w_raw.iter().sum();
    let w: Vec<f64> = w_raw.iter().map(|wi| wi / sw).collect();

    let subst = |v: f64| {
        if v == f64::NEG_INFINITY {
            -ctl.big
        } else if v == f64::INFINITY {
            ctl.big
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

    let (xm, xs) = standardize_lognet_sparse(
        col_ptr,
        row_idx,
        values,
        &w,
        p,
        cfg.standardize,
        cfg.intercept,
        &ju,
    );
    let mat = new_sparse_glm(
        n,
        p,
        col_ptr.to_vec(),
        row_idx.to_vec(),
        values.to_vec(),
        xm.clone(),
        xs.clone(),
    );

    run_path(mat, y, &w, sw, &xm, &xs, &vp, cl_lo, cl_hi, &ju, cfg, ctl)
}

/// Shared lambda path for the logistic solver, generic over dense/sparse.
/// `cl_lo`/`cl_hi` are substituted (not yet xs-scaled) box limits.
#[allow(clippy::too_many_arguments)]
fn run_path<M: GlmMatrix>(
    mat: M,
    y: &[f64],
    w: &[f64],
    sw: f64,
    xm: &[f64],
    xs: &[f64],
    vp: &[f64],
    mut cl_lo: Vec<f64>,
    mut cl_hi: Vec<f64>,
    ju: &[bool],
    cfg: &FitConfig,
    ctl: crate::control::Control,
) -> Result<BinomialFit, FitError> {
    let n = mat.nrows();
    let p = mat.ncols();

    if cfg.standardize {
        for j in 0..p {
            cl_lo[j] *= xs[j];
            cl_hi[j] *= xs[j];
        }
    }

    let (flmin, nlam, ulam): (f64, usize, Vec<f64>) = match &cfg.user_lambda {
        Some(l) => {
            let mut l = l.clone();
            l.sort_by(|a, b| b.partial_cmp(a).unwrap());
            (1.0, l.len(), l)
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
        mat,
        y,
        w,
        vp,
        &cl_lo,
        &cl_hi,
        ju,
        cfg.intercept,
        cfg.thresh,
        cfg.maxit,
        nx,
        ctl.pmin,
    )?;

    let _ = n;

    // --- path --------------------------------------------------------------
    let mut ca = vec![0.0; nx * nlam];
    let mut nin = vec![0usize; nlam];
    let mut almo = vec![0.0; nlam];
    let mut devo = vec![0.0; nlam];
    let mut a0o = vec![0.0; nlam];
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
            alm0 = 0.0;
            for j in 0..p {
                if !ju[j] || vp[j] <= 0.0 {
                    continue;
                }
                alm0 = alm0.max(pt.ga[j].abs() / vp[j]);
            }
            alm0 /= cfg.alpha.max(1e-3);
            alm = alm0 * alf;
        }
        lmda_curr = alm;
        let l2 = alm * omb;
        let l1 = alm * cfg.alpha;

        let cont = match pt.solve_at(cfg.alpha, alm, alm0, l1, l2) {
            Ok(cont) => cont,
            Err(PointErr::MaxIter) => {
                warning = Some(PathWarning::MaxIterReached { lambda_index: m });
                break;
            }
            Err(PointErr::MaxActive) => {
                warning = Some(PathWarning::MaxActiveReached { lambda_index: m });
                break;
            }
        };

        for (l, &k) in pt.ia.iter().enumerate() {
            ca[m * nx + l] = pt.b[k];
        }
        nin[m] = pt.ia.len();
        a0o[m] = pt.b0;
        almo[m] = alm;
        let devi = pt.dev2(ctl.pmin);
        devo[m] = (pt.dev1 - devi) / pt.dev0;
        lmu = m + 1;

        let me = pt.ia.iter().filter(|&&k| pt.b[k] != 0.0).count();
        let prev_dev = if m == 0 { 0.0 } else { devo[m - 1] };
        let dev_change = devo[m] - prev_dev;

        let stop = if lmu < mnl || flmin >= 1.0 {
            false
        } else if me > cfg.dfmax || dev_change < ctl.fdev || devo[m] > ctl.devmax {
            true
        } else {
            !cont
        };
        if stop {
            break;
        }
    }

    // --- unstandardize -----------------------------------------------------
    let mut lambda = vec![0.0; lmu];
    let mut a0 = vec![0.0; lmu];
    let mut beta = vec![0.0; p * lmu];

    for k in 0..lmu {
        lambda[k] = almo[k];
        let mut intercept = a0o[k];
        for l in 0..nin[k] {
            let j = pt.ia[l];
            let b = if cfg.standardize {
                ca[k * nx + l] / xs[j]
            } else {
                ca[k * nx + l]
            };
            beta[k * p + j] = b;
            if cfg.intercept {
                intercept -= b * xm[j];
            }
        }
        a0[k] = if cfg.intercept { intercept } else { 0.0 };
    }

    if cfg.user_lambda.is_none() {
        fix_lam(&mut lambda);
    }

    Ok(BinomialFit {
        lmu,
        lambda,
        a0,
        beta,
        dev_ratio: devo[..lmu].to_vec(),
        nulldev: pt.dev0 * 2.0 * sw,
        npasses: pt.nlp,
        warning,
    })
}

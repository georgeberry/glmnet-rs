//! Poisson (log-link count) elastic-net path.
//!
//! Transliterated from glmnetpp `ElnetDriver<poisson>` /
//! `ElnetPath<poisson,naive>` / `ElnetPointInternal<poisson,naive>`. It reuses
//! the same IRLS-over-WLS machinery as [`crate::binomial`]; only the link and
//! the deviance bookkeeping differ.
//!
//! The mean is `mu = exp(eta)`, and the Poisson variance equals the mean, so the
//! IRLS working weight is `w = q * mu` (`q` = observation weights) and the
//! working residual is `r = q*y - w = q*(y - mu)`. Compare binomial, where the
//! working weight is `q*p*(1-p)`. As in binomial, the working weight and column
//! variances are recomputed every outer IRLS step and state warm-starts across
//! lambdas.
//!
//! Poisson-specific quirks reproduced deliberately:
//! * The `fdev` early-stop threshold is **multiplied by 10** for Poisson
//!   (`sml *= 10` in glmnetpp's `initialize_path`), and the deviance test looks
//!   back `mnl-1` lambdas with a *relative* change `(dev(m)-dev(m-mnl+1))/dev(m)`
//!   rather than binomial's absolute one-step change.
//! * The linear predictor is clamped in magnitude to `log(f64::MAX * 0.1)` before
//!   exponentiating, so `exp(eta)` cannot overflow. There is no variance-collapse
//!   stop (binomial's `vmin`); the clamp plays that role.
//! * A negative response, or weights summing to <= 0, is a hard error.
//!
//! Scope: dense `X`, no offset. (`y` may be any non-negative real, e.g. rates.)

use crate::control::FitConfig;
use crate::error::{FitError, PathWarning};
use crate::kernel::soft_threshold;
use crate::matrix::{chkvars, wdot4, Dense, DesignMatrix};
use crate::standardize::standardize_lognet;

#[derive(Clone, Debug)]
pub struct PoissonFit {
    pub lmu: usize,
    pub lambda: Vec<f64>,
    pub a0: Vec<f64>,
    /// Coefficients, `p x lmu`, column-major, on the original scale.
    pub beta: Vec<f64>,
    /// Fraction of null deviance explained at each lambda. Length `lmu`.
    pub dev_ratio: Vec<f64>,
    /// Null deviance, `2 * sum(w) * dev0` on the original weight scale.
    pub nulldev: f64,
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

/// Persistent Poisson point solver; state carries across the whole path.
struct Point<'a> {
    x: &'a Dense,
    q: &'a [f64], // observation weights, normalized to sum 1
    t: Vec<f64>,  // q .* y  (the "target")

    b0: f64, // intercept (log scale)
    b: Vec<f64>,
    bs0: f64,
    bs: Vec<f64>,

    w: Vec<f64>,   // IRLS working weight, q .* exp(eta)
    r: Vec<f64>,   // working residual, t - w
    eta: Vec<f64>, // linear predictor, reused each IRLS step
    v0: f64,       // sum of w (new weight sum)

    xv: Vec<f64>, // weighted column variance sum_i w_i x_ij^2
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
    fmax: f64, // clamp on |eta|

    nlp: usize,
    dlx: f64,
    dev0: f64, // null deviance (internal scale)
    dv0: f64,  // null-model "intercept" deviance term
}

impl<'a> Point<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: &'a Dense,
        y: &'a [f64],
        q: &'a [f64],
        vp: &'a [f64],
        cl_lo: &'a [f64],
        cl_hi: &'a [f64],
        ju: &'a [bool],
        intr: bool,
        thresh: f64,
        maxit: usize,
        nx: usize,
    ) -> Self {
        let n = x.nrows();
        let p = x.ncols();

        let t: Vec<f64> = q.iter().zip(y).map(|(qi, yi)| qi * yi).collect();
        let yb: f64 = t.iter().sum();

        // Null model (offset is zero), closed form.
        let (b0, w, dv0, v0) = if intr {
            let b0 = yb.ln();
            let w: Vec<f64> = q.iter().map(|qi| yb * qi).collect();
            (b0, w, yb * (b0 - 1.0), yb)
        } else {
            (0.0, q.to_vec(), -1.0, 1.0)
        };
        let r: Vec<f64> = t.iter().zip(&w).map(|(ti, wi)| ti - wi).collect();

        // Null deviance: -yb + sum_{t_i>0} t_i log(y_i) - dv0.
        let mut dev0 = -yb;
        for (ti, yi) in t.iter().zip(y) {
            if *ti > 0.0 {
                dev0 += ti * yi.ln();
            }
        }
        dev0 -= dv0;

        // glmnetpp: fmax = log(DBL_MAX * 0.1).
        let fmax = (f64::MAX * 0.1).ln();

        let mut pt = Point {
            x,
            q,
            t,
            b0,
            b: vec![0.0; p],
            bs0: 0.0,
            bs: vec![0.0; p],
            w,
            r,
            eta: vec![b0; n],
            v0,
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
            nlp: 0,
            dlx: 0.0,
            dev0,
            dv0,
        };

        for j in 0..p {
            if pt.ju[j] {
                pt.ga[j] = pt.x.dot(j, &pt.r).abs();
            }
        }
        pt
    }

    #[inline]
    fn is_excluded(&self, k: usize) -> bool {
        !self.ix[k]
    }

    #[inline]
    fn compute_xv(&self, k: usize) -> f64 {
        let col = self.x.col(k);
        wdot4(&self.w, col, col)
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
    fn update_one(&mut self, k: usize, full: bool, l1: f64, l2: f64) -> Result<(), PointErr> {
        let gk = self.x.dot(k, &self.r); // <x_k, r>, r = t - w
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
        // r -= diff * (w .* x_k)
        for ((ri, wi), xi) in self.r.iter_mut().zip(&self.w).zip(self.x.col(k)) {
            *ri -= diff * wi * xi;
        }
        Ok(())
    }

    #[inline]
    fn update_intercept(&mut self) {
        if !self.intr {
            return;
        }
        let r_sum: f64 = self.r.iter().sum();
        let d = r_sum / self.v0;
        if d != 0.0 {
            self.b0 += d;
            self.dlx = self.dlx.max(self.v0 * d * d);
            for (ri, wi) in self.r.iter_mut().zip(&self.w) {
                *ri -= d * wi;
            }
        }
    }

    fn wls_pass(&mut self, full: bool, l1: f64, l2: f64) -> Result<bool, PointErr> {
        self.nlp += 1;
        self.dlx = 0.0;

        if full {
            for k in 0..self.x.ncols() {
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

    fn setup_wls(&mut self) {
        self.bs0 = self.b0;
        for &k in &self.ia {
            self.bs[k] = self.b[k];
        }
        for k in 0..self.x.ncols() {
            if self.ix[k] {
                self.xv[k] = self.compute_xv(k);
            }
        }
    }

    /// Recompute the linear predictor, working weight and residual after a WLS
    /// solve. Poisson has no variance-collapse stop; the `eta` clamp guards
    /// against overflow instead.
    fn update_irls_recompute(&mut self) {
        let mut eta = std::mem::take(&mut self.eta);
        eta.iter_mut().for_each(|e| *e = self.b0);
        for &k in &self.ia {
            let bk = self.b[k];
            for (ei, xi) in eta.iter_mut().zip(self.x.col(k)) {
                *ei += bk * xi;
            }
        }
        // w = q * exp(clamp(eta)); the clamp is on |eta|, preserving sign.
        self.v0 = 0.0;
        for ((wi, &qi), &f) in self.w.iter_mut().zip(self.q.iter()).zip(eta.iter()) {
            let clamped = f.abs().min(self.fmax).copysign(f);
            *wi = qi * clamped.exp();
            self.v0 += *wi;
        }
        for ((ri, &ti), &wi) in self.r.iter_mut().zip(self.t.iter()).zip(self.w.iter()) {
            *ri = ti - wi;
        }
        self.eta = eta;
    }

    fn has_converged_irls(&self) -> bool {
        let d0 = self.b0 - self.bs0;
        if self.v0 * d0 * d0 > self.thr_scaled {
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

    fn kkt_complete(&mut self, l1: f64) -> bool {
        for k in 0..self.x.ncols() {
            if !self.is_excluded(k) || !self.ju[k] {
                continue;
            }
            self.ga[k] = self.x.dot(k, &self.r).abs();
        }
        let mut ok = true;
        for k in 0..self.x.ncols() {
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

    fn initialize_strong_set(&mut self, alpha: f64, alm: f64, alm0: f64) {
        let tlam = alpha * (2.0 * alm - alm0);
        for k in 0..self.x.ncols() {
            if self.ix[k] || !self.ju[k] {
                continue;
            }
            if self.ga[k] > tlam * self.vp[k] {
                self.ix[k] = true;
            }
        }
    }

    fn solve_at(
        &mut self,
        alpha: f64,
        alm: f64,
        alm0: f64,
        l1: f64,
        l2: f64,
    ) -> Result<(), PointErr> {
        self.initialize_strong_set(alpha, alm, alm0);
        loop {
            if self.nlp > self.maxit {
                return Err(PointErr::MaxIter);
            }
            self.setup_wls();
            self.wls(l1, l2)?;
            self.update_irls_recompute();
            if self.has_converged_irls() && self.kkt_complete(l1) {
                return Ok(());
            }
        }
    }

    /// Fraction of null deviance explained: `(t.eta - v0 - dv0) / dev0`.
    fn deviance(&self) -> f64 {
        let teta: f64 = self.t.iter().zip(&self.eta).map(|(ti, ei)| ti * ei).sum();
        (teta - self.v0 - self.dv0) / self.dev0
    }
}

/// Fit the Poisson (log-link) elastic-net path.
///
/// `x_col_major` is `n * p`, column-major. `y` must be non-negative.
pub fn fishnet(
    x_col_major: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    cfg: &FitConfig,
) -> Result<PoissonFit, FitError> {
    assert_eq!(x_col_major.len(), n * p);
    assert_eq!(y.len(), n);

    if y.iter().any(|&yi| yi < 0.0) {
        return Err(FitError::NegativeResponse);
    }

    let mut ctl = cfg.control;
    // Poisson uses a 10x larger fdev threshold (glmnetpp initialize_path: sml *= 10).
    ctl.fdev *= 10.0;

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

    // --- weights: clamp negatives to 0, normalize to sum 1 -----------------
    let mut w_raw = cfg.weights.clone().unwrap_or_else(|| vec![1.0; n]);
    w_raw.iter_mut().for_each(|wi| *wi = wi.max(0.0));
    let sw: f64 = w_raw.iter().sum();
    if sw <= 0.0 {
        return Err(FitError::NonPositiveWeightSum);
    }
    let q: Vec<f64> = w_raw.iter().map(|wi| wi / sw).collect();

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
    let mut cl_lo: Vec<f64> = cfg
        .lower_limits
        .clone()
        .unwrap_or_else(|| vec![f64::NEG_INFINITY; p])
        .into_iter()
        .map(subst)
        .collect();
    let mut cl_hi: Vec<f64> = cfg
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

    // --- standardize X (weighted, no y-scaling) ----------------------------
    let (xm, xs) = standardize_lognet(&mut x, &q, cfg.standardize, cfg.intercept, &ju);
    if cfg.standardize {
        for j in 0..p {
            cl_lo[j] *= xs[j];
            cl_hi[j] *= xs[j];
        }
    }

    // --- lambda grid setup -------------------------------------------------
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
        &x,
        y,
        &q,
        &vp,
        &cl_lo,
        &cl_hi,
        &ju,
        cfg.intercept,
        cfg.thresh,
        cfg.maxit,
        nx,
    );

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

        match pt.solve_at(cfg.alpha, alm, alm0, l1, l2) {
            Ok(()) => {}
            Err(PointErr::MaxIter) => {
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
            ca[m * nx + l] = pt.b[k];
        }
        nin[m] = pt.ia.len();
        a0o[m] = pt.b0;
        almo[m] = alm;
        devo[m] = pt.deviance();
        lmu = m + 1;

        let me = pt.ia.iter().filter(|&&k| pt.b[k] != 0.0).count();
        // Poisson's early stop: relative change over the last mnl-1 lambdas.
        let prev_dev = if m + 1 >= mnl { devo[m + 1 - mnl] } else { 0.0 };
        let dev_change = (devo[m] - prev_dev) / devo[m];

        let stop = if lmu < mnl || flmin >= 1.0 {
            false
        } else {
            me > cfg.dfmax || dev_change < ctl.fdev || devo[m] > ctl.devmax
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

    Ok(PoissonFit {
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

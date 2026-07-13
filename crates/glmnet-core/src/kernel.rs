//! Coordinate-descent primitives shared across families.
//!
//! The elastic-net coordinate update is identical for every glmnet family; only
//! the gradient `gk` and the column variance `xv` fed into it differ (Gaussian
//! uses the raw residual, the GLM families use the IRLS working residual). This
//! is glmnetpp's `ElnetPointInternalStaticBase::update_beta`.

/// The soft-thresholded, box-constrained elastic-net coordinate update.
///
/// Minimizes the 1-D penalized quadratic in `b_k` given the current gradient
/// `gk = <x_k, r>`, column variance `xv`, and returns the new `b_k`.
///
/// `l1 = lambda*alpha` and `l2 = lambda*(1-alpha)`; `penalty` is the per-feature
/// factor `vp[k]`; `cl_lo`/`cl_hi` are the (already rescaled) box limits.
#[inline]
#[allow(clippy::too_many_arguments)] // a faithful 1-D elastic-net update simply needs them all
pub(crate) fn soft_threshold(
    a_old: f64,
    gk: f64,
    xv: f64,
    penalty: f64,
    cl_lo: f64,
    cl_hi: f64,
    l1: f64,
    l2: f64,
) -> f64 {
    let u = gk + a_old * xv;
    let v = u.abs() - penalty * l1;
    if v > 0.0 {
        let cand = v.copysign(u) / (xv + penalty * l2);
        // max(lo, min(hi, cand)) -- deliberately not f64::clamp, which panics
        // when lo > hi. glmnet lets lo win in that case.
        cand.min(cl_hi).max(cl_lo)
    } else {
        0.0
    }
}

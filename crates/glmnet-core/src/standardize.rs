//! Weighted standardization, a transliteration of glmnetpp `Standardize1`.
//!
//! Two conventions here are easy to get wrong and both are load-bearing:
//!
//! 1. Weights are normalized to sum to 1, and `X`/`y` are pre-multiplied by
//!    `sqrt(w)`. Every later inner product is therefore implicitly weighted,
//!    and the solver never sees `w` again.
//!
//! 2. `y` is scaled to *unit norm* (`ys = ||y||; y /= ys`). This is why `rsq`
//!    accumulates on [0, 1] and why `devmax = 0.999` is a meaningful stopping
//!    rule. Coefficients and lambdas come back on the `ys` scale and must be
//!    multiplied by `ys` on the way out.

use crate::matrix::{Dense, DesignMatrix};

#[derive(Clone, Debug)]
pub struct Standardization {
    /// Weighted column means (zero when `intercept == false`).
    pub xm: Vec<f64>,
    /// Column scales (one when `standardize == false`).
    pub xs: Vec<f64>,
    /// Weighted mean of `y` (zero when `intercept == false`).
    pub ym: f64,
    /// Norm of the centered, weighted `y`.
    pub ys: f64,
    /// Weighted column variances, used as the CD denominator.
    pub xv: Vec<f64>,
}

/// Standardizes `x`, `y` and `w` in place. `w` is normalized to sum to 1.
pub fn standardize_naive(
    x: &mut Dense,
    y: &mut [f64],
    w: &mut [f64],
    isd: bool,
    intr: bool,
    ju: &[bool],
) -> Standardization {
    let n = x.nrows();
    let p = x.ncols();

    let wsum: f64 = w.iter().sum();
    for wi in w.iter_mut() {
        *wi /= wsum;
    }
    let v: Vec<f64> = w.iter().map(|wi| wi.sqrt()).collect();

    let mut xm = vec![0.0; p];
    let mut xs = vec![0.0; p];
    let mut xv = vec![0.0; p];
    let ym;
    let ys;

    if !intr {
        ym = 0.0;
        for i in 0..n {
            y[i] *= v[i];
        }
        // "trevor changed 3/24/2020": y is normalized even without an intercept.
        ys = norm(y);
        for yi in y.iter_mut() {
            *yi /= ys;
        }

        for j in 0..p {
            if !ju[j] {
                continue;
            }
            xm[j] = 0.0;
            {
                let col = x.col_mut(j);
                for i in 0..n {
                    col[i] *= v[i];
                }
            }
            xv[j] = sq_norm(x.col(j));
            if isd {
                // Column j has already been multiplied by sqrt(w), so <x_j, v> is
                // the weighted mean; subtracting its square gives the variance
                // about that mean while the column itself stays uncentered.
                let xbq = dot(x.col(j), &v).powi(2);
                let vc = xv[j] - xbq;
                xs[j] = vc.sqrt();
                let s = xs[j];
                for e in x.col_mut(j) {
                    *e /= s;
                }
                xv[j] = 1.0 + xbq / vc;
            } else {
                xs[j] = 1.0;
            }
        }
    } else {
        for j in 0..p {
            if !ju[j] {
                continue;
            }
            xm[j] = dot(x.col(j), w);
            let m = xm[j];
            {
                let col = x.col_mut(j);
                for i in 0..n {
                    col[i] = v[i] * (col[i] - m);
                }
            }
            xv[j] = sq_norm(x.col(j));
            if isd {
                xs[j] = xv[j].sqrt();
            }
        }
        if !isd {
            // Note: set for *every* column, including excluded ones.
            xs.iter_mut().for_each(|e| *e = 1.0);
        } else {
            for j in 0..p {
                if !ju[j] {
                    continue;
                }
                let s = xs[j];
                for e in x.col_mut(j) {
                    *e /= s;
                }
                xv[j] = 1.0;
            }
        }

        ym = dot(y, w);
        for i in 0..n {
            y[i] = v[i] * (y[i] - ym);
        }
        ys = norm(y);
        for yi in y.iter_mut() {
            *yi /= ys;
        }
    }

    Standardization { xm, xs, ym, ys, xv }
}

/// Sparse Gaussian standardization (glmnetpp `SpStandardize1`).
///
/// Unlike the dense path, `X` is **not** modified (centering would destroy
/// sparsity) and `y` is centered/scaled *without* the `sqrt(w)` premultiply --
/// the weights stay separate and are folded into the solver's column ops. The
/// returned `xm`/`xs`/`xv` are what the sparse solver needs; `y` is overwritten
/// in place with the (unweighted) initial residual `(y - ym) / ys`.
#[allow(clippy::too_many_arguments)]
pub fn standardize_naive_sparse(
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
    y: &mut [f64],
    w: &mut [f64],
    p: usize,
    isd: bool,
    intr: bool,
    ju: &[bool],
) -> Standardization {
    let wsum: f64 = w.iter().sum();
    for wi in w.iter_mut() {
        *wi /= wsum;
    }

    let mut xm = vec![0.0; p];
    let mut xs = vec![1.0; p];
    let mut xv = vec![0.0; p];
    let ym;
    let ys;

    // Weighted sums over the stored nonzeros of column j: (sum w_i x, sum w_i x^2).
    let col_moments = |j: usize| -> (f64, f64) {
        let (b, e) = (col_ptr[j], col_ptr[j + 1]);
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for idx in b..e {
            let wi = w[row_idx[idx]];
            let x = values[idx];
            s1 += wi * x;
            s2 += wi * x * x;
        }
        (s1, s2)
    };

    if !intr {
        ym = 0.0;
        // ys = sqrt(sum w_i y_i^2); note the weight (no sqrt(w) baked into y).
        ys = w
            .iter()
            .zip(y.iter())
            .map(|(wi, yi)| wi * yi * yi)
            .sum::<f64>()
            .sqrt();
        for yi in y.iter_mut() {
            *yi /= ys;
        }
        for j in 0..p {
            if !ju[j] {
                continue;
            }
            xm[j] = 0.0;
            let (mean, ex2) = col_moments(j);
            xv[j] = ex2;
            if isd {
                let xbq = mean * mean;
                let vc = xv[j] - xbq;
                xs[j] = vc.sqrt();
                xv[j] = 1.0 + xbq / vc;
            } else {
                xs[j] = 1.0;
            }
        }
    } else {
        for j in 0..p {
            if !ju[j] {
                continue;
            }
            let (mean, ex2) = col_moments(j);
            xm[j] = mean;
            xv[j] = ex2 - mean * mean;
            if isd {
                xs[j] = xv[j].sqrt();
            }
        }
        if isd {
            xv.iter_mut().for_each(|v| *v = 1.0);
        }
        ym = w.iter().zip(y.iter()).map(|(wi, yi)| wi * yi).sum();
        for yi in y.iter_mut() {
            *yi -= ym;
        }
        ys = w
            .iter()
            .zip(y.iter())
            .map(|(wi, yi)| wi * yi * yi)
            .sum::<f64>()
            .sqrt();
        for yi in y.iter_mut() {
            *yi /= ys;
        }
    }

    Standardization { xm, xs, ym, ys, xv }
}

/// Sparse GLM standardization (glmnetpp `SpLStandardize2`): compute the weighted
/// column means `xm` and scales `xs` from a CSC matrix without touching `X` and
/// without scaling `y`. `w` must already sum to 1.
#[allow(clippy::too_many_arguments)]
pub fn standardize_lognet_sparse(
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
    w: &[f64],
    p: usize,
    isd: bool,
    intr: bool,
    ju: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let mut xm = vec![0.0; p];
    let mut xs = vec![1.0; p];

    // (sum w_i x_ij, sum w_i x_ij^2) over column j's stored nonzeros.
    let moments = |j: usize| -> (f64, f64) {
        let (b, e) = (col_ptr[j], col_ptr[j + 1]);
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for idx in b..e {
            let wi = w[row_idx[idx]];
            let x = values[idx];
            s1 += wi * x;
            s2 += wi * x * x;
        }
        (s1, s2)
    };

    for j in 0..p {
        if !ju[j] {
            continue;
        }
        let (mean, ex2) = moments(j);
        if intr {
            xm[j] = mean;
            if isd {
                xs[j] = (ex2 - mean * mean).sqrt();
            }
        } else {
            xm[j] = 0.0;
            if isd {
                xs[j] = (ex2 - mean * mean).sqrt();
            }
        }
    }

    (xm, xs)
}

/// Weighted standardization for the GLM families (glmnetpp `LStandardize1`).
///
/// Unlike the Gaussian path, `X` is only **centered** (and optionally scaled),
/// never pre-multiplied by `sqrt(w)`, and `y` is left untouched. The observation
/// weights instead enter the solver through the IRLS working weights. `w` must
/// already be normalized to sum to 1.
///
/// Returns `(xm, xs)`: weighted means and scales. Column variances are not
/// returned because the GLM solvers recompute them from the IRLS weights every
/// outer iteration.
pub fn standardize_lognet(
    x: &mut Dense,
    w: &[f64],
    isd: bool,
    intr: bool,
    ju: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let p = x.ncols();
    let mut xm = vec![0.0; p];
    let mut xs = vec![1.0; p];

    for j in 0..p {
        if !ju[j] {
            continue;
        }
        if intr {
            xm[j] = dot(x.col(j), w);
            let m = xm[j];
            for e in x.col_mut(j) {
                *e -= m;
            }
            if isd {
                let var = w
                    .iter()
                    .zip(x.col(j))
                    .map(|(wi, xi)| wi * xi * xi)
                    .sum::<f64>();
                xs[j] = var.sqrt();
                let s = xs[j];
                for e in x.col_mut(j) {
                    *e /= s;
                }
            }
        } else {
            xm[j] = 0.0;
            if isd {
                let mean = dot(x.col(j), w);
                let ex2 = w
                    .iter()
                    .zip(x.col(j))
                    .map(|(wi, xi)| wi * xi * xi)
                    .sum::<f64>();
                xs[j] = (ex2 - mean * mean).sqrt();
                let s = xs[j];
                for e in x.col_mut(j) {
                    *e /= s;
                }
            }
        }
    }

    (xm, xs)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[inline]
fn sq_norm(a: &[f64]) -> f64 {
    dot(a, a)
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    sq_norm(a).sqrt()
}

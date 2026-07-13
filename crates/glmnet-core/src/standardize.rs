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

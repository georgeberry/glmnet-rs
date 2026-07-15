//! Column operations the GLM (binomial/poisson) coordinate descent needs,
//! abstracted over dense vs sparse storage.
//!
//! This is the GLM analogue of [`crate::matrix::DesignMatrix`], but harder: the
//! IRLS working weights change every outer iteration, and the residual is *not*
//! weighted-mean-zero, so the sparse backend carries two correction scalars --
//! `o` (the gradient mean-shift) and `svr` (the true weighted-residual sum) --
//! plus a per-column weighted-mean vector `xm` that is refreshed as the weights
//! change. The dense backend needs none of that; its impl is a thin passthrough,
//! and the existing dense parity fixtures guarantee the refactor is faithful.
//!
//! All coefficients are in *standardized* coordinates. Column `j`'s standardized
//! values are `z_ij = (x_ij - xb_j) / xs_j`; a dense matrix is standardized in
//! place so `z` is stored directly, while a sparse matrix stores raw `x` and
//! folds `xb`/`xs` into each operation.

use crate::matrix::{wdot4, Dense, DesignMatrix};

/// The GLM design-matrix interface. The implementor owns any correction state.
pub trait GlmMatrix {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    /// Gradient w.r.t. the standardized coefficient of column `k`:
    /// `<z_k, r>` with the sparse mean-shift correction folded in. `v` is the
    /// current IRLS working-weight vector (used only by the sparse backend).
    fn grad(&self, k: usize, r: &[f64], v: &[f64]) -> f64;

    /// Apply a change `diff` in the standardized coefficient of column `k`:
    /// subtract its weighted contribution from the residual `r` and update the
    /// correction state. `xmz` is the working-weight sum `sum_i v_i`.
    fn update_resid(&mut self, k: usize, diff: f64, r: &mut [f64], v: &[f64], xmz: f64);

    /// Weighted column variance `sum_i v_i z_ik^2`. May refresh internal `xm_k`.
    fn xv(&mut self, k: usize, v: &[f64], xmz: f64) -> f64;

    /// Hook when column `k` joins the active set (sparse records `xm_k`).
    fn on_activate(&mut self, k: usize, v: &[f64]);

    /// The (true) weighted-residual sum used by the intercept update: `sum_i r_i`
    /// for dense, the tracked `svr` for sparse.
    fn resid_sum(&self, r: &[f64]) -> f64;

    /// After an intercept change `d`: update the residual (`r -= d * v`) and the
    /// correction state.
    fn update_intercept(&mut self, d: f64, r: &mut [f64], v: &[f64], xmz: f64);

    /// Reset the shift accumulators from a freshly recomputed residual `r`
    /// (sparse: `svr = sum r`, `o = 0`). No-op for dense.
    fn reset_shifts(&mut self, r: &[f64]);

    /// Accumulate the linear predictor into `eta`: `eta_i = b0 + sum_k beta_k z_ik`
    /// over the active columns.
    fn accumulate_eta(&self, active: &[usize], beta: &[f64], b0: f64, eta: &mut [f64]);
}

/// Dense GLM matrix: `X` standardized in place, no correction state.
pub(crate) struct DenseGlm {
    x: Dense,
}

impl DenseGlm {
    pub fn new(x: Dense) -> Self {
        DenseGlm { x }
    }
}

impl GlmMatrix for DenseGlm {
    #[inline]
    fn nrows(&self) -> usize {
        self.x.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.x.ncols()
    }

    #[inline]
    fn grad(&self, k: usize, r: &[f64], _v: &[f64]) -> f64 {
        self.x.dot(k, r)
    }

    #[inline]
    fn update_resid(&mut self, k: usize, diff: f64, r: &mut [f64], v: &[f64], _xmz: f64) {
        for ((ri, vi), xi) in r.iter_mut().zip(v).zip(self.x.col(k)) {
            *ri -= diff * vi * xi;
        }
    }

    #[inline]
    fn xv(&mut self, k: usize, v: &[f64], _xmz: f64) -> f64 {
        let col = self.x.col(k);
        wdot4(v, col, col)
    }

    #[inline]
    fn on_activate(&mut self, _k: usize, _v: &[f64]) {}

    #[inline]
    fn resid_sum(&self, r: &[f64]) -> f64 {
        r.iter().sum()
    }

    #[inline]
    fn update_intercept(&mut self, d: f64, r: &mut [f64], v: &[f64], _xmz: f64) {
        for (ri, vi) in r.iter_mut().zip(v) {
            *ri -= d * vi;
        }
    }

    #[inline]
    fn reset_shifts(&mut self, _r: &[f64]) {}

    #[inline]
    fn accumulate_eta(&self, active: &[usize], beta: &[f64], b0: f64, eta: &mut [f64]) {
        eta.iter_mut().for_each(|e| *e = b0);
        for &k in active {
            let bk = beta[k];
            for (ei, xi) in eta.iter_mut().zip(self.x.col(k)) {
                *ei += bk * xi;
            }
        }
    }
}

/// Nonzeros of a CSC column as `(row, value)`.
pub(crate) struct SparseGlm {
    n: usize,
    p: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
    xb: Vec<f64>, // column means
    xs: Vec<f64>, // column scales
    xm: Vec<f64>, // current weighted column means (refreshed as v changes)
    o: f64,       // gradient mean-shift
    svr: f64,     // true weighted-residual sum
}

// NB: kept `pub(crate)` via re-export below; the type itself need not be public.
impl SparseGlm {
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        p: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<f64>,
        xb: Vec<f64>,
        xs: Vec<f64>,
    ) -> Self {
        SparseGlm {
            n,
            p,
            col_ptr,
            row_idx,
            values,
            xb,
            xs,
            xm: vec![0.0; p],
            o: 0.0,
            svr: 0.0,
        }
    }

    #[inline]
    fn col(&self, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.col_ptr[j]..self.col_ptr[j + 1];
        self.row_idx[r.clone()]
            .iter()
            .copied()
            .zip(self.values[r].iter().copied())
    }

    /// `sum_i v_i x_ij` over the stored nonzeros.
    #[inline]
    fn col_dot_v(&self, j: usize, v: &[f64]) -> f64 {
        let mut s = 0.0;
        for (i, x) in self.col(j) {
            s += v[i] * x;
        }
        s
    }
}

impl GlmMatrix for SparseGlm {
    #[inline]
    fn nrows(&self) -> usize {
        self.n
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.p
    }

    // grad_k = [ sum_nz x_ik (r_i + v_i o) - svr * xb_k ] / xs_k.
    #[inline]
    fn grad(&self, k: usize, r: &[f64], v: &[f64]) -> f64 {
        let mut gk = 0.0;
        for (i, x) in self.col(k) {
            gk += x * (r[i] + v[i] * self.o);
        }
        (gk - self.svr * self.xb[k]) / self.xs[k]
    }

    #[inline]
    fn update_resid(&mut self, k: usize, diff: f64, r: &mut [f64], v: &[f64], xmz: f64) {
        let d = diff / self.xs[k];
        for (i, x) in self.col(k) {
            r[i] -= d * v[i] * x;
        }
        self.o += d * self.xb[k];
        self.svr -= d * (self.xm[k] - self.xb[k] * xmz);
    }

    // xv_j = ( sum_nz x_ij^2 v_i - 2 xb_j xm_j + xmz xb_j^2 ) / xs_j^2,
    // refreshing xm_j = sum_nz x_ij v_i first (glmnetpp update_with_new_weights).
    #[inline]
    fn xv(&mut self, k: usize, v: &[f64], xmz: f64) -> f64 {
        let mut x2v = 0.0;
        let mut xv_ = 0.0;
        for (i, x) in self.col(k) {
            x2v += v[i] * x * x;
            xv_ += v[i] * x;
        }
        self.xm[k] = xv_;
        (x2v - 2.0 * self.xb[k] * xv_ + xmz * self.xb[k] * self.xb[k]) / (self.xs[k] * self.xs[k])
    }

    #[inline]
    fn on_activate(&mut self, k: usize, v: &[f64]) {
        self.xm[k] = self.col_dot_v(k, v);
    }

    #[inline]
    fn resid_sum(&self, _r: &[f64]) -> f64 {
        self.svr
    }

    #[inline]
    fn update_intercept(&mut self, d: f64, r: &mut [f64], v: &[f64], xmz: f64) {
        for (ri, vi) in r.iter_mut().zip(v) {
            *ri -= d * vi;
        }
        self.svr -= d * xmz;
    }

    #[inline]
    fn reset_shifts(&mut self, r: &[f64]) {
        self.svr = r.iter().sum();
        self.o = 0.0;
    }

    // eta_i = b0 + sum_k (beta_k / xs_k)(x_ik - xb_k)
    //       = (b0 - sum_k (beta_k/xs_k) xb_k) + sum_nz (beta_k/xs_k) x_ik.
    #[inline]
    fn accumulate_eta(&self, active: &[usize], beta: &[f64], b0: f64, eta: &mut [f64]) {
        let mut shift = b0;
        for &k in active {
            shift -= (beta[k] / self.xs[k]) * self.xb[k];
        }
        eta.iter_mut().for_each(|e| *e = shift);
        for &k in active {
            let s = beta[k] / self.xs[k];
            for (i, x) in self.col(k) {
                eta[i] += s * x;
            }
        }
    }
}

/// Constructor for the sparse GLM matrix (the type itself stays private).
#[allow(clippy::too_many_arguments)]
pub(crate) fn new_sparse_glm(
    n: usize,
    p: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
    xb: Vec<f64>,
    xs: Vec<f64>,
) -> impl GlmMatrix {
    SparseGlm::new(n, p, col_ptr, row_idx, values, xb, xs)
}

//! Column-oriented access to the design matrix.
//!
//! Everything the coordinate-descent kernels need from `X` is expressed as two
//! operations on a single column: an inner product against the residual, and an
//! axpy into the residual. Confining `X` to this interface is what will let a
//! sparse backend drop in without touching the kernels.
//!
//! Why that matters: a dense `X` is standardized *in place* (centered and scaled
//! before the solver ever runs), so both operations are plain BLAS-1. A sparse
//! `X` cannot be centered without destroying its sparsity, so upstream glmnet
//! keeps `X` untouched and folds the `xm`/`xs` correction into every gradient
//! and residual update. That is the entire reason glmnetpp carries a parallel
//! `sp_*` implementation of all twelve point-solvers. Pushing the correction
//! behind this trait means the algorithm is written once.

/// A design matrix presented in *standardized coordinates* to the solver.
pub trait DesignMatrix {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    /// `<x_j, r>` in standardized coordinates.
    fn dot(&self, j: usize, r: &[f64]) -> f64;

    /// `r += alpha * x_j` in standardized coordinates.
    fn axpy(&self, j: usize, alpha: f64, r: &mut [f64]);
}

/// Dense, column-major, standardized in place.
#[derive(Clone, Debug)]
pub struct Dense {
    data: Vec<f64>,
    n: usize,
    p: usize,
}

impl Dense {
    /// `data` must be column-major and of length `n * p`.
    pub fn from_col_major(data: Vec<f64>, n: usize, p: usize) -> Self {
        assert_eq!(
            data.len(),
            n * p,
            "expected {} x {} = {} entries",
            n,
            p,
            n * p
        );
        Dense { data, n, p }
    }

    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.n..(j + 1) * self.n]
    }

    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.n..(j + 1) * self.n]
    }
}

impl DesignMatrix for Dense {
    #[inline]
    fn nrows(&self) -> usize {
        self.n
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.p
    }

    #[inline]
    fn dot(&self, j: usize, r: &[f64]) -> f64 {
        let col = self.col(j);
        debug_assert_eq!(col.len(), r.len());
        // Left as a plain scalar loop: it matches the summation order Eigen uses
        // for un-vectorized small columns, and parity comes before speed. See
        // docs/PORTING.md before replacing this with a chunked/SIMD reduction.
        let mut acc = 0.0;
        for i in 0..col.len() {
            acc += col[i] * r[i];
        }
        acc
    }

    #[inline]
    fn axpy(&self, j: usize, alpha: f64, r: &mut [f64]) {
        let col = self.col(j);
        debug_assert_eq!(col.len(), r.len());
        for i in 0..col.len() {
            r[i] += alpha * col[i];
        }
    }
}

/// `ju[j]` is true when column `j` is non-constant and therefore usable.
/// Mirrors glmnetpp `Chkvars::eval`: compare every entry against the first.
pub fn chkvars(x: &Dense) -> Vec<bool> {
    (0..x.ncols())
        .map(|j| {
            let col = x.col(j);
            let t = col[0];
            col[1..].iter().any(|&v| v != t)
        })
        .collect()
}

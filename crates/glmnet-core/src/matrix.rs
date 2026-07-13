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

/// Dot product with four partial accumulators.
///
/// A strict left-fold `acc += a[i]*b[i]` has a loop-carried dependency on `acc`,
/// which Rust will not reassociate (float `+` is not associative), so it never
/// vectorizes. Four independent accumulators break that dependency and let LLVM
/// emit SIMD, which roughly halves whole-path solve time on the larger problems.
///
/// The reassociation changes the summation order, so results differ from the
/// strict fold at the ~1e-15 level. This is not a parity regression: glmnetpp is
/// built on Eigen, whose dot product is itself SIMD-reassociated, so the strict
/// fold never reproduced glmnet's exact summation order either. What we hold to
/// is the parity *test* -- exact `npasses` and coefficients to 1e-12 against R --
/// which this passes on all fixtures. See docs/PORTING.md.
#[inline]
pub(crate) fn dot4(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let (mut a0, mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0, 0.0);
    let chunks = n / 4;
    for c in 0..chunks {
        let i = c * 4;
        a0 += a[i] * b[i];
        a1 += a[i + 1] * b[i + 1];
        a2 += a[i + 2] * b[i + 2];
        a3 += a[i + 3] * b[i + 3];
    }
    let mut acc = (a0 + a1) + (a2 + a3);
    for i in (chunks * 4)..n {
        acc += a[i] * b[i];
    }
    acc
}

/// Weighted dot `sum_i w_i a_i b_i` with four accumulators. Same rationale as
/// [`dot4`]; used by the GLM families for `sum_i v_i x_ij^2` and gradients.
#[inline]
pub(crate) fn wdot4(w: &[f64], a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), w.len());
    let n = a.len();
    let (mut a0, mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0, 0.0);
    let chunks = n / 4;
    for c in 0..chunks {
        let i = c * 4;
        a0 += w[i] * a[i] * b[i];
        a1 += w[i + 1] * a[i + 1] * b[i + 1];
        a2 += w[i + 2] * a[i + 2] * b[i + 2];
        a3 += w[i + 3] * a[i + 3] * b[i + 3];
    }
    let mut acc = (a0 + a1) + (a2 + a3);
    for i in (chunks * 4)..n {
        acc += w[i] * a[i] * b[i];
    }
    acc
}

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
        dot4(col, r)
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

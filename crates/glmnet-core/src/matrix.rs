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

/// A design matrix presented in *standardized coordinates* to the coordinate
/// descent, abstracting over dense and sparse storage.
///
/// The two operations the Gaussian solver needs from `X` are the weighted
/// gradient of a column and the residual update after a coefficient change.
/// For a **dense** matrix these are trivial: `X` is standardized in place before
/// the solve, so the gradient is a plain dot with the (weighted) residual and
/// the update is an axpy; no correction state is needed (`Corr = ()`).
///
/// A **sparse** matrix cannot be centered without destroying its sparsity, so it
/// is left in raw CSC form and the `xm`/`xs`/`w` standardization correction is
/// folded into each operation. The residual is kept unweighted and a scalar
/// mean-shift `o` (the `Corr` state) accumulates; the true weighted residual is
/// `r + o`. This keeps every column op O(nnz in that column). See glmnetpp's
/// parallel `sp_*` solvers -- here it is one more trait impl.
pub trait DesignMatrix {
    /// Per-solver correction state threaded through `grad`/`update_resid`.
    /// `()` for dense, the mean-shift scalar for sparse.
    type Corr: Copy + Default;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    /// Weighted gradient with respect to the *standardized* coefficient of
    /// column `j`, given the current residual `r` and correction state.
    fn grad(&self, j: usize, r: &[f64], corr: Self::Corr) -> f64;

    /// Apply a change `beta_diff` in the standardized coefficient of column `j`:
    /// subtract its contribution from the residual (and update the correction).
    fn update_resid(&self, j: usize, beta_diff: f64, r: &mut [f64], corr: &mut Self::Corr);
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

    /// `<x_j, r>`. Inherent (not just via the trait) because the GLM solvers
    /// call it directly on a concrete `Dense`.
    #[inline]
    pub fn dot(&self, j: usize, r: &[f64]) -> f64 {
        let col = self.col(j);
        debug_assert_eq!(col.len(), r.len());
        dot4(col, r)
    }
}

impl DesignMatrix for Dense {
    type Corr = ();

    #[inline]
    fn nrows(&self) -> usize {
        self.n
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.p
    }

    // X is standardized in place, so the gradient is a plain (weighted) dot and
    // needs no correction.
    #[inline]
    fn grad(&self, j: usize, r: &[f64], _corr: ()) -> f64 {
        self.dot(j, r)
    }

    #[inline]
    fn update_resid(&self, j: usize, beta_diff: f64, r: &mut [f64], _corr: &mut ()) {
        let col = self.col(j);
        debug_assert_eq!(col.len(), r.len());
        for (ri, xi) in r.iter_mut().zip(col) {
            *ri -= beta_diff * xi;
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

/// Sparse design matrix in compressed-sparse-column (CSC) form, carrying the
/// standardization it is presented under (`w`, `xm`, `xs`).
///
/// The raw values are never centered or scaled; instead every column op folds in
/// the correction, keeping it O(nnz in that column). See [`DesignMatrix`].
#[derive(Clone, Debug)]
pub struct Sparse {
    n: usize,
    p: usize,
    /// Column pointers, length `p + 1` (CSC `indptr`).
    col_ptr: Vec<usize>,
    /// Row indices of the nonzeros, length `nnz`.
    row_idx: Vec<usize>,
    /// Nonzero values, length `nnz`.
    values: Vec<f64>,
    /// Observation weights (normalized to sum 1).
    w: Vec<f64>,
    /// Column means (weighted).
    xm: Vec<f64>,
    /// Column scales.
    xs: Vec<f64>,
}

impl Sparse {
    /// Build from raw CSC arrays. `col_ptr` has length `p+1`; `row_idx` and
    /// `values` have length `col_ptr[p]`. `w`/`xm`/`xs` are the standardization
    /// this matrix is presented under (see [`crate::standardize`]).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        p: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<f64>,
        w: Vec<f64>,
        xm: Vec<f64>,
        xs: Vec<f64>,
    ) -> Self {
        assert_eq!(col_ptr.len(), p + 1, "col_ptr must have length p+1");
        assert_eq!(
            row_idx.len(),
            values.len(),
            "row_idx and values length mismatch"
        );
        assert_eq!(w.len(), n);
        assert_eq!(xm.len(), p);
        assert_eq!(xs.len(), p);
        Sparse {
            n,
            p,
            col_ptr,
            row_idx,
            values,
            w,
            xm,
            xs,
        }
    }

    #[inline]
    fn col_range(&self, j: usize) -> std::ops::Range<usize> {
        self.col_ptr[j]..self.col_ptr[j + 1]
    }

    /// Column `j`'s nonzeros as `(row, value)` pairs.
    #[inline]
    fn col(&self, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.col_range(j);
        self.row_idx[r.clone()]
            .iter()
            .copied()
            .zip(self.values[r].iter().copied())
    }
}

impl DesignMatrix for Sparse {
    type Corr = f64; // mean-shift `o`: true weighted residual is r + o

    #[inline]
    fn nrows(&self) -> usize {
        self.n
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.p
    }

    // grad_j = [ sum_{i in nz(j)} w_i x_ij (r_i + o) ] / xs_j.
    //
    // The `o` term is the mean-shift correction: the true weighted residual is
    // r + o, and because that residual is kept weighted-mean-zero, the omitted
    // `-xm_j * sum_i w_i r_i` cross-term is exactly zero.
    #[inline]
    fn grad(&self, j: usize, r: &[f64], o: f64) -> f64 {
        let mut acc = 0.0;
        for (i, x) in self.col(j) {
            acc += self.w[i] * x * (r[i] + o);
        }
        acc / self.xs[j]
    }

    #[inline]
    fn update_resid(&self, j: usize, beta_diff: f64, r: &mut [f64], o: &mut f64) {
        let bds = beta_diff / self.xs[j];
        for (i, x) in self.col(j) {
            r[i] -= bds * x;
        }
        *o += bds * self.xm[j];
    }
}

/// Sparse analogue of [`chkvars`]: a column is usable unless every stored value
/// equals a common value AND that value fills the column (i.e. it is constant).
/// Mirrors glmnetpp `SpChkvars`.
pub fn chkvars_sparse(n: usize, p: usize, col_ptr: &[usize], values: &[f64]) -> Vec<bool> {
    (0..p)
        .map(|j| {
            let (b, e) = (col_ptr[j], col_ptr[j + 1]);
            let nnz = e - b;
            if nnz == 0 {
                return false; // all zeros -> constant
            }
            // If some entries are implicit zeros, a nonzero stored value already
            // makes the column non-constant.
            if nnz < n {
                return values[b..e].iter().any(|&v| v != 0.0);
            }
            // Fully stored: compare against the first, as in the dense case.
            let t = values[b];
            values[b + 1..e].iter().any(|&v| v != t)
        })
        .collect()
}

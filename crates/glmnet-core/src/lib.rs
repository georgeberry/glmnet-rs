//! A Rust port of glmnet's elastic-net coordinate-descent kernels.
//!
//! Ported from `glmnetpp` (the C++17 core of R glmnet >= 4.1), not from the
//! legacy Mortran/Fortran. Validated against R glmnet 5.0 output; see
//! `tests/parity.rs` and `scripts/gen_fixtures.R`.
//!
//! Terminology follows glmnet, not scikit-learn: `alpha` is the elastic-net
//! mixing parameter and `lambda` is the penalty strength.

pub mod binomial;
pub mod control;
pub mod error;
pub mod gaussian;
pub(crate) mod kernel;
pub mod matrix;
pub mod poisson;
pub mod standardize;

pub use binomial::{lognet, BinomialFit};
pub use control::{Control, FitConfig};
pub use error::{FitError, PathWarning};
pub use gaussian::{elnet_naive, GaussianFit};
pub use matrix::{Dense, DesignMatrix};
pub use poisson::{fishnet, PoissonFit};

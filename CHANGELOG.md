# Changelog

All notable changes to `glmnet-rs` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow semantic versioning (with the usual 0.x caveat that minor releases may
carry breaking changes).

## [Unreleased]

## [0.1.0]

First release. A Rust port of R glmnet's elastic-net coordinate descent
(`glmnetpp`), with a Python front end, validated against R glmnet 5.0 to
~1e-13–1e-15 with iteration counts (`npasses`) identical to R.

### Added

- **Families**: gaussian, two-class binomial (logistic), and poisson, dense `X`.
- **Sparse `X`** (CSC) for gaussian and binomial, validated against R's sparse
  path (`spelnet` / `splognet`).
- **Path API**: the full lambda path with the `big`-sentinel + `fix.lam`
  reconstruction, `fdev`/`devmax` early stopping, sequential strong rules, and
  `coef`/`predict` with `lambda.interp` interpolation.
- **Cross-validation** (`cv_glmnet`): `mse`/`mae`/`deviance`/`class`/`auc`
  measures, `lambda.min`/`lambda.1se`, with `cvm`/`cvsd` bit-exact to R's
  `cv.glmnet` (fold-own-lambda fits + interpolation onto the full grid).
- **Summaries** (`print`/`summary`/`to_frame`) and **plots** (`plot`) matching
  R's `plot.glmnet` and `plot.cv.glmnet` (matplotlib).
- **scikit-learn estimators** (`ElasticNet`, `Lasso`, `LogisticRegression`) with
  the correct parameter translation, including the `ys` factor on the L2 term.
- **Rust core** (`glmnet-core`): a pure-Rust library with a `DesignMatrix` /
  `GlmMatrix` abstraction that shares the solvers across dense and sparse.
- Validation on two real datasets (Wine Quality, Leukemia) and a runnable
  example notebook.

[Unreleased]: https://github.com/georgeberry/glmnet-rust/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/georgeberry/glmnet-rust/releases/tag/v0.1.0

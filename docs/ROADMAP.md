# Roadmap / what's missing vs R glmnet

Status of this port relative to the R `glmnet` package. Everything under "Done"
is validated against R to ~1e-13–1e-15 with iteration counts (`npasses`)
identical to R.

## Done

- **Families**: gaussian, binomial (two-class logistic), poisson — dense `X`.
- **Sparse `X`**: gaussian and binomial (CSC), validated against R's sparse path.
- **Path API**: full lambda path, `coef`/`predict` with `lambda.interp`, the
  `big`-sentinel + `fix.lam`, `fdev`/`devmax` early stopping, strong rules.
- **Cross-validation**: `cv_glmnet` with `mse`/`mae`/`deviance`/`class`/`auc`,
  `lambda.min`/`lambda.1se`, bit-exact `cvm`/`cvsd` vs R (fold-own-lambda +
  interpolation, matching `cv.glmnet.raw`).
- **Summaries**: `print`/`summary` tables + optional pandas `to_frame`.
- **Plots**: coefficient paths and CV curve (matplotlib), see `glmnet.plot`.
- **scikit-learn**: `ElasticNet`, `Lasso`, `LogisticRegression` with the correct
  parameterization (incl. the `ys` L2 correction).

## In flight (partially done)

- **Sparse poisson**: R fixtures generated (`spp_*`), solver not yet written.
  Poisson's sparse correction (`uu`/`tt`, folded into the prediction) differs
  from binomial's (`o`/`svr`), so it needs its own bookkeeping rather than the
  existing `GlmMatrix` trait as-is.
- **Python wiring for sparse binomial**: the Rust `lognet_sparse` is implemented
  and tested, but `glmnet(sparse_X, family="binomial")` still errors — Python
  only routes gaussian to the sparse path. Wire `_core.elnet_sparse`'s binomial
  analogue.

## Missing — families / solvers

Rough order of value:

1. **Offsets** — not supported in any family. Small, broadly useful (Poisson
   rate models, model calibration). Touches each family's `construct`.
2. **Multinomial** (softmax), grouped + ungrouped — the main missing family.
3. **`relax=TRUE`** — relaxed lasso (unpenalized refit of each active set).
   Popular; mostly orchestration over the existing path.
4. **Covariance gaussian solver** — R's default for `nvars < 500`. Same answers,
   different speed profile (gradient/covariance cache, no strong rule).
5. **Modified-Newton logistic** (`type.logistic="modified.Newton"`, `kopt=1`).
6. **Multi-response gaussian** (`mgaussian`).
7. **Cox** proportional hazards — needs risk-set gradient machinery; most
   specialized.
8. **Arbitrary GLM family objects** (`family=<R family>`, glmnet 4.0) — a generic
   IRLS path for any link/variance. Biggest lift.

## Missing — utilities / API

- **`assess.glmnet` / `roc.glmnet` / `confusion.glmnet` / `Cindex`** — held-out
  performance metrics.
- **`bigGlm`** — a single unpenalized GLM fit.
- **`exclude` as a function** (we accept indices only), `makeX` / NA handling.
- **Expose `glmnet.control`** (fdev/eps/big/…) from Python (the Rust `Control`
  exists; it just isn't a Python kwarg).
- **sklearn**: `PoissonRegressor`, a `GlmnetCV`-style estimator exposing the path.

## Missing — infrastructure

- PyPI wheels (`cibuildwheel` matrix), a docs site, CI running the test suite
  and the R-parity fixtures on push, perf-regression tracking.

## Validation assets

- `tests/fixtures/*.json` — R glmnet reference outputs (dense, sparse, cv).
- `scripts/gen_fixtures*.R` — regenerate them.
- `datasets/` + `scripts/compare_datasets.py` — real long/wide datasets and a
  coefficient + timing comparison against R.

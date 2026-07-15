# Porting notes

Working notes for anyone extending this port. The short version: **the elastic-net
math is easy, glmnet's behaviour is not.** Almost all the difficulty is in
faithfully reproducing decisions the algorithm makes *around* the coordinate
descent loop.

## What we are porting from

Not the Fortran. Modern glmnet (>= 4.1) computes everything in `src/glmnetpp/`,
a header-only C++17/Eigen library. The Mortran survives only as one vestigial
file. Get the source with:

```sh
curl -sLO https://cran.r-project.org/src/contrib/glmnet_5.0.tar.gz
```

`glmnetpp` is layered `driver -> path -> point -> internal`, assembled with CRTP
and policy templates. That indirection exists to share code across twelve
point-solvers (six families x dense/sparse). Rust traits do the same job, so the
layers collapse: `gaussian.rs` holds driver, path and point in one readable file.

## The oracle

`scripts/gen_fixtures.R` runs R glmnet 5.0 and dumps inputs plus expected
outputs to `tests/fixtures/*.json`. `crates/glmnet-core/tests/parity.rs` replays
them. Fixtures are committed so tests run without R.

Two things make this a strong oracle rather than a smoke test:

- **`npasses` is compared, not just coefficients.** Total coordinate-descent
  passes matching exactly means the port takes the *same iteration path*, not
  merely a nearby optimum. A coefficient-only test would pass on a subtly
  different algorithm.
- **`lmu` (path length) is compared.** It is data-dependent (see `fdev` below),
  so a mismatch localizes the bug to control flow rather than arithmetic.

Current status: 62 dense fixtures (22 Gaussian + 20 binomial + 20 Poisson) plus
12 genuinely-sparse Gaussian fixtures, all at max relative error ~1e-14 with
`npasses` identical to R on every case. Fixture family is encoded in the filename
prefix (`bin_*` binomial, `pois_*` Poisson, `sp_*` sparse Gaussian, else dense
Gaussian); the Rust parity harness dispatches on it. The sparse fixtures come
from R's *sparse* path (`spelnet` on a `dgCMatrix`), so the CSC solver is pinned
to R directly, not just to our own dense solver.

Two traps when generating fixtures:

- Pass `type.gaussian="naive"` explicitly. glmnet defaults to the **covariance**
  solver when `nvars < 500`, and only the naive solver uses the strong rule.
  Comparing a naive port against covariance output tests the wrong thing.
- Use `toJSON(digits = NA)`. `digits=17` means 17 *decimal* places, which is not
  a full double round-trip.
- In R, `a$lambda` **partial-matches** `lambda.min.ratio`. Use `a[["lambda"]]`.

## Quirks that are load-bearing

Reproducing these is the entire reason for the "parity first, refactor later"
strategy. A clean-room implementation from the papers gets all of them wrong.

### 1. `lambda[0]` is a sentinel, not lambda_max

The path is fit at `lambda[0] = big = 9.9e35`, which forces `beta == 0`.
lambda_max is only computed at `m == 1`. R then reconstructs `lambda[0]` by
log-linear extrapolation (`fix.lam.R`):

```r
lam[1] = exp(2*log(lam[2]) - log(lam[3]))
```

Since `lambda[1] = lmax*alf` and `lambda[2] = lmax*alf^2`, this returns exactly
`lmax`. Applied only when the user did not supply `lambda`.

### 2. The path length is data-dependent

`elnet_path/base.hpp` stops the path early when

```
me > dfmax  ||  (rsq - rsq_prev)/rsq < fdev  ||  rsq > devmax
```

with `fdev = 1e-5`, `devmax = 0.999`. So `nlambda=100` typically returns ~60-70
lambdas. A 1e-12 numerical difference can flip that comparison and truncate the
path one lambda early, changing the shape of every returned array. Any test that
naively compares arrays will fail confusingly. Compare `lmu` first.

### 3. A coefficient bound of exactly zero disables `fdev`

`glmnet.R:510`. A zero bound pins its coefficient, producing no deviance change,
which would spuriously trip the early stop. R sets `fdev = 0` for the whole fit
when `any(cl == 0)` — e.g. every non-negative lasso (`lower.limits=0`). This
lives in the R layer, not the C++, and is easy to miss when porting only the
kernels. It cost us a `lmu 63 != 100` failure.

### 4. `y` is scaled to unit variance, which distorts the L2 penalty

`standardize.hpp` does `ys = ||y||; y /= ys` and solves with
`lambda_tilde = lambda / ys`. L1 is homogeneous of degree 1 and rescales
cleanly. **L2 is degree 2 and does not.** In original units glmnet minimizes

```
(1/2)*sum_i w_i (y_i - b0 - x_i'b)^2
    + lambda*alpha*||b||_1
    + (lambda*(1-alpha)/(2*ys))*||b||_2^2      <-- stray 1/ys
```

Consequences: `?glmnet` warns about this obliquely ("glmnet standardizes y to
have unit variance before computing its lambda sequence"). It means
`glmnet(lambda=A, alpha=r)` is **not** `sklearn.ElasticNet(alpha=A, l1_ratio=r)`
unless `r == 1`. See `python/glmnet/sklearn.py::_to_glmnet` for the correct map:

```
lambda = A*r + A*(1-r)*ys
alpha  = A*r / lambda
```

Pure lasso hides the bug, which is precisely why it is dangerous.

### 5. Other details worth knowing

- `penalty.factor` is rescaled to sum to `nvars`, over **all** columns.
- `±Inf` box limits are replaced by `±big` in R *before* the C++ call. Without
  that, the later `cl *= xs` rescale yields `Inf * 0 = NaN` on constant columns.
- Weights are normalized to sum to 1 and folded into `X`/`y` as `sqrt(w)`; the
  solver never sees them again. But `nulldev` uses the **raw** weights.
- The full CD pass iterates only over the **strong set**, screened by the
  sequential strong rule `tlam = alpha*(2*lambda_m - lambda_{m-1})`, followed by
  a KKT check that readmits violators. Only the *naive* solver does this; the
  covariance solver screens on `ju` alone.
- `jerr` conventions (`util/exceptions.hpp`): `-m-1` = maxit at lambda `m`;
  `-10001-m` = `pmax` exceeded; `7777` = all predictors constant/excluded.

## Numerical fidelity and performance

Inner products go through `matrix::dot4` / `wdot4`, which use four partial
accumulators so LLVM can vectorize the reduction. A strict left-fold
(`acc += a[i]*b[i]`) has a loop-carried dependency Rust will not reassociate, so
it never vectorizes; four accumulators roughly **halve** whole-path solve time on
the larger problems (see `scripts/bench.py`).

The four-accumulator sum is reassociated, so it differs from a strict left-fold
at ~1e-15. This is *not* a fidelity regression, and the reasoning matters:
glmnetpp is built on Eigen, whose dot product is itself SIMD-reassociated, so a
strict scalar fold never reproduced glmnet's exact summation order either. What
the port holds to is the parity *test* — exact `npasses` and coefficients to
1e-12 against R — which `dot4` passes on all 42 fixtures. The convergence and
`fdev` thresholds have enough margin that a ~1e-15 summation difference does not
flip a decision on these problems. That robustness is *empirical*, not
guaranteed: an adversarial dataset sitting exactly on an `fdev` boundary could in
principle land a lambda differently. If you change the reduction again, re-run
the parity suite; do not assume bit-stability.

Current speed (Apple Silicon, full path, vs R glmnet on identical data): Gaussian
~0.6–0.85x of R, binomial ~0.7–1.1x (faster than R on tall/`n >> p`). The
remaining gap is mostly Eigen's more mature short-vector and cache handling; the
`p >> n` ("wide") case is the weakest because dots are short and the KKT sweep is
over many columns. Data marshaling (numpy C-order → column-major) was measured to
be negligible: pure-core timings (`examples/bench_core.rs`) match the Python
numbers.

## Binomial (two-class logistic) — done

`binomial.rs` ports `ElnetPath<binomial,two_class>` for dense `X`, exact-Hessian
Newton (`kopt = 0`, R's default `type.logistic = "Newton"`), no offset. The
IRLS-over-WLS loop it introduces is the architectural unlock that Poisson, Cox
and multinomial all reuse. What differs from Gaussian:

- **`X` is centered but not `sqrt(w)`-scaled**, and `y` is not rescaled. Weights
  enter through the IRLS working weights `v = w*q*(1-q)`, so the weight appears
  explicitly in the gradient `<x_k, w*(y-q)>` and variance `sum_i v_i x_ij^2`.
  This is exactly the seam the sparse `sp_*` solvers exist for.
- **IRLS/WLS control flow.** Outer IRLS freezes `v` and the column variance `xv`,
  runs coordinate descent to convergence (inner WLS), then recomputes `q,v,r` and
  tests convergence (coefficients stable vs the pre-WLS snapshot) plus the strong
  KKT check. State warm-starts across lambdas; the iterate counts depend on it.
- **The `fdev` test uses the *absolute* change in the deviance ratio**
  (`dev(m) - dev(m-1)`), where Gaussian used the *relative* change in R^2. Same
  constant, different quantity.
- **Probability clamps.** The linear predictor is clamped to `±log(1/pmin - 1)`
  so a separating hyperplane cannot drive coefficients to infinity, and the path
  stops when the total working variance falls below `vmin = (1+pmin)pmin(1-pmin)`
  — the logistic analogue of Gaussian saturation.
- **Deviance bookkeeping.** `dev(m) = (dev_null - dev_current)/dev_null` with `p`
  clamped into `[pmin, 1-pmin]` (glmnetpp `dev2`); reported `nulldev = 2*sw*dev0`.
- The zero-bound `fdev` disable (quirk 3 above) applies here too — it bit the
  non-negative logistic (`bin_nonneg`) exactly as it did `lasso_lowerlimit0`.

Point solver specialization: unlike Gaussian, binomial's `Point` is concrete over
`Dense` rather than generic over `DesignMatrix`, because its weighted per-column
operations don't reduce to the trait's `dot`/`axpy`. The right trait surface for
those falls out of the sparse work; guessing it now would likely be wrong.

The sklearn map for logistic is cleaner than for least squares — no `ys` factor,
because logistic does not standardize `y`. `LogisticRegression(C, penalty)` sets
`lambda = 1/(C*N)` and `alpha = l1_ratio` (`penalty="l2"` -> 0, `"l1"` -> 1).

## Poisson (log-link counts) — done

`poisson.rs` ports `ElnetPath<poisson,naive>` for dense `X`, no offset. It reuses
binomial's IRLS-over-WLS loop almost verbatim; the substance is in the link and
the deviance bookkeeping:

- **Log link.** Mean `mu = exp(eta)`, and Poisson variance = mean, so the working
  weight is `w = q*mu` (`q` = observation weights) and the residual is
  `r = q*y - w`. Column variance is `sum_i w_i x_ij^2`, recomputed each IRLS step.
- **No variance-collapse stop.** Instead the linear predictor is clamped in
  magnitude to `log(f64::MAX * 0.1)` before exponentiating, so `exp(eta)` can't
  overflow. That clamp is the Poisson analogue of binomial's `vmin`.
- **Two path-layer quirks, both load-bearing** (`elnet_path/poisson_base.hpp`):
  - `initialize_path` does `sml *= 10`, so the effective `fdev` is **10x larger**
    (1e-4). Miss this and every Poisson path runs too long.
  - The early-stop test is `(dev(m) - dev(m-mnl+1)) / dev(m)` — a *relative*
    change looking back `mnl-1` lambdas — where binomial used an *absolute*
    one-step change. Different quantity, different lookback.
- **Deviance.** `dev(m) = (t.eta - sum(w) - dv0) / dev0` with `t = q*y`; the null
  deviance carries a `-yb + sum_{t>0} t log(y)` correction. Reported
  `nulldev = 2*sw*dev0`, same shape as binomial.
- **Errors.** Negative `y` (jerr 8888) and non-positive weight sum (jerr 9999).
- The zero-bound `fdev` disable applies here too (`pois_nonneg`).

Like binomial, `Point` is concrete over `Dense` for the weighted per-column ops.

## Sparse `X` (Gaussian) — done

Sparse support is the payoff of the `DesignMatrix` trait. A sparse `X` cannot be
centered without destroying its sparsity, so it is kept in raw CSC form and the
standardization correction is folded into each column op (glmnetpp's `sp_*`
solvers). The trait now abstracts exactly the two operations the Gaussian solver
needs, with a per-backend correction state (`type Corr`):

- **`grad(j, r, corr)`** — dense: a plain (weighted) dot, `Corr = ()`. Sparse:
  `[sum_{i in nz(j)} w_i x_ij (r_i + o)] / xs_j`, `Corr = f64` (the mean-shift `o`).
- **`update_resid(j, beta_diff, r, corr)`** — dense: `r -= beta_diff * x_j`.
  Sparse: `r -= (beta_diff/xs_j) * x_j` over the nonzeros, `o += (beta_diff/xs_j)*xm_j`.

The trick works because the true weighted residual is `r + o` and is kept
**weighted-mean-zero** (every update preserves `sum_i w_i r_i = 0`), so the
`xm`-cross-term in the true gradient vanishes and each op stays O(nnz in column
`j`). The dense/binomial/poisson solvers were untouched by the refactor (the
parity suite confirms it); the Gaussian `Point`/path loop is now generic over the
trait, so dense and sparse share everything but standardization and matrix
construction.

Sparse standardization (`standardize_naive_sparse`, glmnetpp `SpStandardize1`)
differs from dense in two ways: `X` is left untouched, and `y` is centered/scaled
*without* the `sqrt(w)` premultiply (the weights stay separate, applied inside
`grad`). Everything downstream is the shared `run_path`.

Measured: on `n=2000, p=20000` at 1% density, the sparse path is ~23x faster than
densifying, with identical `npasses` and coefficients. Gaussian only for now;
binomial/poisson sparse would need their `Point` generalized off `Dense`.

## What's next

In rough order of value:

1. **`gaussian_cov`** — the default solver for `nvars < 500`. Shares the driver;
   different point solver (maintains a gradient/covariance cache, no strong rule).
2. **Sparse binomial / poisson** — generalize their `Point` off `Dense` onto a
   trait carrying the GLM weighted-column ops, then add the `sp_*` correction.
3. **`multinomial` / `cox`** — both reuse the IRLS loop; cox needs risk-set
   gradient machinery.
4. **`cv.glmnet`** — pure Python, on top of the path object.

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

Current status: 22/22 fixtures, max relative error ~1e-15, `npasses` identical
on every case.

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

## Numerical fidelity

`Dense::dot` is a plain scalar loop, matching Eigen's summation order for
un-vectorized columns. Replacing it with a chunked or SIMD reduction changes the
floating-point summation order, which perturbs `dlx` and can change `npasses`.
That is *allowed* (the answer stays correct), but it will break the exact
`npasses` comparison. If you vectorize, relax that assertion deliberately and
say so — do not silently loosen it.

## What's next

The Gaussian naive path is done and verified. In rough order of value:

1. **`gaussian_cov`** — the default solver for `nvars < 500`. Shares the driver;
   different point solver (maintains a gradient/covariance cache, no strong rule).
2. **`binomial`** — introduces the IRLS-over-WLS loop
   (`ElnetPointNonLinearCRTPBase::irls`). This is the architectural unlock: cox,
   poisson and multinomial all reuse that loop. Watch `pmin`/`exmx` clamps.
3. **`poisson`** — reuses the IRLS loop.
4. **Sparse (CSC)** — `matrix.rs::DesignMatrix` already isolates the two
   operations that need the `xm`/`xs` correction. Sparse `X` is never centered
   (that would destroy sparsity), so the correction is applied per gradient and
   per residual update. This is why upstream carries a parallel `sp_*` solver for
   every family; here it should be one more `impl DesignMatrix`.
5. **`cv.glmnet`** — pure Python, on top of the path object.

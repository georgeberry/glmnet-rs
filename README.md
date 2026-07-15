# glmnet-rs

A port of [glmnet](https://glmnet.stanford.edu)'s elastic-net coordinate descent
to Rust, with a Python front end.

Ported from `glmnetpp` (the C++17 core of R glmnet >= 4.1), **not** the legacy
Fortran, and validated against R glmnet 5.0.

**Status:** Gaussian, two-class binomial (logistic), and Poisson (dense `X`,
naive/Newton solvers), plus **sparse `X` for Gaussian** (CSC). 74 parity fixtures
pass at ~1e-14 relative error with iteration counts (`npasses`) identical to R —
the sparse ones validated against R's own sparse path. Multinomial, Cox, the
covariance solver, and sparse GLM families are not implemented yet — see
[`docs/PORTING.md`](docs/PORTING.md).

## Layout

```
crates/glmnet-core/   pure Rust kernels (no Python, no C)
crates/glmnet-py/     PyO3 bindings, deliberately thin
python/glmnet/        the user-facing package
scripts/gen_fixtures.R   generates the R reference fixtures
tests/fixtures/       committed R glmnet output (tests run without R)
```

## Two APIs, one solver

Faithful to R — the lambda path is the primitive, because it is what the
algorithm actually computes:

```python
from glmnet import glmnet

path = glmnet(X, y, alpha=1.0)     # alpha = elastic-net mixing (1 = lasso)
path.lambda_                       # (lmu,) descending
path.beta                          # (p, lmu)
path.coef(s=0.05)                  # interpolated, as in R's coef(fit, s=)
path.predict(X, s=0.05)
path.df                            # nonzeros per lambda

# logistic regression, same path object
lpath = glmnet(X, y01, family="binomial")
lpath.predict(X, s=0.05, type="response")   # class-1 probability

# poisson counts
ppath = glmnet(X, counts, family="poisson")
ppath.predict(X, s=0.05, type="response")   # expected count, exp(eta)

# sparse X (gaussian): pass a scipy sparse matrix, never densified
import scipy.sparse as sp
spath = glmnet(sp.csc_matrix(X), y)          # ~20x faster when p >> n and sparse

# cross-validation to pick lambda (matches R's cv.glmnet)
from glmnet import cv_glmnet
cv = cv_glmnet(X, y, family="gaussian", type_measure="mse", nfolds=10)
cv.lambda_min, cv.lambda_1se
cv.predict(X, s="lambda.1se")

# summaries, like R's print()
print(path)        # Df / %Dev / Lambda table
print(cv)          # lambda.min / lambda.1se with measure and SE
path.to_frame()    # optional pandas DataFrame
```

scikit-learn compatible, using **scikit-learn's** meaning of `alpha`:

```python
from glmnet.sklearn import ElasticNet, Lasso, LogisticRegression

m = ElasticNet(alpha=0.1, l1_ratio=0.7).fit(X, y)   # alpha = penalty strength
m.coef_, m.intercept_

clf = LogisticRegression(C=1.0, penalty="l2").fit(X, y01)
clf.predict_proba(X)
```

> **The `alpha` trap.** In glmnet `alpha` is the mixing parameter and `lambda`
> is the penalty strength. In scikit-learn `alpha` *is* the penalty strength and
> `l1_ratio` is the mixing. Worse, the two objectives are not related by a simple
> rename: glmnet rescales `y` to unit variance, which leaves the L2 term carrying
> a factor of `1/sd(y)`. `glmnet.sklearn` handles the conversion; the derivation
> is in [`docs/PORTING.md`](docs/PORTING.md#4-y-is-scaled-to-unit-variance-which-distorts-the-l2-penalty).

## Develop

```sh
cargo test -p glmnet-core --release        # parity against committed fixtures
maturin develop --release --uv             # build the extension
python -m pytest tests/test_python.py      # end-to-end + sklearn agreement

Rscript scripts/gen_fixtures.R             # regenerate Gaussian fixtures (needs R + glmnet)
Rscript scripts/gen_fixtures_binomial.R    # regenerate binomial fixtures
Rscript scripts/gen_fixtures_poisson.R     # regenerate Poisson fixtures
Rscript scripts/gen_fixtures_sparse.R      # regenerate sparse Gaussian fixtures (needs Matrix)

python scripts/bench.py                    # wall-clock vs R glmnet on identical data
cargo run --release -p glmnet-core --example bench_core   # pure-core timings
```

## Performance

Full-path wall clock vs R glmnet on identical data (Apple Silicon): Gaussian
runs at ~0.6–0.85x of R, two-class logistic at ~0.7–1.1x (faster than R when
`n >> p`). glmnet's compiled core is heavily tuned Eigen/SIMD, so
parity-to-1.5x-slower is the expected range for a pure-Rust port. Inner products
use four-accumulator reductions (`matrix::dot4`) that vectorize; see
[`docs/PORTING.md`](docs/PORTING.md#numerical-fidelity-and-performance).

## License

GPL-2.0-only, matching upstream glmnet.

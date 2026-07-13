# glmnet-rs

A port of [glmnet](https://glmnet.stanford.edu)'s elastic-net coordinate descent
to Rust, with a Python front end.

Ported from `glmnetpp` (the C++17 core of R glmnet >= 4.1), **not** the legacy
Fortran, and validated against R glmnet 5.0.

**Status:** Gaussian family, dense `X`, naive solver. 22/22 parity fixtures pass
at ~1e-15 relative error with iteration counts (`npasses`) identical to R.
Binomial, Poisson, the covariance solver, and sparse `X` are not implemented yet
— see [`docs/PORTING.md`](docs/PORTING.md).

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
```

scikit-learn compatible, using **scikit-learn's** meaning of `alpha`:

```python
from glmnet.sklearn import ElasticNet, Lasso

m = ElasticNet(alpha=0.1, l1_ratio=0.7).fit(X, y)   # alpha = penalty strength
m.coef_, m.intercept_
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

Rscript scripts/gen_fixtures.R             # regenerate fixtures (needs R + glmnet)
```

## License

GPL-2.0-only, matching upstream glmnet.

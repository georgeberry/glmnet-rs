"""glmnet-rs: glmnet's elastic-net coordinate descent, ported to Rust.

Two APIs over one solver:

* :func:`glmnet.glmnet` -- faithful to R. Fits a whole lambda path and returns a
  :class:`~glmnet._path.GlmnetPath`. ``alpha`` is the elastic-net mixing parameter.
* :mod:`glmnet.sklearn` -- ``ElasticNet`` / ``Lasso`` estimators using
  scikit-learn's names, where ``alpha`` is the penalty strength. Requires scikit-learn.

>>> from glmnet import glmnet
>>> path = glmnet(X, y, alpha=1.0)            # doctest: +SKIP
>>> path.coef(s=0.1)                          # doctest: +SKIP
"""

from ._path import GlmnetPath, glmnet, lambda_interp

__all__ = ["GlmnetPath", "glmnet", "lambda_interp"]
__version__ = "0.1.0"

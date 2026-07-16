"""glmnet-rs: glmnet's elastic-net coordinate descent, ported to Rust.

Two APIs over one solver:

* :func:`glmnetrs.glmnet` -- faithful to R. Fits a whole lambda path and returns a
  :class:`~glmnetrs._path.GlmnetPath`. ``alpha`` is the elastic-net mixing
  parameter; ``family`` is ``"gaussian"``, ``"binomial"`` or ``"poisson"``.
* :mod:`glmnetrs.sklearn` -- ``ElasticNet`` / ``Lasso`` estimators using
  scikit-learn's names, where ``alpha`` is the penalty strength. Requires scikit-learn.

>>> from glmnetrs import glmnet
>>> path = glmnet(X, y, alpha=1.0)            # doctest: +SKIP
>>> path.coef(s=0.1)                          # doctest: +SKIP
"""

from importlib.metadata import PackageNotFoundError, version

from ._path import GlmnetPath, glmnet, lambda_interp
from .cv import CVGlmnet, cv_glmnet

try:
    __version__ = version("glmnet-rs")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = ["GlmnetPath", "glmnet", "lambda_interp", "CVGlmnet", "cv_glmnet"]

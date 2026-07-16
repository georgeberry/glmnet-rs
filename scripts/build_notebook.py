#!/usr/bin/env python
"""Build and execute examples/glmnet_demo.ipynb.

Constructs the demo notebook cell-by-cell, runs it end to end (so the plots and
outputs are embedded), and writes the result. Rebuild whenever the API or the
narrative changes:

    python scripts/build_notebook.py
"""

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "glmnet_demo.ipynb"

md = new_markdown_cell
code = new_code_cell

cells = [
    md(
        "# glmnet-rs — a tour on two real datasets\n"
        "\n"
        "[`glmnet-rs`](https://github.com/georgeberry/glmnet-rust) is a Rust port of "
        "the [glmnet](https://glmnet.stanford.edu) elastic-net solver, with a Python "
        "front end. Its coefficients match R glmnet to ~1e-13 (verified on the two "
        "datasets below).\n"
        "\n"
        "We walk through the whole workflow — the lambda path, cross-validation, "
        "coefficient/CV plots, and prediction — on:\n"
        "\n"
        "- **Wine Quality** (*long*: n=4898, p=11) — Gaussian regression.\n"
        "- **Leukemia gene expression** (*wide*: n=72, p=7128) — binomial "
        "classification (ALL vs AML), the classic sparse-selection example."
    ),
    code(
        "%matplotlib inline\n"
        "import pathlib\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from glmnetrs import glmnet, cv_glmnet\n"
        "\n"
        "# Find the repo's datasets/ dir whether this runs from the repo root or examples/.\n"
        "here = pathlib.Path.cwd()\n"
        "while not (here / 'datasets').exists() and here != here.parent:\n"
        "    here = here.parent\n"
        "DATA = here / 'datasets'\n"
        "print('datasets:', DATA)"
    ),
    md(
        "## 1. Wine Quality — long, Gaussian\n"
        "\n"
        "4898 white wines, 11 physicochemical measurements, predict the sensory "
        "`quality` score. glmnet standardizes the features internally."
    ),
    code(
        "raw = np.genfromtxt(DATA / 'winequality-white.csv', delimiter=';', names=True)\n"
        "feat_names = [n for n in raw.dtype.names if n != 'quality']\n"
        "Xw = np.column_stack([raw[n] for n in feat_names])\n"
        "yw = raw['quality'].astype(float)\n"
        "# z-score the features so coefficient paths are on a comparable scale and\n"
        "# readable as importances (glmnet standardizes internally either way).\n"
        "Xw = (Xw - Xw.mean(0)) / Xw.std(0)\n"
        "print(f'X: {Xw.shape},  y range: {yw.min():.0f}..{yw.max():.0f}')\n"
        "print('features:', ', '.join(feat_names))"
    ),
    md(
        "### The lambda path\n"
        "`glmnet` fits the *entire* regularization path in one call. `print` gives "
        "R's `Df / %Dev / Lambda` table (showing the first rows)."
    ),
    code(
        "path = glmnet(Xw, yw)          # family='gaussian', alpha=1 (lasso) by default\n"
        "print('\\n'.join(path.summary().splitlines()[:12]))\n"
        "print('...')\n"
        "print(f'{path.lmu} lambdas fit; up to {100*path.dev_ratio[-1]:.1f}% deviance explained')"
    ),
    md(
        "Each line is one coefficient's trajectory as the penalty relaxes "
        "(left = strong penalty, everything zero; right = weak penalty). The top "
        "axis is the number of nonzero coefficients."
    ),
    code("ax = path.plot(xvar='lambda'); ax.set_title('Wine — coefficient paths'); plt.show()"),
    md(
        "### Cross-validation to pick lambda\n"
        "`cv_glmnet` mirrors R's `cv.glmnet`: it reports `lambda.min` (best CV error) "
        "and `lambda.1se` (the most regularized model within one standard error — the "
        "usual choice for a parsimonious fit)."
    ),
    code(
        "cvw = cv_glmnet(Xw, yw, type_measure='mse', nfolds=10, seed=1)\n"
        "print(cvw)\n"
        "ax = cvw.plot(); ax.set_title('Wine — CV curve'); plt.show()"
    ),
    md(
        "### What the model selected\n"
        "Coefficients at `lambda.1se` (standardized features, so magnitudes are "
        "directly comparable as importances). The signs line up with wine intuition — "
        "higher **alcohol** predicts higher quality, higher **volatile acidity** "
        "predicts lower."
    ),
    code(
        "coef = cvw.coef(s='lambda.1se').ravel()\n"
        "intercept, betas = coef[0], coef[1:]\n"
        "order = np.argsort(-np.abs(betas))\n"
        "print(f'intercept: {intercept:.3f}')\n"
        "for j in order:\n"
        "    if betas[j] != 0:\n"
        "        print(f'  {feat_names[j]:24s} {betas[j]:+.3f}')\n"
        "print(f'\\n{np.count_nonzero(betas)} of {len(betas)} features used')"
    ),
    md(
        "### Held-out accuracy\n"
        "Fit on a train split, evaluate RMSE on the test split at both lambdas, and "
        "compare to ordinary least squares. The `lambda.1se` model is more compact and "
        "generalizes about as well."
    ),
    code(
        "rng = np.random.default_rng(0)\n"
        "idx = rng.permutation(len(yw)); cut = int(0.8 * len(yw))\n"
        "tr, te = idx[:cut], idx[cut:]\n"
        "cv_tr = cv_glmnet(Xw[tr], yw[tr], type_measure='mse', nfolds=10, seed=2)\n"
        "\n"
        "def rmse(pred):\n"
        "    return float(np.sqrt(np.mean((yw[te] - pred) ** 2)))\n"
        "\n"
        "rmse_min = rmse(cv_tr.predict(Xw[te], s='lambda.min').ravel())\n"
        "rmse_1se = rmse(cv_tr.predict(Xw[te], s='lambda.1se').ravel())\n"
        "# plain OLS for reference\n"
        "A = np.column_stack([np.ones(len(tr)), Xw[tr]])\n"
        "beta_ols, *_ = np.linalg.lstsq(A, yw[tr], rcond=None)\n"
        "rmse_ols = rmse(np.column_stack([np.ones(len(te)), Xw[te]]) @ beta_ols)\n"
        "print(f'test RMSE  lasso(min): {rmse_min:.4f}   lasso(1se): {rmse_1se:.4f}   OLS: {rmse_ols:.4f}')\n"
        "nz_1se = np.count_nonzero(cv_tr.coef(s='lambda.1se').ravel()[1:])\n"
        "print(f'lambda.1se uses {nz_1se} features vs OLS\\'s {Xw.shape[1]}')"
    ),
    md(
        "## 2. Leukemia — wide, binomial\n"
        "\n"
        "The Golub et al. (1999) benchmark: 72 patients, 7128 gene-expression values, "
        "classify **ALL vs AML**. With `p ≫ n`, the lasso's job is to find a *handful* "
        "of genes that separate the classes — the canonical glmnet story."
    ),
    code(
        "with open(DATA / 'leukemia_big.csv') as fh:\n"
        "    labels = np.array(fh.readline().strip().split(','))\n"
        "expr = np.loadtxt(DATA / 'leukemia_big.csv', delimiter=',', skiprows=1)\n"
        "Xl = expr.T                       # samples x genes\n"
        "yl = (labels == 'AML').astype(float)\n"
        "print(f'X: {Xl.shape}   ALL: {int((yl==0).sum())}, AML: {int((yl==1).sum())}')"
    ),
    code(
        "lpath = glmnet(Xl, yl, family='binomial')\n"
        "print(f'{lpath.lmu} lambdas; nonzero genes range {lpath.df.min()}..{lpath.df.max()}')\n"
        "ax = lpath.plot(xvar='lambda'); ax.set_title('Leukemia — coefficient paths (7128 genes)'); plt.show()"
    ),
    md(
        "Only a few of the 7128 genes ever leave zero. Cross-validate on "
        "misclassification error (6 folds, so ~12 samples per fold)."
    ),
    code(
        "cvl = cv_glmnet(Xl, yl, family='binomial', type_measure='class', nfolds=6, seed=1)\n"
        "print(cvl)\n"
        "ax = cvl.plot(); ax.set_title('Leukemia — CV misclassification'); plt.show()"
    ),
    md(
        "### A sparse gene signature\n"
        "At `lambda.1se`, a small set of genes classifies the training data — and their "
        "linear predictor cleanly separates ALL from AML."
    ),
    code(
        "genes = np.flatnonzero(cvl.coef(s='lambda.1se').ravel()[1:] != 0)\n"
        "print(f'{len(genes)} genes selected at lambda.1se:  {genes.tolist()}')\n"
        "\n"
        "prob = cvl.predict(Xl, s='lambda.1se', type='response').ravel()\n"
        "pred = (prob > 0.5).astype(float)\n"
        "acc = (pred == yl).mean()\n"
        "print(f'training accuracy with {len(genes)} genes: {acc:.1%}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 3))\n"
        "eta = cvl.predict(Xl, s='lambda.1se', type='link').ravel()\n"
        "ax.scatter(eta[yl == 0], np.zeros((yl == 0).sum()), label='ALL', alpha=.7)\n"
        "ax.scatter(eta[yl == 1], np.ones((yl == 1).sum()), label='AML', alpha=.7)\n"
        "ax.axvline(0, ls=':', c='k'); ax.set_yticks([0, 1]); ax.set_yticklabels(['ALL', 'AML'])\n"
        "ax.set_xlabel('linear predictor'); ax.set_title('Separation from the selected genes'); ax.legend()\n"
        "plt.show()"
    ),
    md(
        "## 3. scikit-learn interface\n"
        "\n"
        "For pipelines/grid-search, `glmnet.sklearn` exposes estimators with "
        "scikit-learn's parameter names (`alpha` = penalty strength, `l1_ratio` = "
        "mixing) — the conversion to glmnet's convention (including the subtle `ys` "
        "factor on the L2 term) is handled for you."
    ),
    code(
        "from glmnetrs.sklearn import ElasticNet, LogisticRegression\n"
        "en = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(Xw, yw)\n"
        "clf = LogisticRegression(C=1.0).fit(Xl, yl)\n"
        "print('ElasticNet nonzero coefs:', int(np.count_nonzero(en.coef_)))\n"
        "print('LogisticRegression train accuracy:', f'{clf.score(Xl, yl):.1%}')"
    ),
    md(
        "---\n"
        "That's the tour: same API as R glmnet, matching coefficients, plus a "
        "scikit-learn shim. See [`docs/ROADMAP.md`](../docs/ROADMAP.md) for what's "
        "implemented and what's next."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

print("executing notebook ...")
client = NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(ROOT)}})
client.execute()

OUT.parent.mkdir(exist_ok=True)
nbformat.write(nb, OUT)
print(f"wrote {OUT.relative_to(ROOT)}  ({len(cells)} cells)")

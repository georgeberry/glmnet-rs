# Datasets

Two real, freely-available datasets used to validate `rust-glmnet` against R
glmnet on genuine data (not just synthetic fixtures): one **long** (`n >> p`) and
one **wide** (`p >> n`). They are committed here for reproducibility but are
**not** part of the installable package — the Python wheel only ships
`python/glmnet/`, so nothing under `datasets/` is included.

Run the comparison with `python scripts/compare_datasets.py` (needs R + glmnet
for the reference side).

## `winequality-white.csv` — long, gaussian

Vinho Verde white-wine physicochemical measurements; predict the sensory
`quality` score from 11 features.

- Shape: **n = 4898, p = 11**.
- Format: `;`-separated, header row; last column `quality` is the response.
- Source: UCI Machine Learning Repository, "Wine Quality" (P. Cortez et al.,
  2009), <https://archive.ics.uci.edu/dataset/186/wine+quality>.
- License: **CC BY 4.0**.

## `leukemia_big.csv` — wide, binomial

The Golub et al. (1999) leukemia gene-expression benchmark — the canonical wide
example in the glmnet / *Elements of Statistical Learning* literature. Classify
ALL vs AML from microarray expression.

- Stored genes-as-rows: **7128 genes x 72 samples**; row 1 is the per-sample
  class label (`ALL`/`AML`). Loaders transpose to `X` of shape **n = 72,
  p = 7128** and set `y = (label == "AML")`.
- Source: Trevor Hastie's data mirror,
  <https://hastie.su.domains/CASI_files/DATA/leukemia_big.csv> (from Golub et
  al., *Science* 1999). Publicly distributed for research use.

## Regenerating / re-downloading

```sh
cd datasets
curl -sLO https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
curl -sLO https://hastie.su.domains/CASI_files/DATA/leukemia_big.csv
```

#!/usr/bin/env python
"""Compare rust-glmnet against R glmnet on two real datasets -- one long
(n >> p) and one wide (p >> n) -- on both coefficients and wall clock.

Both sides read the same CSVs (see datasets/README.md) and fit the full lambda
path with matching solver settings, so the comparison is apples-to-apples. The R
side (scripts/compare_datasets.R) writes its results to datasets/.compare/;
run it first, or let this script invoke it.
"""

import json
import pathlib
import subprocess
import sys
import time

import numpy as np
from glmnetrs import glmnet

HERE = pathlib.Path(__file__).parent
DATA = HERE / ".." / "datasets"
COMPARE = DATA / ".compare"


def load_wine():
    """Long, gaussian: predict wine quality (n=4898, p=11)."""
    arr = np.genfromtxt(DATA / "winequality-white.csv", delimiter=";", skip_header=1)
    return arr[:, :-1], arr[:, -1], "gaussian"


def load_leukemia():
    """Wide, binomial: ALL vs AML from gene expression (n=72, p=7128)."""
    with open(DATA / "leukemia_big.csv") as fh:
        labels = np.array(fh.readline().strip().split(","))
    expr = np.loadtxt(DATA / "leukemia_big.csv", delimiter=",", skiprows=1)
    return expr.T, (labels == "AML").astype(float), "binomial"


def time_ours(X, y, family, reps=5):
    glmnet(X, y, family=family)  # warmup
    best = np.inf
    fit = None
    for _ in range(reps):
        t0 = time.perf_counter()
        fit = glmnet(X, y, family=family)
        best = min(best, time.perf_counter() - t0)
    return fit, best


def compare(name, fit, ours_time):
    r = json.loads((COMPARE / f"{name}.json").read_text())
    p = r["p"]
    lmu = min(fit.lmu, r["lmu"])
    r_beta = np.asarray(r["beta"]).reshape((p, r["lmu"]), order="F")
    r_a0 = np.asarray(r["a0"])
    r_lam = np.asarray(r["lambda"])

    def rel(a, b):
        return np.max(np.abs(a - b) / (1.0 + np.abs(b)))

    lam_err = rel(fit.lambda_[:lmu], r_lam[:lmu])
    a0_err = rel(fit.a0[:lmu], r_a0[:lmu])
    beta_err = rel(fit.beta[:, :lmu], r_beta[:, :lmu])
    return {
        "name": name,
        "p": p,
        "lmu_ours": fit.lmu,
        "lmu_r": r["lmu"],
        "lam_err": lam_err,
        "a0_err": a0_err,
        "beta_err": beta_err,
        "ours_ms": ours_time * 1e3,
        "r_ms": r["time"] * 1e3,
    }


def main():
    if not COMPARE.exists() or not (COMPARE / "wine.json").exists():
        print("Running R side (scripts/compare_datasets.R) ...", flush=True)
        rc = subprocess.run(["Rscript", str(HERE / "compare_datasets.R")])
        if rc.returncode != 0:
            print("R side failed; run it manually.", file=sys.stderr)
            sys.exit(1)

    rows = []
    for name, loader in (("wine", load_wine), ("leukemia", load_leukemia)):
        X, y, family = loader()
        fit, t = time_ours(X, y, family)
        rows.append(compare(name, fit, t))

    print("\n" + "=" * 86)
    print(
        f"{'dataset':10} {'n':>5} {'p':>6} {'lmu':>4} "
        f"{'max coef relerr':>16} {'ours':>9} {'R':>9} {'speedup':>8}"
    )
    print("-" * 86)
    for r, (name, loader) in zip(rows, (("wine", load_wine), ("leukemia", load_leukemia))):
        X, _, _ = loader()
        worst = max(r["lam_err"], r["a0_err"], r["beta_err"])
        speed = r["r_ms"] / r["ours_ms"]
        lmu = f"{r['lmu_ours']}" if r["lmu_ours"] == r["lmu_r"] else f"{r['lmu_ours']}/{r['lmu_r']}"
        print(
            f"{name:10} {X.shape[0]:>5} {X.shape[1]:>6} {lmu:>4} "
            f"{worst:>16.2e} {r['ours_ms']:>7.1f}ms {r['r_ms']:>7.1f}ms {speed:>7.2f}x"
        )
    print("=" * 86)
    print("max coef relerr: worst relative difference in lambda / intercept / beta vs R.")


if __name__ == "__main__":
    main()

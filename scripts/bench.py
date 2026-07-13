#!/usr/bin/env python
"""Benchmark rust-glmnet against R glmnet on identical data.

Both fit the entire lambda path with the same solver settings, so this is an
apples-to-apples wall-clock comparison. Data is generated once here and written
to disk; `bench.R` reads the same matrices so neither side has a data advantage.

Timing methodology: each fit is repeated and the *minimum* is reported (min is
the standard choice for microbenchmarks -- it is the run least perturbed by
scheduler noise, GC, turbo throttling). A short warmup fit is discarded first.
"""

import json
import pathlib
import subprocess
import sys
import time

import numpy as np

from glmnet import glmnet

HERE = pathlib.Path(__file__).parent
SCRATCH = HERE / ".bench_data"
SCRATCH.mkdir(exist_ok=True)

# (name, n, p, family). Chosen to span the regimes glmnet is used in:
# small, tall (n >> p), square-ish, and wide (p >> n, e.g. genomics/text).
CASES = [
    ("small",    200,    20, "gaussian"),
    ("tall",   10000,    50, "gaussian"),
    ("medium",  1000,   200, "gaussian"),
    ("square",  2000,  1000, "gaussian"),
    ("wide",     200,  5000, "gaussian"),
    ("bin_small",   200,   20, "binomial"),
    ("bin_tall",  10000,   50, "binomial"),
    ("bin_medium", 1000,  200, "binomial"),
    ("bin_wide",    200, 2000, "binomial"),
    ("pois_small",   200,   20, "poisson"),
    ("pois_tall",  10000,   50, "poisson"),
    ("pois_medium", 1000,  200, "poisson"),
    ("pois_wide",    200, 2000, "poisson"),
]

REPEATS = 7


def make_data(n, p, family, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    k = min(10, p)
    beta = np.zeros(p)
    beta[:k] = rng.standard_normal(k) * 1.5
    eta = X @ beta
    if family == "gaussian":
        y = eta + rng.standard_normal(n)
    elif family == "binomial":
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:  # poisson; scale eta down so counts stay moderate
        y = rng.poisson(np.exp(0.3 * eta)).astype(float)
    return X, y


def time_ours(X, y, family):
    # warmup (compile caches, allocator warm)
    glmnet(X, y, family=family)
    best = np.inf
    lmu = 0
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fit = glmnet(X, y, family=family)
        best = min(best, time.perf_counter() - t0)
        lmu = fit.lmu
    return best, lmu


def main():
    results = []
    for idx, (name, n, p, family) in enumerate(CASES):
        # Deterministic per-case seed (Python's hash() is per-process randomized,
        # which would make runs non-reproducible and the outlier cases jump around).
        X, y = make_data(n, p, family, seed=1000 + idx)
        # Persist for the R side as raw little-endian f64: X column-major, then y.
        # Raw binary avoids any format dependency on the R side.
        np.asfortranarray(X).ravel(order="F").astype("<f8").tofile(SCRATCH / f"{name}_X.bin")
        y.astype("<f8").tofile(SCRATCH / f"{name}_y.bin")
        meta = {"name": name, "n": n, "p": p, "family": family}
        (SCRATCH / f"{name}_meta.json").write_text(json.dumps(meta))

        ours, lmu = time_ours(X, y, family)
        results.append({**meta, "ours_s": ours, "lmu": lmu})
        print(f"  [ours] {name:12s} n={n:<6d} p={p:<5d} {family:9s} "
              f"{ours*1e3:8.2f} ms   lmu={lmu}", flush=True)

    (SCRATCH / "cases.json").write_text(json.dumps([c[0] for c in CASES]))
    print("\nRunning R glmnet on the same data ...\n", flush=True)
    r = subprocess.run(
        ["Rscript", str(HERE / "bench.R")],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("R benchmark failed:\n", r.stderr, file=sys.stderr)
        r_times = {}
    else:
        r_times = json.loads(r.stdout.strip().splitlines()[-1])

    # --- combined table ---
    print("\n" + "=" * 78)
    print(f"{'case':12s} {'n':>6s} {'p':>5s} {'family':9s} "
          f"{'ours':>9s} {'R':>9s} {'speedup':>8s} {'lmu':>5s}")
    print("-" * 78)
    for row in results:
        rt = r_times.get(row["name"])
        rt_s = f"{rt*1e3:7.1f}ms" if rt else "   n/a"
        speed = f"{rt/row['ours_s']:6.2f}x" if rt else "   -"
        print(f"{row['name']:12s} {row['n']:>6d} {row['p']:>5d} {row['family']:9s} "
              f"{row['ours_s']*1e3:7.1f}ms {rt_s:>9s} {speed:>8s} {row['lmu']:>5d}")
    print("=" * 78)


if __name__ == "__main__":
    main()

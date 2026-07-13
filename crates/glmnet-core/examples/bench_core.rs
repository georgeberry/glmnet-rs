//! Times the pure-Rust core on column-major data, excluding all Python/binding
//! overhead. Compare against scripts/bench.py numbers to see how much of the
//! wall clock is marshaling vs solving.
//!
//! Run: `cargo run --release -p glmnet-core --example bench_core`

use glmnet_core::{elnet_naive, lognet, FitConfig};
use std::time::Instant;

// A tiny deterministic RNG (xorshift) so the example needs no dependencies and
// the data is reproducible run to run.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    // Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn make(n: usize, p: usize, binomial: bool, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng(seed | 1);
    let mut x = vec![0.0; n * p];
    for v in x.iter_mut() {
        *v = rng.normal();
    }
    let k = p.min(10);
    let mut beta = vec![0.0; p];
    for b in beta.iter_mut().take(k) {
        *b = rng.normal() * 1.5;
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut eta = 0.0;
        for (j, &b) in beta.iter().enumerate() {
            eta += x[j * n + i] * b; // column-major
        }
        y[i] = if binomial {
            let pr = 1.0 / (1.0 + (-eta).exp());
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            if u < pr {
                1.0
            } else {
                0.0
            }
        } else {
            eta + rng.normal()
        };
    }
    (x, y)
}

fn bench(name: &str, n: usize, p: usize, binomial: bool) {
    let (x, y) = make(
        n,
        p,
        binomial,
        0x9E3779B97F4A7C15 ^ (n as u64) ^ ((p as u64) << 20),
    );
    let cfg = FitConfig::new(n, p);

    let run = || {
        if binomial {
            lognet(&x, &y, n, p, &cfg).map(|f| f.lmu).unwrap()
        } else {
            elnet_naive(&x, &y, n, p, &cfg).map(|f| f.lmu).unwrap()
        }
    };

    run(); // warmup
    let repeats = 7;
    let mut best = f64::INFINITY;
    let mut lmu = 0;
    for _ in 0..repeats {
        let t0 = Instant::now();
        lmu = run();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let fam = if binomial { "binomial" } else { "gaussian" };
    println!(
        "  {name:12} n={n:<6} p={p:<5} {fam:9} {:9.2} ms   lmu={lmu}",
        best * 1e3
    );
}

fn main() {
    println!("pure-core timings (no Python/binding overhead):\n");
    bench("small", 200, 20, false);
    bench("tall", 10000, 50, false);
    bench("medium", 1000, 200, false);
    bench("square", 2000, 1000, false);
    bench("wide", 200, 5000, false);
    bench("bin_small", 200, 20, true);
    bench("bin_tall", 10000, 50, true);
    bench("bin_medium", 1000, 200, true);
    bench("bin_wide", 200, 2000, true);
}

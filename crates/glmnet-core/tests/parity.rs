//! Parity against R glmnet 5.0. Regenerate fixtures with `Rscript scripts/gen_fixtures.R`.

use glmnet_core::{elnet_naive, fishnet, lognet, Control, FitConfig};
use serde::{Deserialize, Deserializer};
use std::path::PathBuf;

/// The subset of a fit compared against R. Both families expose these fields.
struct Solved {
    lmu: usize,
    lambda: Vec<f64>,
    a0: Vec<f64>,
    beta: Vec<f64>,
    dev_ratio: Vec<f64>,
    npasses: usize,
}

/// jsonlite emits non-finite doubles as JSON strings ("Inf", "-Inf", "NaN"),
/// since JSON has no literal for them. Accept either form.
fn loose_f64_vec<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Loose {
        Num(f64),
        Str(String),
    }
    let raw: Vec<Loose> = Vec::deserialize(d)?;
    raw.into_iter()
        .map(|v| match v {
            Loose::Num(x) => Ok(x),
            Loose::Str(s) => match s.as_str() {
                "Inf" => Ok(f64::INFINITY),
                "-Inf" => Ok(f64::NEG_INFINITY),
                "NaN" => Ok(f64::NAN),
                other => Err(serde::de::Error::custom(format!("bad float {other:?}"))),
            },
        })
        .collect()
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct Fixture {
    name: String,
    n: usize,
    p: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    weights: Vec<f64>,
    alpha: f64,
    intercept: bool,
    standardize: bool,
    nlambda: usize,
    lambda_min_ratio: f64,
    user_lambda: Option<Vec<f64>>,
    penalty_factor: Vec<f64>,
    #[serde(deserialize_with = "loose_f64_vec")]
    lower_limits: Vec<f64>,
    #[serde(deserialize_with = "loose_f64_vec")]
    upper_limits: Vec<f64>,
    dfmax: usize,
    pmax: usize,
    thresh: f64,
    maxit: usize,

    lmu: usize,
    lambda: Vec<f64>,
    a0: Vec<f64>,
    beta: Vec<f64>,
    dev_ratio: Vec<f64>,
    npasses: usize,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures")
}

fn load_all() -> Vec<Fixture> {
    let dir = fixture_dir();
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| {
            panic!(
                "cannot read {}: {e}. Run scripts/gen_fixtures.R",
                dir.display()
            )
        })
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    entries.sort();
    for path in entries {
        let txt = std::fs::read_to_string(&path).unwrap();
        // jsonlite wraps scalars in length-1 arrays unless auto_unbox; we set
        // auto_unbox=TRUE, but length-1 *vectors* also unbox. Patch those back.
        let f: Fixture =
            serde_json::from_str(&txt).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        out.push(f);
    }
    assert!(!out.is_empty(), "no fixtures found in {}", dir.display());
    out
}

fn cfg_of(f: &Fixture) -> FitConfig {
    FitConfig {
        alpha: f.alpha,
        nlambda: f.nlambda,
        lambda_min_ratio: f.lambda_min_ratio,
        user_lambda: f.user_lambda.clone(),
        standardize: f.standardize,
        intercept: f.intercept,
        thresh: f.thresh,
        maxit: f.maxit,
        dfmax: f.dfmax,
        pmax: f.pmax,
        penalty_factor: Some(f.penalty_factor.clone()),
        lower_limits: Some(f.lower_limits.clone()),
        upper_limits: Some(f.upper_limits.clone()),
        weights: Some(f.weights.clone()),
        exclude: Vec::new(),
        control: Control::default(),
    }
}

/// Relative error with an absolute floor, so exact zeros compare cleanly.
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + b.abs())
}

#[test]
fn matches_r_glmnet() {
    // Observed worst case is ~1e-14; 1e-12 leaves headroom without hiding a regression.
    const TOL: f64 = 1e-12;
    let mut failures = Vec::new();

    for f in load_all() {
        // Fixture family is encoded in the filename prefix: `bin_*` binomial,
        // `pois_*` Poisson, everything else Gaussian.
        let solved = if f.name.starts_with("bin_") {
            lognet(&f.x, &f.y, f.n, f.p, &cfg_of(&f)).map(|fit| Solved {
                lmu: fit.lmu,
                lambda: fit.lambda,
                a0: fit.a0,
                beta: fit.beta,
                dev_ratio: fit.dev_ratio,
                npasses: fit.npasses,
            })
        } else if f.name.starts_with("pois_") {
            fishnet(&f.x, &f.y, f.n, f.p, &cfg_of(&f)).map(|fit| Solved {
                lmu: fit.lmu,
                lambda: fit.lambda,
                a0: fit.a0,
                beta: fit.beta,
                dev_ratio: fit.dev_ratio,
                npasses: fit.npasses,
            })
        } else {
            elnet_naive(&f.x, &f.y, f.n, f.p, &cfg_of(&f)).map(|fit| Solved {
                lmu: fit.lmu,
                lambda: fit.lambda,
                a0: fit.a0,
                beta: fit.beta,
                dev_ratio: fit.dev_ratio,
                npasses: fit.npasses,
            })
        };
        let fit = match solved {
            Ok(fit) => fit,
            Err(e) => {
                failures.push(format!("{}: solver error {e}", f.name));
                continue;
            }
        };

        // Path length is data-dependent (fdev / devmax / dfmax / pmax early stop),
        // so a mismatch here means a control-flow bug, not a precision one.
        if fit.lmu != f.lmu {
            failures.push(format!("{}: lmu {} != {} (R)", f.name, fit.lmu, f.lmu));
            continue;
        }

        let mut worst = (0.0f64, String::new());
        let mut bump = |e: f64, what: String| {
            if e > worst.0 {
                worst = (e, what);
            }
        };

        for k in 0..f.lmu {
            bump(rel(fit.lambda[k], f.lambda[k]), format!("lambda[{k}]"));
            bump(rel(fit.a0[k], f.a0[k]), format!("a0[{k}]"));
            bump(
                rel(fit.dev_ratio[k], f.dev_ratio[k]),
                format!("dev_ratio[{k}]"),
            );
            for j in 0..f.p {
                bump(
                    rel(fit.beta[k * f.p + j], f.beta[k * f.p + j]),
                    format!("beta[{j},{k}]"),
                );
            }
        }

        if worst.0 > TOL {
            failures.push(format!(
                "{}: max rel err {:.3e} at {} (tol {:.0e})",
                f.name, worst.0, worst.1, TOL
            ));
        } else {
            println!(
                "{:<26} lmu={:<4} max_rel_err={:.2e}  npasses {} vs {} (R)",
                f.name, fit.lmu, worst.0, fit.npasses, f.npasses
            );
        }
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

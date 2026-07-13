//! Error and warning conditions, carrying glmnet's `jerr` codes so that
//! diagnostics can be compared directly against the R implementation.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitError {
    /// jerr = 7777. Every usable predictor has zero variance, or all were excluded.
    AllExcluded,
    /// jerr = 10000. All penalty factors are <= 0.
    NonPositivePenalty,
    /// Not a glmnet `jerr`: R rejects these in `glmnet()` before reaching C++.
    PositiveLowerLimit,
    NegativeUpperLimit,
    /// jerr = 8001. A (binomial) class probability underflowed `pmin` at the
    /// null model -- effectively all responses are one class.
    ProbMinReached,
    /// jerr = 9001. A class probability exceeded `1 - pmin` at the null model.
    ProbMaxReached,
}

/// Conditions that truncate the lambda path but still return the lambdas
/// computed so far. glmnet reports these as negative `jerr` (a warning).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathWarning {
    /// jerr = -m-1. Coordinate descent hit `maxit` at lambda index `m`.
    MaxIterReached { lambda_index: usize },
    /// jerr = -10001-m. Active set exceeded `pmax` at lambda index `m`.
    MaxActiveReached { lambda_index: usize },
}

impl FitError {
    pub fn jerr(&self) -> i32 {
        match self {
            FitError::AllExcluded => 7777,
            FitError::NonPositivePenalty => 10000,
            FitError::PositiveLowerLimit | FitError::NegativeUpperLimit => 0,
            // glmnetpp reports these as 8001+m / 9001+m; at the null model m = 0.
            FitError::ProbMinReached => 8001,
            FitError::ProbMaxReached => 9001,
        }
    }
}

impl PathWarning {
    pub fn jerr(&self) -> i32 {
        match *self {
            PathWarning::MaxIterReached { lambda_index } => -(lambda_index as i32) - 1,
            PathWarning::MaxActiveReached { lambda_index } => -10001 - (lambda_index as i32),
        }
    }
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::AllExcluded => {
                write!(f, "all predictors are constant or excluded (jerr 7777)")
            }
            FitError::NonPositivePenalty => write!(f, "all penalty factors are <= 0 (jerr 10000)"),
            FitError::PositiveLowerLimit => write!(f, "lower limits should be non-positive"),
            FitError::NegativeUpperLimit => write!(f, "upper limits should be non-negative"),
            FitError::ProbMinReached => {
                write!(
                    f,
                    "null probability underflowed pmin; response is ~constant (jerr 8001)"
                )
            }
            FitError::ProbMaxReached => {
                write!(
                    f,
                    "null probability exceeded 1-pmin; response is ~constant (jerr 9001)"
                )
            }
        }
    }
}

impl std::error::Error for FitError {}

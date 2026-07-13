#!/usr/bin/env Rscript
# Generates reference fixtures from R glmnet, which is the oracle the Rust
# kernels are validated against.
#
# type.gaussian="naive" is passed explicitly everywhere: glmnet defaults to the
# *covariance* solver whenever nvars < 500, and the two solvers differ in path
# internals (only the naive one applies the sequential strong rule). Comparing a
# naive port against covariance output silently tests the wrong thing.

suppressMessages(library(glmnet))
suppressMessages(library(jsonlite))

set.seed(20260709)
outdir <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "tests", "fixtures")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cases <- list()
add <- function(name, x, y, ...) {
  cases[[length(cases) + 1]] <<- list(name = name, x = x, y = y, args = list(...))
}

mk <- function(n, p, rho = 0) {
  x <- matrix(rnorm(n * p), n, p)
  if (rho > 0) x <- x + rho * matrix(rnorm(n), n, p) # shared factor -> correlation
  x
}
sig <- function(x, k = 5, sd = 1) {
  b <- c(rep(1.5, min(k, ncol(x))), rep(0, max(0, ncol(x) - k)))
  as.vector(x %*% b) + rnorm(nrow(x), sd = sd)
}

x <- mk(100, 20); y <- sig(x)
add("lasso_n100_p20", x, y, alpha = 1)
add("enet_a50_n100_p20", x, y, alpha = 0.5)
add("ridge_a0_n100_p20", x, y, alpha = 0)          # alpha=0 -> max(alpha,1e-3) guard in lambda_max
add("lasso_nostd", x, y, alpha = 1, standardize = FALSE)
add("lasso_nointercept", x, y, alpha = 1, intercept = FALSE)
add("lasso_nostd_nointercept", x, y, alpha = 1, standardize = FALSE, intercept = FALSE)
add("lasso_userlambda", x, y, alpha = 1, lambda = exp(seq(log(2), log(0.005), length.out = 40)))
add("lasso_dfmax", x, y, alpha = 1, dfmax = 5)     # -> me > ne early break
add("lasso_pmax", x, y, alpha = 1, pmax = 7)       # -> max_active_reached
add("enet_weights", x, y, alpha = 0.6, weights = runif(100, 0.2, 2))
add("lasso_penfactor", x, y, alpha = 1, penalty.factor = c(0, 0, rep(1, 18)))  # 2 unpenalized
add("lasso_penfactor_hetero", x, y, alpha = 1, penalty.factor = runif(20, 0.5, 3))
add("enet_boxconstraint", x, y, alpha = 0.5, lower.limits = -0.3, upper.limits = 0.4)
add("lasso_lowerlimit0", x, y, alpha = 1, lower.limits = 0)  # nonneg lasso

xc <- x; xc[, 7] <- 3.0                             # constant column -> chkvars ju[j]=FALSE
add("lasso_constcol", xc, y, alpha = 1)

xw <- mk(50, 200); yw <- sig(xw)                    # p > n
add("lasso_p200_n50", xw, yw, alpha = 1)
add("enet_p200_n50", xw, yw, alpha = 0.3)

xr <- mk(80, 30, rho = 2); yr <- sig(xr)            # correlated -> strong-rule violations
add("lasso_correlated", xr, yr, alpha = 1)

ynull <- rnorm(100)                                  # y ~ noise -> fdev early stop fires
add("lasso_nullsignal", x, ynull, alpha = 1)

add("lasso_thresh_tight", x, y, alpha = 1, thresh = 1e-12)
add("lasso_nlambda20", x, y, alpha = 1, nlambda = 20)
add("lasso_lmr_tiny", x, y, alpha = 1, lambda.min.ratio = 1e-6)

out <- list()
for (cs in cases) {
  fit <- tryCatch(
    do.call(glmnet, c(list(x = cs$x, y = cs$y, family = "gaussian",
                           type.gaussian = "naive"), cs$args)),
    error = function(e) { message("SKIP ", cs$name, ": ", conditionMessage(e)); NULL })
  if (is.null(fit)) next

  n <- nrow(cs$x); p <- ncol(cs$x)
  a <- cs$args
  g <- function(k, d) if (is.null(a[[k]])) d else a[[k]]

  # scalars that may be given as length-1 and recycled by glmnet
  rec <- function(v, len) if (length(v) == 1) rep(v, len) else v
  dfmax_v <- g("dfmax", p + 1)

  beta <- as.matrix(fit$beta)   # p x lmu
  rec_obj <- list(
    name              = cs$name,
    n                 = n,
    p                 = p,
    x                 = as.vector(cs$x),            # column-major
    y                 = as.vector(cs$y),
    weights           = rec(g("weights", rep(1, n)), n),
    alpha             = g("alpha", 1),
    intercept         = g("intercept", TRUE),
    standardize       = g("standardize", TRUE),
    nlambda           = g("nlambda", 100),
    lambda_min_ratio  = g("lambda.min.ratio", ifelse(n < p, 0.01, 1e-04)),
    # a[["lambda"]] not a$lambda: `$` partial-matches, so a$lambda silently
    # returns lambda.min.ratio when only the latter is set.
    user_lambda       = if (is.null(a[["lambda"]])) NULL else sort(a[["lambda"]], decreasing = TRUE),
    penalty_factor    = rec(g("penalty.factor", rep(1, p)), p),
    lower_limits      = rec(g("lower.limits", rep(-Inf, p)), p),
    upper_limits      = rec(g("upper.limits", rep(Inf, p)), p),
    dfmax             = dfmax_v,
    pmax              = g("pmax", min(dfmax_v * 2 + 20, p)),   # R: pmax defaults off dfmax, not p
    thresh            = g("thresh", 1e-07),
    maxit             = g("maxit", 1e+05),
    # --- expected ---
    lmu               = length(fit$lambda),
    lambda            = as.vector(fit$lambda),
    a0                = as.vector(fit$a0),
    beta              = as.vector(beta),            # column-major p x lmu
    dev_ratio         = as.vector(fit$dev.ratio),
    nulldev           = fit$nulldev,
    npasses           = fit$npasses
  )
  out[[cs$name]] <- rec_obj
  fn <- file.path(outdir, paste0(cs$name, ".json"))
  # digits=NA => full double precision round-trip; anything less breaks 1e-9 parity
  write(toJSON(rec_obj, digits = NA, auto_unbox = TRUE, null = "null"), fn)
  cat(sprintf("%-26s n=%3d p=%3d lmu=%3d npasses=%6d\n", cs$name, n, p, rec_obj$lmu, rec_obj$npasses))
}
cat("\nwrote", length(out), "fixtures to", normalizePath(outdir), "\n")

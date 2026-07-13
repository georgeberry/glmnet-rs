#!/usr/bin/env Rscript
# Reference fixtures for the Poisson (log-link) path.

suppressMessages(library(glmnet))
suppressMessages(library(jsonlite))

set.seed(20260711)
outdir <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "tests", "fixtures")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cases <- list()
add <- function(name, x, y, ...) {
  cases[[length(cases) + 1]] <<- list(name = name, x = x, y = y, args = list(...))
}

mk <- function(n, p, rho = 0) {
  x <- matrix(rnorm(n * p), n, p)
  if (rho > 0) x <- x + rho * matrix(rnorm(n), n, p)
  x
}
# Poisson counts from a sparse log-linear signal. b0 keeps the mean modest so
# counts don't explode.
mky <- function(x, k = 4, b = 0.4, b0 = 0.2) {
  eta <- b0 + as.vector(x[, seq_len(min(k, ncol(x)))] %*% rep(b, min(k, ncol(x))))
  rpois(nrow(x), exp(eta))
}

x <- mk(200, 20); y <- mky(x)
add("pois_lasso_n200_p20", x, y, alpha = 1)
add("pois_enet_a50", x, y, alpha = 0.5)
add("pois_ridge_a0", x, y, alpha = 0)
add("pois_nostd", x, y, alpha = 1, standardize = FALSE)
add("pois_nointercept", x, y, alpha = 1, intercept = FALSE)
add("pois_nostd_nointercept", x, y, alpha = 1, standardize = FALSE, intercept = FALSE)
add("pois_userlambda", x, y, alpha = 1, lambda = exp(seq(log(0.3), log(0.002), length.out = 30)))
add("pois_dfmax", x, y, alpha = 1, dfmax = 5)
add("pois_pmax", x, y, alpha = 1, pmax = 8)
add("pois_penfactor", x, y, alpha = 1, penalty.factor = c(0, 0, rep(1, 18)))
add("pois_penfactor_hetero", x, y, alpha = 1, penalty.factor = runif(20, 0.5, 3))
add("pois_boxconstraint", x, y, alpha = 0.5, lower.limits = -0.2, upper.limits = 0.3)
add("pois_nonneg", x, y, alpha = 1, lower.limits = 0)
add("pois_nlambda30", x, y, alpha = 1, nlambda = 30)
add("pois_lmr_tiny", x, y, alpha = 1, lambda.min.ratio = 1e-5)

xw <- mk(150, 25); yw <- mky(xw, b0 = -0.2)
add("pois_weights", xw, yw, alpha = 0.7, weights = runif(150, 0.3, 2))

xc <- x; xc[, 5] <- 2.0
add("pois_constcol", xc, y, alpha = 1)

xp <- mk(80, 150); yp <- mky(xp)                 # p > n
add("pois_p150_n80", xp, yp, alpha = 1)

xr <- mk(150, 30, rho = 1.2); yr <- mky(xr)      # correlated -> strong-rule stress
add("pois_correlated", xr, yr, alpha = 1)

xz <- mk(200, 15); yz <- mky(xz, b0 = -1.0)      # many zero counts (low rate)
add("pois_lowrate", xz, yz, alpha = 1)

out <- list()
for (cs in cases) {
  fit <- tryCatch(
    do.call(glmnet, c(list(x = cs$x, y = cs$y, family = "poisson"), cs$args)),
    error = function(e) { message("SKIP ", cs$name, ": ", conditionMessage(e)); NULL })
  if (is.null(fit)) next

  n <- nrow(cs$x); p <- ncol(cs$x)
  a <- cs$args
  g <- function(k, d) if (is.null(a[[k]])) d else a[[k]]
  rec <- function(v, len) if (length(v) == 1) rep(v, len) else v
  dfmax_v <- g("dfmax", p + 1)

  beta <- as.matrix(fit$beta)
  rec_obj <- list(
    name = cs$name, n = n, p = p,
    x = as.vector(cs$x), y = as.numeric(cs$y),
    weights = rec(g("weights", rep(1, n)), n),
    alpha = g("alpha", 1),
    intercept = g("intercept", TRUE),
    standardize = g("standardize", TRUE),
    nlambda = g("nlambda", 100),
    lambda_min_ratio = g("lambda.min.ratio", ifelse(n < p, 0.01, 1e-04)),
    user_lambda = if (is.null(a[["lambda"]])) NULL else sort(a[["lambda"]], decreasing = TRUE),
    penalty_factor = rec(g("penalty.factor", rep(1, p)), p),
    lower_limits = rec(g("lower.limits", rep(-Inf, p)), p),
    upper_limits = rec(g("upper.limits", rep(Inf, p)), p),
    dfmax = dfmax_v,
    pmax = g("pmax", min(dfmax_v * 2 + 20, p)),
    thresh = g("thresh", 1e-07),
    maxit = g("maxit", 1e+05),
    # --- expected ---
    lmu = length(fit$lambda),
    lambda = as.vector(fit$lambda),
    a0 = as.vector(fit$a0),
    beta = as.vector(beta),
    dev_ratio = as.vector(fit$dev.ratio),
    nulldev = fit$nulldev,
    npasses = fit$npasses
  )
  out[[cs$name]] <- rec_obj
  write(toJSON(rec_obj, digits = NA, auto_unbox = TRUE, null = "null"),
        file.path(outdir, paste0(cs$name, ".json")))
  cat(sprintf("%-24s n=%3d p=%3d lmu=%3d npasses=%7d dev<=%.3f\n",
              cs$name, n, p, rec_obj$lmu, rec_obj$npasses, tail(rec_obj$dev_ratio, 1)))
}
cat("\nwrote", length(out), "poisson fixtures\n")

#!/usr/bin/env Rscript
# Reference fixtures for the sparse Gaussian path. glmnet has its own sparse code
# path (spelnet), so this validates the Rust sparse solver against R directly,
# not just against our own dense solver.
#
# The design matrix is a genuinely sparse dgCMatrix; we record its CSC arrays
# (@p column pointers, @i row indices, @x values) so the Rust side reconstructs
# exactly the same matrix.

suppressMessages(library(glmnet))
suppressMessages(library(Matrix))
suppressMessages(library(jsonlite))

set.seed(20260712)
outdir <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "tests", "fixtures")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cases <- list()
add <- function(name, X, y, ...) {
  cases[[length(cases) + 1]] <<- list(name = name, X = X, y = y, args = list(...))
}

# Sparse X with a given density; signal from the first few columns.
mk <- function(n, p, density = 0.1) {
  X <- rsparsematrix(n, p, density = density)
  X
}
sig <- function(X, k = 5, sd = 1) {
  b <- c(rep(1.5, min(k, ncol(X))), rep(0, max(0, ncol(X) - k)))
  as.vector(X %*% b) + rnorm(nrow(X), sd = sd)
}

X <- mk(200, 40, 0.15); y <- sig(X)
add("sp_lasso_n200_p40", X, y, alpha = 1)
add("sp_enet_a50", X, y, alpha = 0.5)
add("sp_ridge_a0", X, y, alpha = 0)
add("sp_nostd", X, y, alpha = 1, standardize = FALSE)
add("sp_nointercept", X, y, alpha = 1, intercept = FALSE)
add("sp_nostd_nointercept", X, y, alpha = 1, standardize = FALSE, intercept = FALSE)
add("sp_userlambda", X, y, alpha = 1, lambda = exp(seq(log(1), log(0.01), length.out = 30)))
add("sp_penfactor", X, y, alpha = 1, penalty.factor = c(0, 0, runif(38, 0.5, 2)))
add("sp_weights", X, y, alpha = 0.6, weights = runif(200, 0.3, 2))
add("sp_boxconstraint", X, y, alpha = 0.5, lower.limits = -0.3, upper.limits = 0.4)

Xs <- mk(150, 500, 0.03); ys <- sig(Xs)              # very sparse, p > n
add("sp_wide_p500_n150", Xs, ys, alpha = 1)

Xd <- mk(300, 30, 0.4); yd <- sig(Xd)                # denser
add("sp_dense_ish", Xd, yd, alpha = 1)

out <- list()
for (cs in cases) {
  X <- cs$X; y <- cs$y
  n <- nrow(X); p <- ncol(X)
  fit <- tryCatch(
    do.call(glmnet, c(list(x = X, y = y, family = "gaussian"), cs$args)),
    error = function(e) { message("SKIP ", cs$name, ": ", conditionMessage(e)); NULL })
  if (is.null(fit)) next

  a <- cs$args
  g <- function(k, d) if (is.null(a[[k]])) d else a[[k]]
  rec <- function(v, len) if (length(v) == 1) rep(v, len) else v
  dfmax_v <- g("dfmax", p + 1)

  Xc <- as(X, "CsparseMatrix")   # ensure dgCMatrix CSC layout
  beta <- as.matrix(fit$beta)
  rec_obj <- list(
    name = cs$name, n = n, p = p,
    # CSC arrays (0-based, as stored in dgCMatrix)
    col_ptr = as.integer(Xc@p),
    row_idx = as.integer(Xc@i),
    values  = as.numeric(Xc@x),
    y = as.numeric(y),
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
  cat(sprintf("%-22s n=%3d p=%4d nnz=%6d lmu=%3d npasses=%6d\n",
              cs$name, n, p, length(Xc@x), rec_obj$lmu, rec_obj$npasses))
}
cat("\nwrote", length(out), "sparse fixtures\n")

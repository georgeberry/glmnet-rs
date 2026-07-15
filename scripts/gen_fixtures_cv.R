#!/usr/bin/env Rscript
# Reference fixtures for cross-validation. A fixed foldid makes cv.glmnet fully
# deterministic, so cvm/cvsd/lambda.min/lambda.1se can be compared to the Python
# implementation given the same folds.

suppressMessages(library(glmnet))
suppressMessages(library(jsonlite))

set.seed(20260713)
outdir <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "tests", "fixtures")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cases <- list()
add <- function(name, X, y, family, measure, foldid, ...) {
  cases[[length(cases) + 1]] <<- list(name = name, X = X, y = y, family = family,
                                      measure = measure, foldid = foldid, args = list(...))
}

nfolds <- 8
mkfold <- function(n) sample(rep(seq_len(nfolds), length.out = n))

n <- 200; p <- 20
x <- matrix(rnorm(n * p), n, p)
yg <- as.vector(x[, 1:5] %*% rep(1.5, 5)) + rnorm(n)
fid <- mkfold(n)
add("cv_gaussian_mse", x, yg, "gaussian", "mse", fid, alpha = 1)
add("cv_gaussian_mae", x, yg, "gaussian", "mae", fid, alpha = 1)
add("cv_gaussian_enet", x, yg, "gaussian", "mse", fid, alpha = 0.5)

pb <- 1 / (1 + exp(-(x[, 1:4] %*% rep(1.2, 4))))
yb <- rbinom(n, 1, pb)
fidb <- mkfold(n)
add("cv_binomial_deviance", x, yb, "binomial", "deviance", fidb, alpha = 1)
add("cv_binomial_class", x, yb, "binomial", "class", fidb, alpha = 1)
add("cv_binomial_mse", x, yb, "binomial", "mse", fidb, alpha = 1)

yp <- rpois(n, exp(0.2 + x[, 1:4] %*% rep(0.4, 4)))
fidp <- mkfold(n)
add("cv_poisson_deviance", x, yp, "poisson", "deviance", fidp, alpha = 1)
add("cv_poisson_mse", x, yp, "poisson", "mse", fidp, alpha = 0.7)

out <- list()
for (cs in cases) {
  cv <- tryCatch(
    do.call(cv.glmnet, c(list(x = cs$X, y = cs$y, family = cs$family,
                              type.measure = cs$measure, foldid = cs$foldid), cs$args)),
    error = function(e) { message("SKIP ", cs$name, ": ", conditionMessage(e)); NULL })
  if (is.null(cv)) next
  n <- nrow(cs$X); p <- ncol(cs$X)
  a <- cs$args
  g <- function(k, d) if (is.null(a[[k]])) d else a[[k]]

  rec_obj <- list(
    name = cs$name, n = n, p = p,
    x = as.vector(cs$X), y = as.numeric(cs$y),
    family = cs$family, measure = cs$measure,
    foldid0 = as.integer(cs$foldid - 1L),   # 0-based for Python
    nfolds = nfolds,
    alpha = g("alpha", 1),
    # --- expected ---
    lambda = as.vector(cv$lambda),
    cvm = as.vector(cv$cvm),
    cvsd = as.vector(cv$cvsd),
    nzero = as.integer(cv$nzero),
    lambda_min = cv$lambda.min,
    lambda_1se = cv$lambda.1se
  )
  out[[cs$name]] <- rec_obj
  write(toJSON(rec_obj, digits = NA, auto_unbox = TRUE, null = "null"),
        file.path(outdir, paste0(cs$name, ".json")))
  cat(sprintf("%-22s %-9s %-9s lmu=%3d lam.min=%.4f lam.1se=%.4f\n",
              cs$name, cs$family, cs$measure, length(cv$lambda), cv$lambda.min, cv$lambda.1se))
}
cat("\nwrote", length(out), "cv fixtures\n")

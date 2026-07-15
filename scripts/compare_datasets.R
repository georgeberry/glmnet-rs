#!/usr/bin/env Rscript
# R side of the real-dataset comparison. Fits glmnet on each dataset and writes
# the lambda path, coefficients, and fit time so the Python side can compare
# coefficients and wall clock. Reads the same CSVs the Python side reads.

suppressMessages(library(glmnet))
suppressMessages(library(jsonlite))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
data_dir <- file.path(here, "..", "datasets")
out_dir <- file.path(here, "..", "datasets", ".compare")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

time_fit <- function(fun, reps = 5) {
  fun() # warmup
  best <- Inf
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    fit <- fun()
    best <- min(best, proc.time()[["elapsed"]] - t0)
  }
  list(fit = fun(), time = best)
}

dump <- function(name, fit, time) {
  write(toJSON(list(
    lmu = length(fit$lambda),
    lambda = as.vector(fit$lambda),
    a0 = as.vector(fit$a0),
    beta = as.vector(as.matrix(fit$beta)), # p x lmu, column-major
    p = nrow(fit$beta),
    time = time,
    npasses = fit$npasses
  ), digits = NA, auto_unbox = TRUE), file.path(out_dir, paste0(name, ".json")))
}

# --- wine quality (long, gaussian) -----------------------------------------
w <- read.csv(file.path(data_dir, "winequality-white.csv"), sep = ";")
Xw <- as.matrix(w[, setdiff(names(w), "quality")])
yw <- w$quality
rw <- time_fit(function() glmnet(Xw, yw, family = "gaussian", type.gaussian = "naive"))
dump("wine", rw$fit, rw$time)
cat(sprintf("wine     n=%d p=%d  lmu=%d  R fit=%.1f ms\n",
            nrow(Xw), ncol(Xw), length(rw$fit$lambda), rw$time * 1e3))

# --- leukemia (wide, binomial) ---------------------------------------------
lab <- as.character(read.csv(file.path(data_dir, "leukemia_big.csv"), header = FALSE, nrows = 1))
expr <- as.matrix(read.csv(file.path(data_dir, "leukemia_big.csv"), header = FALSE, skip = 1))
Xl <- t(expr) # samples x genes
yl <- as.integer(lab == "AML")
rl <- time_fit(function() glmnet(Xl, yl, family = "binomial", type.logistic = "Newton"))
dump("leukemia", rl$fit, rl$time)
cat(sprintf("leukemia n=%d p=%d  lmu=%d  R fit=%.1f ms\n",
            nrow(Xl), ncol(Xl), length(rl$fit$lambda), rl$time * 1e3))

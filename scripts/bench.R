#!/usr/bin/env Rscript
# R side of the benchmark. Reads the raw f64 matrices bench.py wrote and times
# glmnet on the full path with matching solver settings. Emits a single JSON
# line (name -> best time in seconds) on the last line of stdout.

suppressMessages(library(glmnet))
suppressMessages(library(jsonlite))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
scratch <- file.path(here, ".bench_data")
cases <- fromJSON(file.path(scratch, "cases.json"))

REPEATS <- 7

read_mat <- function(path, n, p) {
  con <- file(path, "rb")
  on.exit(close(con))
  v <- readBin(con, what = "double", n = n * p, size = 8, endian = "little")
  matrix(v, nrow = n, ncol = p) # data was written column-major
}

times <- list()
for (nm in cases) {
  meta <- fromJSON(file.path(scratch, paste0(nm, "_meta.json")))
  n <- meta$n; p <- meta$p; fam <- meta$family
  X <- read_mat(file.path(scratch, paste0(nm, "_X.bin")), n, p)
  y <- readBin(file.path(scratch, paste0(nm, "_y.bin")), what = "double",
               n = n, size = 8, endian = "little")

  # Match the Rust port's solver: naive gaussian, Newton logistic.
  fitfun <- if (fam == "gaussian") {
    function() glmnet(X, y, family = "gaussian", type.gaussian = "naive")
  } else if (fam == "poisson") {
    function() glmnet(X, y, family = "poisson")
  } else {
    function() glmnet(X, y, family = "binomial", type.logistic = "Newton")
  }

  fitfun() # warmup
  best <- Inf
  for (i in seq_len(REPEATS)) {
    t0 <- proc.time()[["elapsed"]]
    fitfun()
    best <- min(best, proc.time()[["elapsed"]] - t0)
  }
  times[[nm]] <- best
  message(sprintf("  [R]    %-12s n=%-6d p=%-5d %-9s %8.2f ms",
                  nm, n, p, fam, best * 1e3))
}

cat(toJSON(times, auto_unbox = TRUE), "\n")

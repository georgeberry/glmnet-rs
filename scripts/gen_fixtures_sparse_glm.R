#!/usr/bin/env Rscript
# Sparse binomial + poisson fixtures from R's sparse path (splognet/spfishnet on
# a dgCMatrix), recording CSC arrays and expected output.
suppressMessages(library(glmnet)); suppressMessages(library(Matrix)); suppressMessages(library(jsonlite))
set.seed(20260714)
outdir <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])), "..", "tests", "fixtures")
cases <- list()
add <- function(name, X, y, family, ...) cases[[length(cases)+1]] <<- list(name=name, X=X, y=y, family=family, args=list(...))

mkx <- function(n,p,d=0.12) rsparsematrix(n,p,density=d)
linb <- function(X,k=4,b=1.2,b0=0) as.vector(X[,1:min(k,ncol(X))] %*% rep(b,min(k,ncol(X)))) + b0

Xb <- mkx(200,30); etab <- linb(Xb); yb <- rbinom(200,1,1/(1+exp(-etab)))
add("spb_lasso", Xb, yb, "binomial", alpha=1)
add("spb_enet", Xb, yb, "binomial", alpha=0.5)
add("spb_ridge", Xb, yb, "binomial", alpha=0)
add("spb_nostd", Xb, yb, "binomial", alpha=1, standardize=FALSE)
add("spb_noint", Xb, yb, "binomial", alpha=1, intercept=FALSE)
add("spb_penf", Xb, yb, "binomial", alpha=1, penalty.factor=c(0,0,runif(28,.5,2)))
add("spb_wide", mkx(120,300,0.04), NA, "binomial", alpha=1)  # filled below

# fix spb_wide y
cases[[length(cases)]]$y <- rbinom(120,1,1/(1+exp(-linb(cases[[length(cases)]]$X))))

Xp <- mkx(200,30); etap <- 0.2+linb(Xp,4,0.4); yp <- rpois(200, exp(etap))
add("spp_lasso", Xp, yp, "poisson", alpha=1)
add("spp_enet", Xp, yp, "poisson", alpha=0.5)
add("spp_ridge", Xp, yp, "poisson", alpha=0)
add("spp_nostd", Xp, yp, "poisson", alpha=1, standardize=FALSE)
add("spp_noint", Xp, yp, "poisson", alpha=1, intercept=FALSE)
add("spp_penf", Xp, yp, "poisson", alpha=1, penalty.factor=c(0,0,runif(28,.5,2)))

n_written <- 0
for (cs in cases) {
  X <- cs$X; y <- cs$y; n <- nrow(X); p <- ncol(X)
  fit <- tryCatch(do.call(glmnet, c(list(x=X, y=y, family=cs$family), cs$args)),
                  error=function(e){message("SKIP ",cs$name,": ",conditionMessage(e)); NULL})
  if (is.null(fit)) next
  a <- cs$args; g <- function(k,d) if(is.null(a[[k]])) d else a[[k]]
  rec <- function(v,len) if(length(v)==1) rep(v,len) else v
  dfmax_v <- g("dfmax", p+1)
  Xc <- as(X, "CsparseMatrix")
  ro <- list(name=cs$name, n=n, p=p, family=cs$family,
    col_ptr=as.integer(Xc@p), row_idx=as.integer(Xc@i), values=as.numeric(Xc@x),
    y=as.numeric(y), weights=rec(g("weights",rep(1,n)),n), alpha=g("alpha",1),
    intercept=g("intercept",TRUE), standardize=g("standardize",TRUE),
    nlambda=g("nlambda",100), lambda_min_ratio=g("lambda.min.ratio", ifelse(n<p,0.01,1e-4)),
    user_lambda=if(is.null(a[["lambda"]])) NULL else sort(a[["lambda"]],decreasing=TRUE),
    penalty_factor=rec(g("penalty.factor",rep(1,p)),p),
    lower_limits=rec(g("lower.limits",rep(-Inf,p)),p), upper_limits=rec(g("upper.limits",rep(Inf,p)),p),
    dfmax=dfmax_v, pmax=g("pmax",min(dfmax_v*2+20,p)), thresh=g("thresh",1e-7), maxit=g("maxit",1e5),
    lmu=length(fit$lambda), lambda=as.vector(fit$lambda), a0=as.vector(fit$a0),
    beta=as.vector(as.matrix(fit$beta)), dev_ratio=as.vector(fit$dev.ratio), npasses=fit$npasses)
  write(toJSON(ro, digits=NA, auto_unbox=TRUE, null="null"), file.path(outdir, paste0(cs$name,".json")))
  cat(sprintf("%-12s %-9s n=%3d p=%4d lmu=%3d npasses=%6d\n", cs$name, cs$family, n, p, ro$lmu, ro$npasses))
  n_written <- n_written + 1
}
cat("\nwrote", n_written, "sparse GLM fixtures\n")

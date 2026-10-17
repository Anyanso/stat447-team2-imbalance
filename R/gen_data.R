# =============================================================
# gen_data.R  --  OWNER: Person 1   (Step 1)
# Written in the base-R syntax of the ISLR (R edition, 2nd ed.) labs.
# Page numbers below are the BOOK's printed page numbers.
# =============================================================
# THE TRUE MODEL (we play nature -- class simulation deck S.3)
#
#   x1, ..., x5 ~ independent N(0, 1)
#   log-odds:  eta = beta0 + sep * (x1 - x2 + 0.5 x3 - 0.5 x4 + 0 x5)
#   P(y = 1 | x) = exp(eta) / (1 + exp(eta))      (ISLR eq. 4.2, p. 134)
#   y = 1 with probability P(y = 1 | x), else 0
#
# - The SLOPES never change with prevalence. Only beta0 does, so the
#   x -> log-odds relationship stays fixed (project handout, Team 2).
# - sep (separation) scales the slopes: bigger sep = easier to separate.
# - x5 has true slope 0: a pure-noise predictor, like a useless real column.
# =============================================================

beta <- c(1, -1, 0.5, -0.5, 0)   # direction of the true slopes

# -------------------------------------------------------------
# find_beta0(prev, sep): the intercept that gives the target prevalence
# -------------------------------------------------------------
# Prevalence = average of P(y = 1 | x) over the population of x.
# We approximate that population with ONE big reference sample of the
# linear part  sep * x'beta.  Since the x's are independent N(0, 1),
# x'beta is N(0, sum(beta^2)), so we only need one rnorm() vector.
#
# Prevalence goes UP as beta0 goes up, so we can find beta0 by
# bisection: keep halving the interval [lo, hi] that contains it.
# 50 halvings pin it down far more precisely than we need.
# -------------------------------------------------------------
find_beta0 <- function(prev, sep, n.ref = 1e6) {
  lin <- sep * sqrt(sum(beta^2)) * rnorm(n.ref)   # reference sample of sep * x'beta
  lo <- -20
  hi <- 20
  for (i in 1:50) {                               # for loop: ISLR p. 274
    mid <- (lo + hi) / 2
    prev.mid <- mean(exp(mid + lin) / (1 + exp(mid + lin)))
    if (prev.mid > prev) {
      hi <- mid      # too many positives: beta0 must be smaller
    } else {
      lo <- mid      # too few positives: beta0 must be larger
    }
  }
  (lo + hi) / 2
}

# -------------------------------------------------------------
# gen_data(n, prev, sep, beta0): one simulated data set
# -------------------------------------------------------------
# RETURNS a data frame:  x1 ... x5,  y (factor "0"/"1"),  p.true
# p.true is the TRUE P(y = 1 | x). The methods never see it; we use it
# only to grade them (the Bayes benchmark, Step 4).
# -------------------------------------------------------------
gen_data <- function(n, prev, sep, beta0) {
  p <- length(beta)
  x <- matrix(rnorm(n * p), ncol = p)             # ISLR p. 389
  colnames(x) <- paste0("x", 1:p)

  eta <- beta0 + sep * (x %*% beta)               # true log-odds
  p.true <- exp(eta) / (1 + exp(eta))             # logistic function, ISLR p. 316
  y <- ifelse(runif(n) < p.true, 1, 0)            # runif: p. 127; ifelse: p. 353

  data.frame(x,                                   # data frame + factor y: p. 389
             y = factor(y, levels = c(0, 1)),     # keep both levels even if no 1s
             p.true = as.vector(p.true))
}

# -------------------------------------------------------------
# make_beta0_table(): one beta0 per (prevalence, separation)
# Run ONCE, right after set.seed(SEED), and save it -- every replicate
# then reuses the same intercepts.
# -------------------------------------------------------------
make_beta0_table <- function(prevs = PREVALENCES, seps = SEPARATION) {
  tab <- expand.grid(prev = prevs, sep.level = names(seps),
                     stringsAsFactors = FALSE)
  tab$sep <- seps[tab$sep.level]
  tab$beta0 <- NA
  for (i in 1:nrow(tab)) {
    tab$beta0[i] <- find_beta0(tab$prev[i], tab$sep[i])
  }
  tab
}

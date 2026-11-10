# gen_data.R -- simulating the data (John Anyanso)
#
# True model:
#   x1, ..., x5 independent N(0, 1)
#   log-odds = beta0 + sep * (x1 - x2 + 0.5 x3 - 0.5 x4 + 0 x5)
#   P(y = 1 | x) = exp(log-odds) / (1 + exp(log-odds))   (ISLR eq. 4.2)
#
# Only beta0 changes with the prevalence; the slopes stay fixed, so a
# rarer class is not also a harder problem. sep scales the slopes, and
# x5 has slope 0 (a noise variable).

beta <- c(1, -1, 0.5, -0.5, 0)

# Intercept that gives the target prevalence.
# Prevalence is the average of P(y = 1 | x) over all x. Since the x's are
# independent N(0, 1), x'beta is N(0, sum(beta^2)), so one big rnorm()
# sample stands in for the population. Prevalence increases with beta0,
# so we find beta0 by bisection.
find_beta0 <- function(prev, sep, n.ref = 1e6) {
  lin <- sep * sqrt(sum(beta^2)) * rnorm(n.ref)
  lo <- -20
  hi <- 20
  for (i in 1:50) {
    mid <- (lo + hi) / 2
    prev.mid <- mean(exp(mid + lin) / (1 + exp(mid + lin)))
    if (prev.mid > prev) {
      hi <- mid      # too many positives, go lower
    } else {
      lo <- mid      # too few positives, go higher
    }
  }
  (lo + hi) / 2
}

# One simulated data set: x1-x5, y (factor "0"/"1") and p.true.
# p.true is the true probability. The models never see it; we only use
# it for the Bayes benchmark.
gen_data <- function(n, prev, sep, beta0) {
  p <- length(beta)
  x <- matrix(rnorm(n * p), ncol = p)
  colnames(x) <- paste0("x", 1:p)

  eta <- beta0 + sep * (x %*% beta)
  p.true <- exp(eta) / (1 + exp(eta))
  y <- ifelse(runif(n) < p.true, 1, 0)

  data.frame(x,
             y = factor(y, levels = c(0, 1)),     # keep both levels even with no 1s
             p.true = as.vector(p.true))
}

# beta0 for every prevalence x separation. Made once after set.seed(SEED)
# and saved, so all replicates use the same intercepts.
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

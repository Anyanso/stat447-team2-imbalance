# =============================================================
# gen_data.R  --  OWNER: Person 1   (built in Step 1)
# =============================================================
# CONTRACT (agreed on day 1 -- do not change without telling the team)
#
# gen_data(n, prev, sep, p = 5)
#   n    : number of rows
#   prev : target minority prevalence, e.g. 0.05
#   sep  : class separation, multiplies the true slopes
#   p    : number of predictors
#
# RETURNS a tibble with columns
#   x1 ... xp  : numeric predictors
#   y          : factor with levels c("0", "1")   ("1" = minority/positive)
#   p_true     : TRUE P(y = 1 | x)  -> used for the Bayes benchmark
#
# Imbalance is controlled ONLY through the intercept beta0 (found by uniroot),
# so the x -> log-odds relationship stays fixed (handout, Team 2).
# =============================================================

gen_data <- function(n, prev, sep, p = 5) {
  stop("gen_data(): not built yet -- Step 1")
}

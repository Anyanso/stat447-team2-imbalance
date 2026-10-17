# =============================================================
# approaches.R  --  the three approach families (built in Step 3)
#   Approach A, resampling ........ OWNER: Person 1
#   Approach B, class weights ..... OWNER: Person 2
#   Approach C, threshold ......... OWNER: Person 3
# =============================================================
# CONTRACT (agreed on day 1)
#
# fit_and_predict(train, test, approach, classifier)
#   train, test : tibbles with predictors, and y = factor c("0","1")
#   approach    : one of APPROACHES  (see R/00_setup.R)
#   classifier  : "logistic" (cv.glmnet) or "forest" (ranger)
#
# RETURNS list(
#   prob   = numeric vector, P(y = 1) for every row of `test`,
#   cutoff = number used to turn prob into a class (0.5 unless approach = "threshold")
# )
#
# TWO RULES (handout + Lecture 9):
#   1. Resample / reweight the TRAINING data only. Never touch `test`.
#   2. Tune anything (lambda, cutoff) using TRAINING data only.
# =============================================================

fit_and_predict <- function(train, test, approach, classifier) {
  stop("fit_and_predict(): not built yet -- Step 3")
}

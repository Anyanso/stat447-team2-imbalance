# =============================================================
# evaluate.R  --  OWNER: Person 3   (built in Step 4)
# =============================================================
# CONTRACT (agreed on day 1)
#
# evaluate(y, prob, cutoff = 0.5)
#   y      : factor c("0","1") -- true classes of the test set
#   prob   : numeric, predicted P(y = 1)
#   cutoff : classify as "1" when prob > cutoff
#
# RETURNS a one-row tibble:
#   sensitivity, specificity, precision, f1, bal_accuracy, accuracy,
#   roc_auc, pr_auc
#
# Bayes benchmark = evaluate(y, p_true) using the TRUE probabilities
# from gen_data(). No method can beat it on average.
# =============================================================

evaluate <- function(y, prob, cutoff = 0.5) {
  stop("evaluate(): not built yet -- Step 4")
}

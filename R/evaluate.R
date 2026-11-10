# evaluate.R -- scoring the predictions (Nidhi Prajapati)
# evaluate(y, prob, cutoff) returns a one-row data frame of metrics.
#   y      true classes, factor "0"/"1"
#   prob   predicted P(y = 1)
#   cutoff predict "1" when prob > cutoff
# Definitions as in ISLR Tables 4.6 and 4.7.

evaluate <- function(y, prob, cutoff = 0.5) {

  pred <- ifelse(prob > cutoff, "1", "0")

  TP <- sum(pred == "1" & y == "1")   # fraud caught
  FN <- sum(pred == "0" & y == "1")   # fraud missed
  FP <- sum(pred == "1" & y == "0")   # false alarm
  TN <- sum(pred == "0" & y == "0")   # normal, correctly cleared

  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision   <- if (TP + FP > 0) TP / (TP + FP) else NA   # undefined if nothing is flagged
  f1          <- if (is.na(precision) || precision + sensitivity == 0) 0 else
                   2 * precision * sensitivity / (precision + sensitivity)
  bal.accuracy <- (sensitivity + specificity) / 2
  accuracy     <- (TP + TN) / length(y)         # kept to show how misleading it is

  # ROC-AUC and PR-AUC look at the ranking over all cutoffs
  rocr   <- prediction(prob, y)
  roc.auc <- performance(rocr, "auc")@y.values[[1]]
  pr.auc  <- performance(rocr, "aucpr")@y.values[[1]]

  data.frame(sensitivity, specificity, precision, f1, bal.accuracy, accuracy,
             roc.auc, pr.auc, cutoff,
             n.flagged = TP + FP)
}

# Bayes benchmark: score the true probabilities, the best any method can do.
# The cutoff is the prevalence, which for the true probabilities maximises
# sensitivity + specificity. (The usual Bayes classifier cuts at 0.5 and
# minimises overall error, which is the wrong goal here.)
bayes_benchmark <- function(test, prev) {
  evaluate(test$y, test$p.true, cutoff = prev)
}

# =============================================================
# evaluate.R  --  OWNER: Person 3   (Step 4)
# The grading function every approach is scored with.
# Definitions follow ISLR Tables 4.6-4.7 (pp. 152-153) and Lecture 6.
# =============================================================
# CONTRACT
#   evaluate(y, prob, cutoff = 0.5)
#     y      : factor c("0","1"), the TRUE classes of the test set
#     prob   : predicted P(y = 1) for each test row
#     cutoff : classify as "1" when prob > cutoff
#   -> one-row data frame of metrics
# =============================================================

evaluate <- function(y, prob, cutoff = 0.5) {

  # ---- Block 1: turn probabilities into classes ----
  # Same idea as the ISLR Ch. 4 lab, p. 174:
  #   glm.pred[glm.probs > .5] <- "Up"
  pred <- ifelse(prob > cutoff, "1", "0")

  # ---- Block 2: the confusion matrix, as four counts (ISLR Table 4.6) ----
  TP <- sum(pred == "1" & y == "1")   # fraud, flagged        (true positive)
  FN <- sum(pred == "0" & y == "1")   # fraud, missed         (false negative)
  FP <- sum(pred == "1" & y == "0")   # normal, flagged       (false positive)
  TN <- sum(pred == "0" & y == "0")   # normal, cleared       (true negative)

  # ---- Block 3: the threshold metrics (ISLR Table 4.7) ----
  sensitivity <- TP / (TP + FN)                 # share of frauds caught (recall)
  specificity <- TN / (TN + FP)                 # share of normals cleared
  precision   <- if (TP + FP > 0) TP / (TP + FP) else NA   # of those flagged, share real
  f1          <- if (is.na(precision) || precision + sensitivity == 0) 0 else
                   2 * precision * sensitivity / (precision + sensitivity)
  bal.accuracy <- (sensitivity + specificity) / 2
  accuracy     <- (TP + TN) / length(y)         # reported only to show it misleads

  # ---- Block 4: the threshold-free metrics (ROCR, ISLR 9.6.3, p. 394) ----
  # These judge the RANKING of the probabilities over ALL cutoffs at once.
  rocr   <- prediction(prob, y)
  roc.auc <- performance(rocr, "auc")@y.values[[1]]    # area under ROC curve
  pr.auc  <- performance(rocr, "aucpr")@y.values[[1]]  # area under precision-recall curve

  data.frame(sensitivity, specificity, precision, f1, bal.accuracy, accuracy,
             roc.auc, pr.auc, cutoff,
             n.flagged = TP + FP)
}

# -------------------------------------------------------------
# bayes_benchmark(test, prev): the best ANY method could do
# -------------------------------------------------------------
# Uses the TRUE probabilities p.true, which only exist in a simulation.
# - ROC-AUC and PR-AUC: the true probabilities give the best possible
#   ranking (no method can beat them on average).
# - Cutoff = prevalence: for the TRUE probabilities this is the rule that
#   maximizes sensitivity + specificity (balanced accuracy).  Flag x when
#   p(x) > prev, i.e. when x is more typical of positives than negatives.
#   (The ordinary Bayes classifier uses 0.5 and minimizes OVERALL error,
#   Lecture 5 -- which is exactly the goal we argue is wrong here.)
# -------------------------------------------------------------
bayes_benchmark <- function(test, prev) {
  evaluate(test$y, test$p.true, cutoff = prev)
}

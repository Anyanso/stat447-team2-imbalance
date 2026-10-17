# =============================================================
# 04_check_evaluate.R  --  Step 4: is the grading function right?
# =============================================================
source("R/00_setup.R")

# ---- Check 1: reproduce the textbook's numbers ----
# ISLR Table 4.4 (p. 148), LDA on the Default data:
#                 True No   True Yes
#   Predicted No    9644       252
#   Predicted Yes     23        81
# Textbook / Lecture 6 values: sensitivity 81/333 = 0.243,
# specificity 9644/9667 = 0.998, precision 81/104 = 0.779, error 2.75%.
y    <- factor(c(rep("0", 9644), rep("1", 252), rep("0", 23), rep("1", 81)),
               levels = c("0", "1"))
prob <- c(rep(0.1, 9644 + 252), rep(0.9, 23 + 81))   # flagged rows get 0.9
m <- evaluate(y, prob, cutoff = 0.5)
cat("Check 1: textbook Table 4.4\n")
print(round(m[, c("sensitivity", "specificity", "precision", "accuracy")], 4),
      row.names = FALSE)
stopifnot(abs(m$sensitivity - 81/333) < 1e-9,
          abs(m$specificity - 9644/9667) < 1e-9,
          abs(m$precision - 81/104) < 1e-9,
          abs(1 - m$accuracy - 0.0275) < 1e-9)
cat("matches the textbook\n\n")

# ---- Check 2: the useless "always say not fraud" classifier ----
# Handout: "a model predicting negative for everyone scores 99% accuracy".
y2 <- factor(c(rep("0", 990), rep("1", 10)), levels = c("0", "1"))
m2 <- evaluate(y2, rep(0, 1000))
cat("Check 2: always predict 'not fraud' (1% positives)\n")
print(m2[, c("accuracy", "sensitivity", "precision", "f1", "bal.accuracy", "roc.auc")],
      row.names = FALSE)
cat("\n")

# ---- Check 3: grade one replicate, every approach + the Bayes benchmark ----
design <- readRDS(file.path(RESULTS_DIR, "design.rds"))
row <- design[design$prev == 0.05 & design$n == 1000 &
              design$sep.level == "high" & design$rep.id == 1, ]
set.seed(row$seed)
train <- gen_data(row$n,  row$prev, row$sep, row$beta0)
test  <- gen_data(N_TEST, row$prev, row$sep, row$beta0)

out <- cbind(classifier = "truth", approach = "BAYES",
             bayes_benchmark(test, row$prev))
for (cl in CLASSIFIERS) {
  for (ap in APPROACHES) {
    res <- fit_and_predict(train, test, ap, cl)
    out <- rbind(out, cbind(classifier = cl, approach = ap,
                            evaluate(test$y, res$prob, res$cutoff)))
  }
}
cat("Check 3:", row$setting, "(replicate 1)\n")
num <- sapply(out, is.numeric)
out[num] <- round(out[num], 3)
print(out[, c("classifier", "approach", "sensitivity", "specificity", "precision",
              "f1", "bal.accuracy", "accuracy", "roc.auc", "pr.auc", "cutoff")],
      row.names = FALSE)

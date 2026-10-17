# =============================================================
# 03_check_approaches.R  --  Step 3 checks: do the approaches run,
# give sensible probabilities, and how long do they take?
# =============================================================
source("R/00_setup.R")
design <- readRDS(file.path(RESULTS_DIR, "design.rds"))

# Quick scores until Step 4's evaluate() exists
auc  <- function(prob, y) performance(prediction(prob, y), "auc")@y.values[[1]]  # ISLR p. 394
sens <- function(prob, y, cutoff) mean(prob[y == "1"] > cutoff)   # share of positives caught
spec <- function(prob, y, cutoff) mean(prob[y == "0"] <= cutoff)  # share of negatives cleared

# ---- Check 1: does each resampler balance the TRAINING data? ----
row <- design[design$prev == 0.05 & design$n == 1000 &
              design$sep.level == "high" & design$rep.id == 1, ]
set.seed(row$seed)
train <- gen_data(row$n, row$prev, row$sep, row$beta0)
balance <- data.frame(
  method = c("original", "oversample", "undersample", "smote"),
  rows   = NA, negatives = NA, positives = NA)
sets <- list(train, oversample(train), undersample(train), smote(train))
for (i in 1:4) {
  balance$rows[i]      <- nrow(sets[[i]])
  balance$negatives[i] <- sum(sets[[i]]$y == "0")
  balance$positives[i] <- sum(sets[[i]]$y == "1")
}
cat("Check 1: class counts after resampling (5%, n = 1000)\n")
print(balance, row.names = FALSE)

# SMOTE rows must be NEW points, not copies of real ones
sm <- smote(train)
synthetic <- sm[is.na(sm$p.true), ]
cat("\nSMOTE: synthetic rows =", nrow(synthetic),
    "| exact copies of real rows =",
    sum(duplicated(rbind(train[, 1:5], synthetic[, 1:5]))[-(1:nrow(train))]), "\n\n")

# ---- Check 2: every approach built so far, on three settings ----
check.rows <- c(
  which(design$prev == 0.05 & design$n == 1000  & design$sep.level == "high" & design$rep.id == 1),
  which(design$prev == 0.01 & design$n == 1000  & design$sep.level == "high" & design$rep.id == 1),
  which(design$prev == 0.05 & design$n == 20000 & design$sep.level == "high" & design$rep.id == 1)
)
built <- c("baseline", "oversample", "undersample", "smote")

out <- NULL
for (r in check.rows) {
  row <- design[r, ]
  set.seed(row$seed)                                   # this replicate's seed
  train <- gen_data(row$n,  row$prev, row$sep, row$beta0)
  test  <- gen_data(N_TEST, row$prev, row$sep, row$beta0)

  for (cl in CLASSIFIERS) {
    for (ap in built) {
      secs <- system.time(res <- fit_and_predict(train, test, ap, cl))["elapsed"]
      out <- rbind(out, data.frame(
        setting    = row$setting,
        classifier = cl,
        approach   = ap,
        auc        = round(auc(res$prob, test$y), 3),
        sens       = round(sens(res$prob, test$y, res$cutoff), 3),
        spec       = round(spec(res$prob, test$y, res$cutoff), 3),
        mean.prob  = round(mean(res$prob), 3),
        seconds    = round(secs, 1)))
    }
  }
  cat("Bayes best AUC for", row$setting, ":", round(auc(test$p.true, test$y), 3), "\n")
}
cat("\n")
print(out, row.names = FALSE)

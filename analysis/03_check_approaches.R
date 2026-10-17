# =============================================================
# 03_check_approaches.R  --  Step 3 checks: do the approaches run,
# give sensible probabilities, and how long do they take?
# =============================================================
source("R/00_setup.R")
design <- readRDS(file.path(RESULTS_DIR, "design.rds"))

# Quick ranking score until Step 4's evaluate() exists (ROCR, ISLR p. 394)
auc <- function(prob, y) performance(prediction(prob, y), "auc")@y.values[[1]]

# Three test settings: small count, tiny count, large n
check.rows <- c(
  which(design$prev == 0.05 & design$n == 1000  & design$sep.level == "high" & design$rep.id == 1),
  which(design$prev == 0.01 & design$n == 1000  & design$sep.level == "high" & design$rep.id == 1),
  which(design$prev == 0.05 & design$n == 20000 & design$sep.level == "high" & design$rep.id == 1)
)

out <- NULL
for (r in check.rows) {
  row <- design[r, ]
  set.seed(row$seed)                                   # this replicate's seed
  train <- gen_data(row$n,  row$prev, row$sep, row$beta0)
  test  <- gen_data(N_TEST, row$prev, row$sep, row$beta0)

  for (cl in CLASSIFIERS) {
    secs <- system.time(
      res <- fit_and_predict(train, test, "baseline", cl)
    )["elapsed"]
    out <- rbind(out, data.frame(
      setting   = row$setting,
      positives = sum(train$y == "1"),
      classifier = cl,
      auc       = round(auc(res$prob, test$y), 3),
      bayes.auc = round(auc(test$p.true, test$y), 3),
      mean.prob = round(mean(res$prob), 4),     # should be near the prevalence
      seconds   = round(secs, 1)))
  }
}
print(out, row.names = FALSE)

# 11_calibration.R
# Do the remedies keep the predicted probabilities honest?
# For a few replicates we refit the methods on exactly the same data sets
# as the main simulation (same seeds as run_one()) and record
#   mean.prob  the average predicted P(y = 1) on the test set
#   brier      the Brier score, mean (prob - y)^2 (lower is better)
# A calibrated model's mean.prob should be close to the true prevalence.
# The threshold approach uses the baseline model, so it has the baseline's
# probabilities and is not refitted here.
# We also keep ROC-AUC to check it matches sim_results.rds exactly.
# Output: results/table_calibration.csv. About 15 minutes on 2 cores.

source("R/00_setup.R")
library(parallel)

N_CAL_REPS <- 5
design <- readRDS(file.path(RESULTS_DIR, "design.rds"))
rows <- design[design$n == 20000 & design$sep.level == "high" &
               design$prev < 0.5 & design$rep.id <= N_CAL_REPS, ]

calibrate_one <- function(i) {
  row <- rows[i, ]
  set_forest_rows(row$n)
  set.seed(row$seed)
  train <- gen_data(row$n,  row$prev, row$sep, row$beta0)
  test  <- gen_data(N_TEST, row$prev, row$sep, row$beta0)
  y <- as.numeric(test$y == "1")
  out <- NULL
  j <- 0
  for (cl in CLASSIFIERS) {
    for (ap in APPROACHES) {
      j <- j + 1
      if (ap == "threshold") next              # same probabilities as baseline
      set.seed(row$seed * 100 + j)              # same seed as run_one()
      res <- fit_and_predict(train, test, ap, cl)
      out <- rbind(out, data.frame(prev = row$prev, rep.id = row$rep.id,
                                   classifier = cl, approach = ap,
                                   test.prev = mean(y),
                                   mean.prob = mean(res$prob),
                                   brier = mean((res$prob - y)^2),
                                   roc.auc = evaluate(test$y, res$prob)$roc.auc))
    }
  }
  out
}

cl <- makeCluster(max(1, detectCores()))
clusterCall(cl, setwd, getwd())
clusterEvalQ(cl, suppressMessages(source("R/00_setup.R")))
clusterExport(cl, "rows")
cal <- do.call(rbind, parLapply(cl, 1:nrow(rows), calibrate_one))
stopCluster(cl)

# check: same ROC-AUC as the main simulation, so these are the same fits
sim <- readRDS(file.path(RESULTS_DIR, "sim_results.rds"))
chk <- merge(cal, sim[sim$n == 20000 & sim$sep.level == "high",
                      c("prev", "rep.id", "classifier", "approach", "roc.auc")],
             by = c("prev", "rep.id", "classifier", "approach"))
cat("Largest ROC-AUC difference from sim_results.rds:",
    max(abs(chk$roc.auc.x - chk$roc.auc.y)), "\n")

tab <- cal |>
  group_by(prev, classifier, approach) |>
  summarise(test.prev = mean(test.prev), mean.prob = mean(mean.prob),
            brier = mean(brier), .groups = "drop")
write.csv(tab, file.path(RESULTS_DIR, "table_calibration.csv"), row.names = FALSE)
print(as.data.frame(tab), digits = 3)

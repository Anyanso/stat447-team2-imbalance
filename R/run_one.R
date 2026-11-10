# run_one.R -- one replicate of the simulation (John Anyanso)
# Takes one row of the design and returns one result row per
# classifier x approach, plus the Bayes benchmark.

# Forests at n = 100,000 draw BIG_N_SAMPSIZE rows per tree, as on the
# real data, otherwise the run takes days. Smaller n use the usual bootstrap.
set_forest_rows <- function(n) {
  FOREST_SAMPSIZE <<- if (n > 20000) BIG_N_SAMPSIZE else NULL
}

run_one <- function(row) {

  set_forest_rows(row$n)
  set.seed(row$seed)
  train <- gen_data(row$n,  row$prev, row$sep, row$beta0)
  test  <- gen_data(N_TEST, row$prev, row$sep, row$beta0)   # large, fresh test set

  out <- cbind(classifier = "truth", approach = "bayes",
               bayes_benchmark(test, row$prev), seconds = 0)

  j <- 0
  for (cl in CLASSIFIERS) {
    for (ap in APPROACHES) {
      j <- j + 1
      # separate seed per method so results don't depend on the order they run in
      set.seed(row$seed * 100 + j)
      secs <- system.time(res <- fit_and_predict(train, test, ap, cl))["elapsed"]
      out <- rbind(out, cbind(classifier = cl, approach = ap,
                              evaluate(test$y, res$prob, res$cutoff),
                              seconds = secs))
    }
  }

  cbind(prev = row$prev, n = row$n, sep.level = row$sep.level,
        rep.id = row$rep.id, setting = row$setting,
        train.positives = sum(train$y == "1"),
        out)
}

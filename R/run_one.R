# =============================================================
# run_one.R  --  OWNER: Person 1   (Step 5)
# ONE replicate of the simulation = one row of the design grid.
# Class simulation deck, S.7 Step 2: "one replicate returns one row
# of results" -- here one row per (classifier, approach), plus Bayes.
# =============================================================

run_one <- function(row) {

  # ---- Block 1: this replicate's data (same seed -> same data, any laptop) ----
  set.seed(row$seed)
  train <- gen_data(row$n,  row$prev, row$sep, row$beta0)   # what the methods see
  test  <- gen_data(N_TEST, row$prev, row$sep, row$beta0)   # fresh and large (deck, Mistake 3)

  # ---- Block 2: the Bayes benchmark (the answer key) ----
  out <- cbind(classifier = "truth", approach = "bayes",
               bayes_benchmark(test, row$prev), seconds = 0)

  # ---- Block 3: every classifier x approach on the SAME data ----
  j <- 0
  for (cl in CLASSIFIERS) {
    for (ap in APPROACHES) {
      j <- j + 1
      # Each method gets its OWN seed, so its result does not depend on
      # which methods ran before it.
      set.seed(row$seed * 100 + j)
      secs <- system.time(res <- fit_and_predict(train, test, ap, cl))["elapsed"]
      out <- rbind(out, cbind(classifier = cl, approach = ap,
                              evaluate(test$y, res$prob, res$cutoff),
                              seconds = secs))
    }
  }

  # ---- Block 4: label every result row with its settings ----
  cbind(prev = row$prev, n = row$n, sep.level = row$sep.level,
        rep.id = row$rep.id, setting = row$setting,
        train.positives = sum(train$y == "1"),
        out)
}

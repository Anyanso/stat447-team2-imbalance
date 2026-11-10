# ensemble.R -- EasyEnsemble, the STAT 847 extension (Ambali Wadud)
#
# Under-sampling throws away most of the majority class. EasyEnsemble
# (Liu, Wu & Zhou, 2009) under-samples B times with different majority
# rows, fits a model to each balanced set and averages the predicted
# probabilities. The paper uses boosting inside each subset; we use our
# own two classifiers so the comparison with the other approaches is fair.

easy_ensemble <- function(train, test, classifier, B = ENSEMBLE_B) {
  prob.sum <- rep(0, nrow(test))
  for (b in 1:B) {
    bal <- undersample(train)

    # forest members get N_TREES / B trees, so the ensemble has as many
    # trees in total as one ordinary forest
    if (classifier == "logistic") {
      model <- fit_logistic(bal)
    } else {
      model <- fit_forest(bal, ntree = max(1, round(N_TREES / B)))
    }

    prob.sum <- prob.sum + predict_prob(model, test)
  }
  # members were trained on 50:50 data, so we cut at 0.5
  list(prob = prob.sum / B, cutoff = 0.5)
}

# The functions below rebuild exactly the same train/test data as
# run_one() and run_real_split() (same seeds), so the ensemble can be
# compared with the other approaches replicate by replicate.
# Seeds 13 and 14 continue after the 12 methods' seeds.

run_one_ensemble <- function(row) {
  set_forest_rows(row$n)
  set.seed(row$seed)
  train <- gen_data(row$n,  row$prev, row$sep, row$beta0)
  test  <- gen_data(N_TEST, row$prev, row$sep, row$beta0)
  out <- NULL
  for (k in seq_along(CLASSIFIERS)) {
    set.seed(row$seed * 100 + 12 + k)
    secs <- system.time(res <- easy_ensemble(train, test, CLASSIFIERS[k]))["elapsed"]
    out <- rbind(out, cbind(classifier = CLASSIFIERS[k], approach = "easyensemble",
                            evaluate(test$y, res$prob, res$cutoff), seconds = secs))
  }
  cbind(prev = row$prev, n = row$n, sep.level = row$sep.level,
        rep.id = row$rep.id, setting = row$setting,
        train.positives = sum(train$y == "1"), out)
}

run_real_split_ensemble <- function(split.id, data) {
  split.seed <- SEED + 10000 + split.id
  set.seed(split.seed)
  sp <- standardize_split(split_real(data))
  out <- NULL
  for (k in seq_along(CLASSIFIERS)) {
    set.seed(split.seed * 100 + 12 + k)
    secs <- system.time(res <- easy_ensemble(sp$train, sp$test, CLASSIFIERS[k]))["elapsed"]
    out <- rbind(out, cbind(classifier = CLASSIFIERS[k], approach = "easyensemble",
                            evaluate(sp$test$y, res$prob, res$cutoff), seconds = secs))
  }
  cbind(split.id = split.id, train.frauds = sum(sp$train$y == "1"),
        test.frauds = sum(sp$test$y == "1"), out)
}

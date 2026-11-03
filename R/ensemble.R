# =============================================================
# ensemble.R  --  OWNER: Person 3 (STAT 847 extension, Step 9)
# EasyEnsemble (Liu, Wu & Zhou, 2009), with our two classifiers.
# =============================================================
# The weakness of under-sampling: it throws most majority rows away.
# EasyEnsemble repeats under-sampling B times with DIFFERENT majority
# rows, fits a model to each balanced set, and AVERAGES the B predicted
# probabilities. Each model sees balanced data; together they see most
# of the majority class. (The original paper boosts within each subset;
# we use our two classifiers instead, so the comparison stays like-for-like.)
#
# CONTRACT
#   easy_ensemble(train, test, classifier, B = 10)
#   -> list(prob = averaged P(y = 1) on test, cutoff = 0.5)
# =============================================================

easy_ensemble <- function(train, test, classifier, B = ENSEMBLE_B) {
  prob.sum <- rep(0, nrow(test))
  for (b in 1:B) {
    # ---- Block 1: a fresh balanced under-sample (Approach A2 code) ----
    bal <- undersample(train)

    # ---- Block 2: fit one member on it ----
    # Forest members get N_TREES / B trees each, so the whole ensemble has
    # the same number of trees as one ordinary forest (fair compute).
    if (classifier == "logistic") {
      model <- fit_logistic(bal)
    } else {
      model <- fit_forest(bal, ntree = max(1, round(N_TREES / B)))
    }

    # ---- Block 3: add its probabilities to the running total ----
    prob.sum <- prob.sum + predict_prob(model, test)
  }
  # ---- Block 4: the ensemble's probability is the average ----
  # Every member was trained on 50:50 data, so 0.5 is the natural cutoff.
  list(prob = prob.sum / B, cutoff = 0.5)
}

# =============================================================
# Rerun the ensemble on EXACTLY the data the other approaches saw.
# Same seeds as run_one() / run_real_split() -> identical train/test,
# so the ensemble can be compared pair by pair with Steps 5 and 7.
# Seeds 13 and 14 follow the 12 methods' seeds 1-12 used there.
# =============================================================

# ---- Simulation: one design row ----
run_one_ensemble <- function(row) {
  set.seed(row$seed)
  train <- gen_data(row$n,  row$prev, row$sep, row$beta0)   # same as run_one()
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

# ---- Real data: one split ----
run_real_split_ensemble <- function(split.id, data) {
  split.seed <- SEED + 10000 + split.id
  set.seed(split.seed)
  sp <- standardize_split(split_real(data))                 # same as run_real_split()
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

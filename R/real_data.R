# =============================================================
# real_data.R  --  OWNER: Person 2   (Step 7)
# The Kaggle Credit Card Fraud data: 284,807 transactions, 492 frauds.
# Columns: Time, V1-V28 (already PCA-transformed by the data owners),
#          Amount, Class (1 = fraud).
# =============================================================

# ---- Block 1: load, and name the response y like the simulation ----
load_creditcard <- function(path = DATA_PATH) {
  cc <- read.csv(path)
  cc$y <- factor(cc$Class, levels = c(0, 1))   # same coding as gen_data()
  cc$Class <- NULL
  cc
}

# ---- Block 2: stratified train / test split ----
# Split positives and negatives SEPARATELY, so both sets keep the real
# fraud rate (0.17%). A plain random split could give the test set
# noticeably more or fewer frauds.
# Same idea as ISLR's  train <- sample(392, 196)  (p. 213), done per class.
split_real <- function(data, test.frac = 0.3) {
  test.rows <- NULL
  for (cl in levels(data$y)) {
    idx <- which(data$y == cl)
    test.rows <- c(test.rows, idx[sample(length(idx), round(test.frac * length(idx)))])
  }
  list(train = data[-test.rows, ], test = data[test.rows, ])
}

# ---- Block 3: standardize Time and Amount with TRAINING statistics ----
# Their scales (sd about 47,000 and 250) dwarf the V's (sd about 1), which
# would dominate SMOTE's nearest-neighbour distances.
# The mean and sd come from the TRAINING set only, then are applied to the
# test set -- the test set must not influence anything (Lecture 9 rule).
# (glmnet standardizes internally anyway; this matters for SMOTE.)
standardize_split <- function(sp, cols = c("Time", "Amount")) {
  for (v in cols) {
    m <- mean(sp$train[[v]])
    s <- sd(sp$train[[v]])
    sp$train[[v]] <- (sp$train[[v]] - m) / s
    sp$test[[v]]  <- (sp$test[[v]]  - m) / s
  }
  sp
}

# ---- Block 4: one split = every classifier x approach ----
# Same structure as run_one() in the simulation, minus the Bayes row:
# with real data nobody knows the true probabilities.
run_real_split <- function(split.id, data) {
  split.seed <- SEED + 10000 + split.id
  set.seed(split.seed)
  sp <- standardize_split(split_real(data))

  out <- cbind(classifier = "none", approach = "always-not-fraud",
               evaluate(sp$test$y, rep(0, nrow(sp$test))), seconds = 0)
  j <- 0
  for (cl in CLASSIFIERS) {
    for (ap in APPROACHES) {
      j <- j + 1
      set.seed(split.seed * 100 + j)
      secs <- system.time(res <- fit_and_predict(sp$train, sp$test, ap, cl))["elapsed"]
      out <- rbind(out, cbind(classifier = cl, approach = ap,
                              evaluate(sp$test$y, res$prob, res$cutoff),
                              seconds = secs))
    }
  }
  cbind(split.id = split.id,
        train.frauds = sum(sp$train$y == "1"),
        test.frauds  = sum(sp$test$y == "1"),
        out)
}

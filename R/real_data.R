# real_data.R -- the credit card fraud data (Nidhi Prajapati)
# Kaggle data: 284,807 transactions, 492 frauds. Columns are Time,
# V1-V28 (PCA components from the data owners), Amount and Class.

load_creditcard <- function(path = DATA_PATH) {
  cc <- read.csv(path)
  cc$y <- factor(cc$Class, levels = c(0, 1))   # same coding as gen_data()
  cc$Class <- NULL
  cc
}

# Stratified 70/30 split: sample frauds and normals separately so both
# halves keep the 0.17% fraud rate.
split_real <- function(data, test.frac = 0.3) {
  test.rows <- NULL
  for (cl in levels(data$y)) {
    idx <- which(data$y == cl)
    test.rows <- c(test.rows, idx[sample(length(idx), round(test.frac * length(idx)))])
  }
  list(train = data[-test.rows, ], test = data[test.rows, ])
}

# Time and Amount are on much bigger scales than the V's and would
# dominate SMOTE's distances, so we standardise them, using the training
# mean and sd for both sets.
standardize_split <- function(sp, cols = c("Time", "Amount")) {
  for (v in cols) {
    m <- mean(sp$train[[v]])
    s <- sd(sp$train[[v]])
    sp$train[[v]] <- (sp$train[[v]] - m) / s
    sp$test[[v]]  <- (sp$test[[v]]  - m) / s
  }
  sp
}

# One split: every classifier x approach, like run_one() but without a
# Bayes row (the true probabilities are unknown for real data).
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

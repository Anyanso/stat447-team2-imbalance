# =============================================================
# approaches.R  --  the three approach families (Step 3)
#   Shared helpers + classifiers + baseline ... all (piece 1)
#   Approach A, resampling ................... Person 1 (piece 2)
#   Approach B, class weights ................ Person 2 (piece 3)
#   Approach C, threshold .................... Person 3 (piece 4)
# Base-R syntax of the ISLR (R edition) labs; book page numbers cited.
# =============================================================
# CONTRACT (agreed on day 1)
#   fit_and_predict(train, test, approach, classifier)
#   -> list(prob = P(y = 1) for every row of test,
#           cutoff = number used to turn prob into a class)
# TWO RULES: change / tune using TRAINING data only. Never touch test.
# =============================================================


# ---- Helper 1: which columns are predictors? -----------------
# Everything except the response y and the answer key p.true.
# p.true must NEVER reach a model -- that would be cheating.
predictor_names <- function(data) {
  setdiff(names(data), c("y", "p.true"))
}

# ---- Helper 2: the numeric x matrix glmnet needs --------------
# ISLR 6.5.2 lab, p. 275:  x <- model.matrix(Salary ~ ., Hitters)[, -1]
# [, -1] drops the column of 1s (glmnet adds its own intercept).
make_x <- function(data) {
  model.matrix(y ~ ., data = data[, c(predictor_names(data), "y")])[, -1]
}

# ---- Helper 3: stratified fold numbers for cross-validation ----
# ISLR p. 273:  folds <- sample(rep(1:k, length = n))
# We do it SEPARATELY for positives and negatives, so every fold gets
# its share of the rare class. With only 10 positives, plain random
# folds could leave a fold with none.
stratified_folds <- function(y, k = 10) {
  folds <- rep(NA, length(y))
  for (cl in levels(y)) {
    idx <- which(y == cl)
    folds[idx] <- sample(rep(1:k, length = length(idx)))
  }
  folds
}

# =============================================================
# THE TWO CLASSIFIERS
# Each takes training data (+ optional case weights) and returns a
# fitted model; predict_prob() turns a model into P(y = 1) on new data.
# =============================================================

# ---- Classifier 1: penalized (lasso) logistic regression -------
# ISLR 6.5.2 lab, p. 279:  cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
#                          bestlam <- cv.out$lambda.min
# family = "binomial" makes it LOGISTIC (same word as glm(), p. 172).
# lambda is chosen by 10-fold CV on the TRAINING data only (Lecture 9 rule).
fit_logistic <- function(train, weights = NULL) {
  x <- make_x(train)
  if (is.null(weights)) weights <- rep(1, nrow(train))
  cv.glmnet(x, train$y, family = "binomial", alpha = 1,
            weights = weights,
            foldid = stratified_folds(train$y, k = 10),
            keep = TRUE)   # keep each row's out-of-fold prediction (used by Approach C)
}

# ---- Classifier 2: random forest --------------------------------
# ISLR 8.3.3 lab, p. 358:  randomForest(medv ~ ., data = Boston, mtry = 6)
# For classification the default mtry is sqrt(p) (p. 358), here 2 of 5.
fit_forest <- function(train, classwt = NULL, ntree = N_TREES) {
  randomForest(y ~ ., data = train[, c(predictor_names(train), "y")],
               ntree = ntree, classwt = classwt)
}

# ---- Predicted probabilities on new data ------------------------
# glmnet:       predict(..., s = bestlam, newx = ..., type = "response")
#               (type = "response" gives probabilities, as glm on p. 174)
# randomForest: type = "prob" gives one column per class; we keep "1".
predict_prob <- function(model, test) {
  if (inherits(model, "cv.glmnet")) {
    as.vector(predict(model, newx = make_x(test), s = "lambda.min",
                      type = "response"))
  } else {
    as.vector(predict(model, newdata = test, type = "prob")[, "1"])
  }
}

fit_classifier <- function(train, classifier) {
  if (classifier == "logistic") fit_logistic(train) else fit_forest(train)
}

# =============================================================
# APPROACH A: RESAMPLING  (Person 1)
# Change the TRAINING data so the classes are balanced 50 : 50.
# The model and the 0.5 cutoff stay the same. The test set is untouched.
# =============================================================

# ---- A1. Over-sampling: copy minority rows until the classes match ----
# Drawing WITH replacement, as in the bootstrap lab:
#   ISLR p. 216:  sample(100, 100, replace = T)
oversample <- function(train) {
  pos <- which(train$y == "1")
  neg <- which(train$y == "0")
  extra <- sample(pos, length(neg) - length(pos), replace = TRUE)
  train[c(neg, pos, extra), ]
}

# ---- A2. Under-sampling: throw away majority rows until the classes match ----
# Drawing WITHOUT replacement: keep a random subset of the negatives.
undersample <- function(train) {
  pos <- which(train$y == "1")
  neg <- which(train$y == "0")
  keep <- sample(neg, length(pos))
  train[c(keep, pos), ]
}

# ---- A3. SMOTE: create NEW minority rows between real ones ----
# Chawla et al. (2002). For each synthetic row:
#   1. pick a random minority row  a
#   2. pick one of its k nearest minority neighbours  b
#   3. new row = a + u * (b - a),  u ~ Uniform(0, 1)
#      i.e. a random point on the line segment from a to b.
# Nearest neighbours come from FNN::get.knn (same idea as knn(), ISLR p. 181).
smote <- function(train, k = 5) {
  preds <- predictor_names(train)
  pos.x <- as.matrix(train[train$y == "1", preds])
  n.pos <- nrow(pos.x)
  n.new <- sum(train$y == "0") - n.pos
  if (n.new <= 0 || n.pos < 2) return(train)    # already balanced / can't interpolate

  k  <- min(k, n.pos - 1)                       # can't have more neighbours than rows
  nn <- get.knn(pos.x, k = k)$nn.index          # row i: its k nearest minority rows

  a <- sample(1:n.pos, n.new, replace = TRUE)               # step 1
  b <- nn[cbind(a, sample(1:k, n.new, replace = TRUE))]     # step 2
  u <- runif(n.new)                                         # step 3
  new.x <- pos.x[a, , drop = FALSE] + u * (pos.x[b, , drop = FALSE] - pos.x[a, , drop = FALSE])

  new.rows <- data.frame(new.x, y = factor("1", levels = c("0", "1")))
  if ("p.true" %in% names(train)) new.rows$p.true <- NA     # synthetic: no true p
  rbind(train, new.rows[, names(train)])
}

# =============================================================
# APPROACH B: CLASS WEIGHTS  (Person 2)
# Keep the data as it is, but make each positive COUNT more.
# Weight = (number of negatives) / (number of positives), so both
# classes carry the same total weight -- "balanced" weights.
# Lecture 6: a weight ratio is a cost ratio -- "missing one positive
# costs w times as much as one false alarm."
# =============================================================
positive_weight <- function(y) {
  sum(y == "0") / sum(y == "1")
}

# =============================================================
# APPROACH C: THRESHOLD ADJUSTMENT  (Person 3)
# Leave the model alone; only move the cutoff away from 0.5.
# Lecture 6: "Nothing forces 0.5" -- a cutoff t encodes a cost ratio
# (1 - t) / t.
#
# The cutoff must be chosen WITHOUT the test set (Lecture 9 rule).
# We choose it on predictions for TRAINING rows that the model did not
# use for that row -- so they behave like new data:
#   logistic: out-of-fold predictions from the 10-fold CV that already
#             chose lambda (cv.glmnet keep = TRUE) -- the CV idea, Ch. 5
#   forest:   out-of-bag (OOB) votes -- each tree skips about 1/3 of
#             the rows, ISLR 8.2.1, p. 342
# Both are free: no extra models are fitted.
# =============================================================

# ---- C1. Honest "unseen" probabilities for the training rows ----
out_of_sample_prob <- function(model) {
  if (inherits(model, "cv.glmnet")) {
    eta <- model$fit.preval[, which(model$lambda == model$lambda.min)]  # log-odds
    exp(eta) / (1 + exp(eta))                                          # ISLR p. 316
  } else {
    model$votes[, "1"]                                                 # OOB vote share
  }
}

# ---- C2. Choose the cutoff that maximizes a criterion ----
# ROCR (ISLR 9.6.3, p. 394) computes sensitivity and specificity at
# EVERY possible cutoff in one call.
#   "youden": sensitivity + specificity - 1  (same as maximizing
#             balanced accuracy -- treats both classes equally)
#   "f1"    : F1 score (focuses on the positive class and precision)
choose_cutoff <- function(prob, y, criterion = CUTOFF_CRITERION) {
  ok <- !is.na(prob)                              # rows never out-of-bag (rare)
  pred <- prediction(prob[ok], y[ok])
  if (criterion == "youden") {
    perf  <- performance(pred, "sens", "spec")
    score <- perf@y.values[[1]] + perf@x.values[[1]] - 1
  } else {
    perf  <- performance(pred, "f")
    score <- perf@y.values[[1]]
  }
  cuts <- perf@alpha.values[[1]]
  keep <- is.finite(cuts) & !is.na(score)         # drop ROCR's "Inf" start point
  cuts[keep][which.max(score[keep])]
}

# =============================================================
# THE APPROACHES
# =============================================================
fit_and_predict <- function(train, test, approach, classifier) {

  if (approach == "baseline") {
    # Nothing changed: fit as usual, classify at the usual 0.5 cutoff.
    model <- fit_classifier(train, classifier)
    return(list(prob = predict_prob(model, test), cutoff = 0.5))
  }

  if (approach %in% c("oversample", "undersample", "smote")) {
    # Approach A: rebalance the TRAINING data only, then fit as usual.
    new.train <- switch(approach,
                        oversample  = oversample(train),
                        undersample = undersample(train),
                        smote       = smote(train))
    model <- fit_classifier(new.train, classifier)
    return(list(prob = predict_prob(model, test), cutoff = 0.5))
  }

  if (approach == "weights") {
    # Approach B: same data, but every positive counts w times.
    w <- positive_weight(train$y)
    if (classifier == "logistic") {
      # glmnet: one weight per ROW in the log-likelihood
      model <- fit_logistic(train, weights = ifelse(train$y == "1", w, 1))
    } else {
      # randomForest: one weight per CLASS (classwt), in level order "0", "1"
      model <- fit_forest(train, classwt = c("0" = 1, "1" = w))
    }
    return(list(prob = predict_prob(model, test), cutoff = 0.5))
  }

  if (approach == "threshold") {
    # Approach C: the SAME model as the baseline; only the cutoff moves.
    model  <- fit_classifier(train, classifier)
    cutoff <- choose_cutoff(out_of_sample_prob(model), train$y)
    return(list(prob = predict_prob(model, test), cutoff = cutoff))
  }

  stop("approach '", approach, "' not built yet")
}

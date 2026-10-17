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
            foldid = stratified_folds(train$y, k = 10))
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
# THE APPROACHES
# =============================================================
fit_and_predict <- function(train, test, approach, classifier) {

  if (approach == "baseline") {
    # Nothing changed: fit as usual, classify at the usual 0.5 cutoff.
    model <- fit_classifier(train, classifier)
    return(list(prob = predict_prob(model, test), cutoff = 0.5))
  }

  stop("approach '", approach, "' not built yet")
}

# approaches.R -- the classifiers and the three families of remedies
#   A. resampling     John Anyanso
#   B. class weights  Nidhi Prajapati
#   C. threshold      Ambali Wadud
#
# Main function: fit_and_predict(train, test, approach, classifier)
# returns list(prob = P(y = 1) for each test row, cutoff = cutoff to use).
# Anything that is tuned or rebalanced uses the training data only.


# predictors = every column except y and p.true (p.true is the answer key)
predictor_names <- function(data) {
  setdiff(names(data), c("y", "p.true"))
}

# x matrix for glmnet, as in the ISLR lab: model.matrix(...)[, -1]
make_x <- function(data) {
  model.matrix(y ~ ., data = data[, c(predictor_names(data), "y")])[, -1]
}

# CV folds made separately within each class, so every fold gets some
# positives (with ~10 positives, random folds could miss them entirely)
stratified_folds <- function(y, k = 10) {
  folds <- rep(NA, length(y))
  for (cl in levels(y)) {
    idx <- which(y == cl)
    folds[idx] <- sample(rep(1:k, length = length(idx)))
  }
  folds
}


# ---- Classifiers ----

# Lasso logistic regression, lambda chosen by 10-fold CV (lambda.min).
# keep = TRUE saves the out-of-fold predictions for the threshold approach.
# On the real data glmnet's default lambda path doesn't converge, so
# 00_setup.R can set a finer path (LASSO_MIN_RATIO, LASSO_NLAMBDA).
fit_logistic <- function(train, weights = NULL) {
  x <- make_x(train)
  if (is.null(weights)) weights <- rep(1, nrow(train))
  args <- list(x, train$y, family = "binomial", alpha = 1,
               weights = weights,
               foldid = stratified_folds(train$y, k = 10),
               keep = TRUE)
  if (!is.null(LASSO_MIN_RATIO)) args$lambda.min.ratio <- LASSO_MIN_RATIO
  if (!is.null(LASSO_NLAMBDA))   args$nlambda          <- LASSO_NLAMBDA
  do.call(cv.glmnet, args)
}

# Random forest with the default mtry (sqrt(p), so 2 of 5 here).
# If FOREST_SAMPSIZE is set, each tree draws that many rows instead of a
# full bootstrap sample. We do this for the large real data set.
fit_forest <- function(train, classwt = NULL, ntree = N_TREES) {
  if (is.null(FOREST_SAMPSIZE) || nrow(train) <= FOREST_SAMPSIZE) {
    randomForest(y ~ ., data = train[, c(predictor_names(train), "y")],
                 ntree = ntree, classwt = classwt)
  } else {
    randomForest(y ~ ., data = train[, c(predictor_names(train), "y")],
                 ntree = ntree, classwt = classwt, sampsize = FOREST_SAMPSIZE)
  }
}

# predicted P(y = 1) from either model
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


# ---- A. Resampling: balance the training set 50:50 ----

# Which class is smaller in this training set. Usually "1", but at 50%
# prevalence it can go either way.
# Note: we use idx[sample(length(idx), ...)] because sample(idx, ...)
# misbehaves when idx has length 1.
class_rows <- function(y) {
  pos <- which(y == "1")
  neg <- which(y == "0")
  if (length(pos) <= length(neg)) list(min = pos, maj = neg) else list(min = neg, maj = pos)
}

# over-sampling: copy minority rows (with replacement) until the classes match
oversample <- function(train) {
  cr <- class_rows(train$y)
  extra <- cr$min[sample(length(cr$min), length(cr$maj) - length(cr$min), replace = TRUE)]
  train[c(cr$maj, cr$min, extra), ]
}

# under-sampling: keep a random subset of the majority class
undersample <- function(train) {
  cr <- class_rows(train$y)
  keep <- cr$maj[sample(length(cr$maj), length(cr$min))]
  train[c(keep, cr$min), ]
}

# SMOTE (Chawla et al., 2002): each new row is a random point on the line
# between a minority row a and one of its k nearest minority neighbours b,
#   new = a + u * (b - a),  u ~ Uniform(0, 1)
smote <- function(train, k = 5) {
  preds <- predictor_names(train)
  cr    <- class_rows(train$y)
  min.class <- as.character(train$y[cr$min[1]])
  pos.x <- as.matrix(train[cr$min, preds])
  n.pos <- nrow(pos.x)
  n.new <- length(cr$maj) - n.pos
  if (n.new <= 0 || n.pos < 2) return(train)    # nothing to add, or can't interpolate

  k  <- min(k, n.pos - 1)
  nn <- get.knn(pos.x, k = k)$nn.index          # row i = indices of its k neighbours

  a <- sample(1:n.pos, n.new, replace = TRUE)
  b <- nn[cbind(a, sample(1:k, n.new, replace = TRUE))]
  u <- runif(n.new)
  new.x <- pos.x[a, , drop = FALSE] + u * (pos.x[b, , drop = FALSE] - pos.x[a, , drop = FALSE])

  new.rows <- data.frame(new.x, y = factor(min.class, levels = c("0", "1")))
  if ("p.true" %in% names(train)) new.rows$p.true <- NA     # no true p for synthetic rows
  rbind(train, new.rows[, names(train)])
}


# ---- B. Class weights ----
# Each positive gets weight (#negatives / #positives), so both classes
# carry the same total weight.
positive_weight <- function(y) {
  sum(y == "0") / sum(y == "1")
}


# ---- C. Threshold: same model, different cutoff ----
# The cutoff is picked without the test set, using training predictions
# from rows the model didn't use for that prediction:
#   logistic: out-of-fold predictions from the CV that chose lambda
#   forest:   out-of-bag votes
# So no extra models are needed.

out_of_sample_prob <- function(model) {
  if (inherits(model, "cv.glmnet")) {
    eta <- model$fit.preval[, which(model$lambda == model$lambda.min)]  # log-odds
    exp(eta) / (1 + exp(eta))
  } else {
    model$votes[, "1"]
  }
}

# cutoff that maximises the criterion, using ROCR to get every cutoff at once
#   youden: sensitivity + specificity - 1
#   f1:     F1 score
choose_cutoff <- function(prob, y, criterion = CUTOFF_CRITERION) {
  ok <- !is.na(prob)                              # a row may never be out-of-bag
  pred <- prediction(prob[ok], y[ok])
  if (criterion == "youden") {
    perf  <- performance(pred, "sens", "spec")
    score <- perf@y.values[[1]] + perf@x.values[[1]] - 1
  } else {
    perf  <- performance(pred, "f")
    score <- perf@y.values[[1]]
  }
  cuts <- perf@alpha.values[[1]]
  keep <- is.finite(cuts) & !is.na(score)         # drop the Inf cutoff ROCR adds
  cuts[keep][which.max(score[keep])]
}


# ---- Putting it together ----
fit_and_predict <- function(train, test, approach, classifier) {

  if (approach == "baseline") {
    model <- fit_classifier(train, classifier)
    return(list(prob = predict_prob(model, test), cutoff = 0.5))
  }

  if (approach %in% c("oversample", "undersample", "smote")) {
    new.train <- switch(approach,
                        oversample  = oversample(train),
                        undersample = undersample(train),
                        smote       = smote(train))
    model <- fit_classifier(new.train, classifier)
    return(list(prob = predict_prob(model, test), cutoff = 0.5))
  }

  if (approach == "weights") {
    w <- positive_weight(train$y)
    if (classifier == "logistic") {
      # glmnet takes a weight for each row
      model <- fit_logistic(train, weights = ifelse(train$y == "1", w, 1))
    } else {
      # randomForest takes a weight for each class
      model <- fit_forest(train, classwt = c("0" = 1, "1" = w))
    }
    return(list(prob = predict_prob(model, test), cutoff = 0.5))
  }

  if (approach == "threshold") {
    model  <- fit_classifier(train, classifier)
    cutoff <- choose_cutoff(out_of_sample_prob(model), train$y)
    return(list(prob = predict_prob(model, test), cutoff = cutoff))
  }

  stop("unknown approach: ", approach)
}

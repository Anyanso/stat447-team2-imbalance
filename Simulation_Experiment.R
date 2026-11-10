################################################################################
# REVISION 2: PROJECT-ROOT FOLDER STRUCTURE, SAVED SIMULATED DATA, AND 8 FINAL PLOTS
# In RStudio open your .Rproj; then source this file from that project directory.
# This code creates data/simulation, results/simulation, and plots/simulation.
################################################################################
################################################################################
# STAT 847 -- Team 2 | University of Saskatchewan | Dr. Li Xing | Fall 2026
# RESEARCH QUESTION: Can class-imbalance remedies improve prediction?
# This file is SELF-CONTAINED: no dependency on the other R script.
# FILE LOCATIONS: data/, results/, and plots/ under the RStudio working directory.
# REPRODUCIBILITY: fixed seeds, source code, data provenance and sessionInfo.
# CRITICAL: Keep the entire held-out TEST set untouched and at natural prevalence.
################################################################################
options(warn = 1)
set.seed(847L)

# FILE 1 OF 3: SIMULATION STUDY, WITH EDA AND MONTE CARLO REPLICATIONS.
CONFIG <- list(
  SEED = 847L,
  TEST_FRACTION = 0.20,
  CV_FOLDS = 5L,
  INNER_GLMNET_FOLDS = 3L,
  RIDGE_LAMBDAS = c(0.005, 0.02, 0.08, 0.32),
  RF_TREES = 150L,
  RF_THREADS = 1L,
  TARGET_OVER_RATIO = 0.25,
  UNDER_MAJ_TO_MIN = 4L,
  POS_CLASS_WEIGHT = "sqrt",
  ENSEMBLE_WEIGHTS = seq(0.1, 0.9, by = 0.1),
  BOOTSTRAP_REPS = 200L,
  SAVE_OOF = FALSE,
  OUTPUT_DIR = "results",
  DATA_DIR = "data"
)

# Paths are rooted at getwd(): open the GitHub/RStudio project before sourcing.
# Generated data and results are kept separate for reproducibility.
CONFIG$PROJECT_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
CONFIG$OUTPUT_DIR <- file.path(CONFIG$PROJECT_ROOT, "results", "simulation")
CONFIG$TABLE_DIR <- file.path(CONFIG$OUTPUT_DIR, "tables")
CONFIG$METADATA_DIR <- file.path(CONFIG$OUTPUT_DIR, "metadata")
CONFIG$DATA_DIR <- file.path(CONFIG$PROJECT_ROOT, "data", "simulation")
CONFIG$GENERATED_DATA_DIR <- file.path(CONFIG$DATA_DIR, "generated")
CONFIG$DESIGN_DATA_DIR <- file.path(CONFIG$DATA_DIR, "design")
CONFIG$PLOT_EDA_DIR <- file.path(CONFIG$PROJECT_ROOT, "plots", "simulation", "eda")
CONFIG$PLOT_FINAL_DIR <- file.path(CONFIG$PROJECT_ROOT, "plots", "simulation", "final")
CONFIG$PLOT_CURVE_DIR <- file.path(CONFIG$PROJECT_ROOT, "plots", "simulation", "curves")
CONFIG$SAVE_SIMULATED_DATA <- TRUE     # saves each replicate as compressed CSV
CONFIG$SAVE_TRUE_PROBABILITY <- FALSE # keep oracle out of saved predictors by default
for (path in c(CONFIG$TABLE_DIR, CONFIG$METADATA_DIR,
               CONFIG$GENERATED_DATA_DIR, CONFIG$DESIGN_DATA_DIR,
               CONFIG$PLOT_EDA_DIR, CONFIG$PLOT_FINAL_DIR, CONFIG$PLOT_CURVE_DIR)) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

needed <- c("data.table", "ggplot2", "glmnet", "ranger", "themis")
missing <- setdiff(needed, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
for (pkg in needed) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop("Missing R package: ", pkg)
}
dir.create(CONFIG$OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CONFIG$DATA_DIR, recursive = TRUE, showWarnings = FALSE)
log_message <- function(...) cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", ..., "\n", sep = "")
write_csv <- function(x, filename) data.table::fwrite(x, file.path(CONFIG$TABLE_DIR, filename))
write_compressed_data <- function(x, filename, folder) {
  data.table::fwrite(x, file.path(folder, filename), compress = "gzip")
}
# ========================= 2. EVALUATION METRICS ===============================
# Positive class is always 1 (fraud). The code implements average precision,
# often reported as PR-AUC/AP. AP is the non-interpolated integral/sum across
# recall jumps; it is NOT identical to every trapezoidal PR-AUC convention.

clip_prob <- function(p) pmin(1 - 1e-12, pmax(1e-12, as.numeric(p)))

auc_roc <- function(y, p) {
  y <- as.integer(as.character(y))
  n1 <- sum(y == 1L); n0 <- sum(y == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

average_precision <- function(y, p) {
  y <- as.integer(as.character(y))
  if (sum(y == 1L) == 0L) return(NA_real_)
  # Treat tied scores as a group to avoid arbitrary tie-order dependence.
  ord <- order(p, decreasing = TRUE)
  yy <- y[ord]; pp <- p[ord]
  ends <- which(!duplicated(pp, fromLast = TRUE))
  tp <- cumsum(yy)[ends]
  prec <- tp / ends
  recall <- tp / sum(y == 1L)
  sum(diff(c(0, recall)) * prec)
}

metrics <- function(y, p, threshold = 0.5) {
  y <- as.integer(as.character(y)); p <- clip_prob(p)
  pred <- as.integer(p >= threshold)
  tp <- sum(pred == 1 & y == 1)
  tn <- sum(pred == 0 & y == 0)
  fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  div <- function(a, b, absent = NA_real_) if (b == 0) absent else a / b
  sens <- div(tp, tp + fn)
  spec <- div(tn, tn + fp)
  precision <- div(tp, tp + fp, absent = 0)
  f1 <- div(2 * tp, 2 * tp + fp + fn, absent = 0)
  data.frame(
    N = length(y), Positives = sum(y == 1L),
    Prevalence = mean(y == 1L), Threshold = threshold,
    Accuracy = mean(pred == y), Sensitivity = sens,
    Specificity = spec, Precision = precision, F1 = f1,
    Balanced_Accuracy = mean(c(sens, spec)),
    ROC_AUC = auc_roc(y, p), PR_AUC_AP = average_precision(y, p),
    Brier = mean((p - y)^2),
    Log_Loss = -mean(y * log(p) + (1 - y) * log(1 - p)),
    TP = tp, FP = fp, TN = tn, FN = fn,
    Flagged_Count = tp + fp,
    stringsAsFactors = FALSE
  )
}

majority_prediction <- function(y) {
  # Naive classifier: everybody negative; constant risk is observed train rate
  # for probability metrics, not literally zero.
  rep(0, length(y))
}

# All cutoffs are chosen on OUT-OF-FOLD training predictions, never on test data.
tune_threshold <- function(y, p, beta = 1) {
  y <- as.integer(as.character(y)); p <- clip_prob(p)
  if (sum(y == 1L) == 0L) return(0.5)
  oo <- order(p, decreasing = TRUE)
  yy <- y[oo]; pp <- p[oo]
  ends <- which(!duplicated(pp, fromLast = TRUE))
  tp <- cumsum(yy)[ends]
  fp <- ends - tp
  fn <- sum(y) - tp
  b2 <- beta^2
  fbeta <- (1 + b2) * tp / pmax(1, (1 + b2) * tp + b2 * fn + fp)
  # Ties: prefer the higher cutoff (fewer needless fraud investigations).
  best <- which(fbeta == max(fbeta))[1]
  pp[ends[best]]
}

# Stratify assignments so scarce positives are spread among the k folds.
stratified_folds <- function(y, k = 5L, seed = 1L) {
  y <- as.character(y)
  if (min(table(y)) < k) stop("Too few minority examples for ", k, "-fold CV")
  set.seed(seed)
  fold <- integer(length(y))
  for (level in unique(y)) {
    ind <- sample(which(y == level))
    fold[ind] <- rep(seq_len(k), length.out = length(ind))
  }
  fold
}

stratified_split <- function(d, test_frac, seed) {
  set.seed(seed)
  is_test <- rep(FALSE, nrow(d))
  for (level in levels(d$Class)) {
    ind <- which(d$Class == level)
    n_te <- max(1L, round(length(ind) * test_frac))
    is_test[sample(ind, n_te)] <- TRUE
  }
  list(train = d[!is_test, , drop = FALSE],
       test = d[is_test, , drop = FALSE])
}

chronological_split <- function(d, test_frac) {
  if (!"Time" %in% names(d)) stop("Chronological split requires Time")
  oo <- order(d$Time, seq_len(nrow(d)))
  ntr <- floor(nrow(d) * (1 - test_frac))
  list(train = d[oo[seq_len(ntr)], , drop = FALSE],
       test = d[oo[seq.int(ntr + 1L, nrow(d))], , drop = FALSE])
}

# ============================ 3. SIMULATION ====================================
# Y | X ~ Bernoulli(logit^{-1}(beta0 + beta1*X1 + ... + beta3*X3)).
# beta1..beta3 and X distributions DO NOT CHANGE as prevalence changes.
# Only beta0 changes. MC integration sets the EXPECTED positive fraction;
# sampled positive counts are intentionally random, not forced to be exact.

BETA <- c(X1 = 1.25, X2 = -1.00, X3 = 0.80, X4 = 0.00)
get_intercept <- function(target) {
  uniroot(function(b0) mean(plogis(b0 + MC_ETA)) - target,
          interval = c(-20, 20), tol = 1e-10)$root
}

simulate_dataset <- function(n, prevalence, seed) {
  set.seed(seed)
  x <- matrix(rnorm(n * length(BETA)), ncol = length(BETA))
  colnames(x) <- names(BETA)
  b0 <- SIM_INTERCEPTS[as.character(prevalence)]
  true_prob <- plogis(as.numeric(b0 + x %*% BETA))
  y <- rbinom(n, size = 1L, prob = true_prob)
  d <- as.data.frame(x)
  d$Class <- factor(as.character(y), levels = c("0", "1"))
  list(data = d, truth = true_prob, intercept = as.numeric(b0))
}

# ======================= 4. LEAKAGE-FREE TRAINING ===============================
# All scaling, resampling, and inner lambda tuning are learned only on the
# CURRENT training fold. Validation/test observations are NEVER resampled.

get_predictors <- function(d) setdiff(names(d), "Class")

standardize_train <- function(d) {
  cols <- get_predictors(d)
  mu <- vapply(d[cols], mean, numeric(1))
  sig <- vapply(d[cols], sd, numeric(1))
  sig[!is.finite(sig) | sig < 1e-10] <- 1
  scaled <- d
  for (v in cols) scaled[[v]] <- (d[[v]] - mu[v]) / sig[v]
  list(data = scaled, mean = mu, sd = sig, predictors = cols)
}

apply_scaling <- function(d, prep) {
  ans <- d
  for (v in prep$predictors) ans[[v]] <- (d[[v]] - prep$mean[v]) / prep$sd[v]
  ans
}

resample_train <- function(d, method, seed) {
  if (!method %in% c("baseline", "oversample", "undersample", "smote", "weighted")) {
    stop("Unrecognized remedy: ", method)
  }
  if (method %in% c("baseline", "weighted")) return(d)
  set.seed(seed)
  pos <- which(d$Class == "1"); neg <- which(d$Class == "0")
  np <- length(pos); nn <- length(neg)
  if (np < 1L || nn < 1L) stop("Both classes required")
  if (method == "oversample") {
    target <- max(np, ceiling(nn * CONFIG$TARGET_OVER_RATIO))
    extra <- if (target > np) sample(pos, target - np, replace = TRUE) else integer(0)
    out <- d[c(seq_len(nrow(d)), extra), , drop = FALSE]
  } else if (method == "undersample") {
    target <- min(nn, as.integer(CONFIG$UNDER_MAJ_TO_MIN * np))
    out <- d[c(pos, sample(neg, target)), , drop = FALSE]
  } else { # smote -- called ONLY after fold-specific scaling
    if (np < 2L) stop("SMOTE requires at least two minority observations")
    if (np < 6L) warning("SMOTE has fewer than 6 minority cases: synthetic geometry is unstable")
    if (np / nn >= CONFIG$TARGET_OVER_RATIO) return(d)
    out <- as.data.frame(themis::smote(
      df = d, var = "Class", k = min(5L, np - 1L),
      over_ratio = CONFIG$TARGET_OVER_RATIO, distance = "euclidean"
    ))
  }
  out <- out[sample.int(nrow(out)), , drop = FALSE]
  out$Class <- factor(as.character(out$Class), levels = c("0", "1"))
  rownames(out) <- NULL
  out
}

# Ridge hyperparameter selection is truly NESTED: for every inner split,
# scaling and sampling are independently fit to inner-training rows only.
# Resampled duplicates can therefore NEVER be placed on the inner validation side.
calc_weights <- function(d, remedy, original_d) {
  imbalance_ratio <- sum(original_d$Class == "0") / sum(original_d$Class == "1")
  posweight <- switch(CONFIG$POS_CLASS_WEIGHT,
                      sqrt = sqrt(imbalance_ratio),
                      inverse = imbalance_ratio,
                      stop("Unknown POS_CLASS_WEIGHT configuration"))
  if (remedy == "weighted") {
    return(ifelse(d$Class == "1", posweight, 1))
  }
  rep(1, nrow(d))
}

fit_classifier <- function(train, classifier, remedy, seed) {
  if (!identical(levels(train$Class), c("0", "1"))) stop("Class must have levels 0,1")
  if (anyNA(train) || any(!is.finite(as.matrix(train[get_predictors(train)])))) {
    stop("Missing or non-finite predictors encountered")
  }
  if (classifier == "Penalized_Logistic") {
    k_inner <- min(CONFIG$INNER_GLMNET_FOLDS, min(table(train$Class)))
    if (k_inner < 2L) stop("Insufficient minority cases for inner CV")
    inner <- stratified_folds(train$Class, k_inner, seed + 9000L)
    lambdas <- CONFIG$RIDGE_LAMBDAS
    losses <- matrix(NA_real_, k_inner, length(lambdas))
    for (fold in seq_len(k_inner)) {
      d_inner <- train[inner != fold, , drop = FALSE]
      d_valid <- train[inner == fold, , drop = FALSE]
      pre <- standardize_train(d_inner)
      fit_data <- resample_train(pre$data, remedy, seed + fold + 1000L)
      yy <- as.integer(as.character(fit_data$Class))
      wt <- calc_weights(fit_data, remedy, d_inner)
      model_inner <- glmnet::glmnet(
        as.matrix(fit_data[pre$predictors]), yy, family = "binomial",
        alpha = 0, lambda = lambdas, weights = wt,
        standardize = FALSE, maxit = 100000L)
      valid_scaled <- apply_scaling(d_valid, pre)
      probs <- as.matrix(predict(model_inner,
                     newx = as.matrix(valid_scaled[pre$predictors]),
                     s = lambdas, type = "response"))
      actual <- as.integer(as.character(d_valid$Class))
      for (l in seq_along(lambdas)) {
        pr <- clip_prob(probs[, l])
        losses[fold, l] <- -mean(actual * log(pr) +
                               (1 - actual) * log(1 - pr))
      }
    }
    best_lambda <- lambdas[which.min(colMeans(losses))]
    prep <- standardize_train(train)
    d <- resample_train(prep$data, remedy, seed + 2000L)
    fit <- glmnet::glmnet(
      x = as.matrix(d[prep$predictors]),
      y = as.integer(as.character(d$Class)), family = "binomial",
      alpha = 0, lambda = CONFIG$RIDGE_LAMBDAS,
      weights = calc_weights(d, remedy, train),
      standardize = FALSE, maxit = 100000L)
    return(list(classifier = classifier, remedy = remedy, model = fit,
                prep = prep, chosen_lambda = best_lambda))
  }
  if (classifier == "Random_Forest") {
    prep <- standardize_train(train)
    d <- resample_train(prep$data, remedy, seed)
    cw <- if (remedy == "weighted") {
      c("0" = 1, "1" = switch(CONFIG$POS_CLASS_WEIGHT,
          sqrt = sqrt(sum(train$Class == "0") / sum(train$Class == "1")),
          inverse = sum(train$Class == "0") / sum(train$Class == "1")))
    } else NULL
    fit <- ranger::ranger(
      Class ~ ., data = d, probability = TRUE,
      num.trees = CONFIG$RF_TREES,
      mtry = max(1L, floor(sqrt(length(prep$predictors)))),
      min.node.size = 10L, class.weights = cw,
      num.threads = CONFIG$RF_THREADS, seed = seed, write.forest = TRUE)
    return(list(classifier = classifier, remedy = remedy, model = fit,
                prep = prep, chosen_lambda = NA_real_))
  }
  stop("Unknown classifier")
}

predict_classifier <- function(object, new_data) {
  d <- apply_scaling(new_data, object$prep)
  if (object$classifier == "Penalized_Logistic") {
    p <- predict(object$model, newx = as.matrix(d[object$prep$predictors]),
                 s = object$chosen_lambda, type = "response")
  } else {
    p <- predict(object$model, data = d,
                 num.threads = CONFIG$RF_THREADS)$predictions[, "1"]
  }
  clip_prob(p)
}

# ====================== 5. FOLD-LEVEL MODEL COMPARISON ==========================
CLASSIFIERS <- c("Penalized_Logistic", "Random_Forest")
REMEDIES <- c("baseline", "oversample", "undersample", "smote", "weighted")

out_of_fold <- function(train, classifier, remedy, folds, seed) {
  p <- rep(NA_real_, nrow(train))
  for (k in sort(unique(folds))) {
    in_fold <- which(folds != k)
    held_out <- which(folds == k)
    fitted <- fit_classifier(train[in_fold, , drop = FALSE],
                             classifier, remedy, seed + k)
    p[held_out] <- predict_classifier(fitted, train[held_out, , drop = FALSE])
  }
  stopifnot(all(is.finite(p)))
  p
}

# A candidate is a learner + a training remedy; threshold changes are separate
# operating rules on the SAME predicted probability scores.
candidate_name <- function(classifier, remedy) paste(classifier, remedy, sep = "__")

select_best_by_cv_ap <- function(p_mat, y, classifier) {
  names_here <- grep(paste0("^", classifier, "__"), colnames(p_mat), value = TRUE)
  ap <- vapply(names_here, function(nm) average_precision(y, p_mat[, nm]),
               numeric(1))
  names_here[which.max(ap)]
}

# ====================== 6. ENSEMBLE (STAT 847) ================================
make_ensemble <- function(oof, test_scores, y_train) {
  best_logit <- select_best_by_cv_ap(oof, y_train, "Penalized_Logistic")
  best_forest <- select_best_by_cv_ap(oof, y_train, "Random_Forest")
  o1 <- oof[, best_logit]; o2 <- oof[, best_forest]
  t1 <- test_scores[, best_logit]; t2 <- test_scores[, best_forest]
  w_grid <- CONFIG$ENSEMBLE_WEIGHTS
  ap_grid <- vapply(w_grid, function(w) {
    average_precision(y_train, w * o1 + (1 - w) * o2)
  }, numeric(1))
  # Force 0 < weight < 1 to make this a real blend, not an endpoint model.
  w_best <- w_grid[which.max(ap_grid)]
  oof_p <- w_best * o1 + (1 - w_best) * o2
  test_p <- w_best * t1 + (1 - w_best) * t2
  list(oof = oof_p, test = test_p, weight = w_best,
       logit_model = best_logit, forest_model = best_forest,
       weights = data.frame(Logistic_Weight = w_grid,
                            RF_Weight = 1 - w_grid,
                            CV_AP = ap_grid))
}

bootstrap_ensemble_difference <- function(y, ensemble_p, single_p, B, seed) {
  set.seed(seed)
  y <- as.integer(as.character(y))
  i0 <- which(y == 0); i1 <- which(y == 1)
  vals <- replicate(B, {
    ind <- c(sample(i0, length(i0), replace = TRUE),
             sample(i1, length(i1), replace = TRUE))
    average_precision(y[ind], ensemble_p[ind]) -
      average_precision(y[ind], single_p[ind])
  })
  data.frame(
    Ensemble_minus_Single_AP = average_precision(y, ensemble_p) -
      average_precision(y, single_p),
    Bootstrap_CI_Lower = unname(quantile(vals, 0.025)),
    Bootstrap_CI_Upper = unname(quantile(vals, 0.975)),
    Bootstrap_Pr_Difference_Positive = mean(vals > 0),
    Bootstrap_Replicates = B
  )
}

# ======================= 7. RUN ONE EXPERIMENT ================================
experiment <- function(train, test, name, seed, target_prevalence = NA_real_,
                       true_test_prob = NULL, is_real = FALSE) {
  log_message("START: ", name,
              " | training: ", nrow(train), " (positive ", sum(train$Class == "1"), ")",
              " | test: ", nrow(test), " (positive ", sum(test$Class == "1"), ")")
  if (min(table(train$Class)) < CONFIG$CV_FOLDS) {
    stop("Not enough minority observations for stratified CV: ", name)
  }
  folds <- stratified_folds(train$Class, CONFIG$CV_FOLDS, seed)
  cv_preds <- matrix(NA_real_, nrow(train), length(CLASSIFIERS) * length(REMEDIES))
  te_preds <- matrix(NA_real_, nrow(test), ncol(cv_preds))
  names_vec <- character(0)
  idx <- 0L
  for (classifier in CLASSIFIERS) for (remedy in REMEDIES) {
    idx <- idx + 1L
    nm <- candidate_name(classifier, remedy)
    names_vec <- c(names_vec, nm)
    log_message("  fitting ", nm, " (", idx, "/", ncol(cv_preds), ")")
    cv_preds[, idx] <- out_of_fold(train, classifier, remedy, folds,
                                  seed = seed + 100L * idx)
    final_fit <- fit_classifier(train, classifier, remedy,
                                seed = seed + 100000L + idx)
    te_preds[, idx] <- predict_classifier(final_fit, test)
    rm(final_fit); gc(verbose = FALSE)
  }
  colnames(cv_preds) <- colnames(te_preds) <- names_vec

  # Every candidate produces a default 0.5 rule and a CV-selected F1 rule.
  # Baselines also get CV-selected F2 rule to expose recall/precision tradeoff.
  # PR/ROC AUC cannot change merely by changing the cutoff.
  rows <- list(); row_no <- 0L
  append_row <- function(classifier, remedy, rule, threshold, oof_p, test_p,
                         addon = list()) {
    cv_m <- metrics(train$Class, oof_p, threshold)
    test_m <- metrics(test$Class, test_p, threshold)
    df <- data.frame(
      Experiment = name, Target_Prevalence = target_prevalence,
      Train_N = nrow(train), Test_N = nrow(test),
      Train_Positive_Count = sum(train$Class == "1"),
      Test_Positive_Count = sum(test$Class == "1"),
      Classifier = classifier, Remedy = remedy,
      Operating_Rule = rule, Threshold = threshold,
      CV_ROC_AUC = cv_m$ROC_AUC, CV_PR_AUC_AP = cv_m$PR_AUC_AP,
      CV_F1_Selected = cv_m$F1,
      test_m[, setdiff(names(test_m), "Threshold"), drop = FALSE],
      stringsAsFactors = FALSE
    )
    row_no <<- row_no + 1L
    rows[[row_no]] <<- df
    invisible(NULL)
  }
  append_row("Majority_Class", "predict_all_negative", "all_negative", 0.5,
             majority_prediction(train$Class), majority_prediction(test$Class))

  for (nm in names_vec) {
    bits <- strsplit(nm, "__", fixed = TRUE)[[1]]
    oof_p <- cv_preds[, nm]; te_p <- te_preds[, nm]
    append_row(bits[1], bits[2], "default_0.5", 0.5, oof_p, te_p)
    th1 <- tune_threshold(train$Class, oof_p, beta = 1)
    append_row(bits[1], bits[2], "CV_tuned_F1", th1, oof_p, te_p)
    if (bits[2] == "baseline") {
      th2 <- tune_threshold(train$Class, oof_p, beta = 2)
      append_row(bits[1], bits[2], "CV_tuned_F2", th2, oof_p, te_p)
    }
  }

  ens <- make_ensemble(cv_preds, te_preds, train$Class)
  append_row("Ensemble", "CV_optimized_blend", "default_0.5", 0.5,
             ens$oof, ens$test)
  ens_cut <- tune_threshold(train$Class, ens$oof, beta = 1)
  append_row("Ensemble", "CV_optimized_blend", "CV_tuned_F1", ens_cut,
             ens$oof, ens$test)
  if (!is.null(true_test_prob)) {
    # Useful upper reference for discrimination: population true conditional P(Y=1|X).
    # This is an ORACLE BENCHMARK, never an available real-world method.
    append_row("Oracle_True_P", "simulation_truth", "default_0.5", 0.5,
               rep(mean(train$Class == "1"), nrow(train)), true_test_prob)
  }

  scores <- do.call(rbind, rows)
  write_csv(scores, paste0(name, "__test_metrics.csv"))
  oof_metadata <- data.frame(Truth = as.integer(as.character(train$Class)),
                             cv_preds, check.names = FALSE)
  # Do not export massive OOF predictions by default; test scores suffice.
  if (isTRUE(CONFIG$SAVE_OOF)) {
    write_csv(oof_metadata, paste0(name, "__training_OOF_probabilities.csv"))
  }
  ens$weights$Experiment <- name
  ens$weights$Selected_Weight <- ens$weight
  ens$weights$Selected_Logistic <- ens$logit_model
  ens$weights$Selected_Forest <- ens$forest_model
  write_csv(ens$weights, paste0(name, "__ensemble_weight_search.csv"))

  # Choose best single learner STRICTLY BY OOF AP (not the test results).
  cv_ap <- vapply(colnames(cv_preds), function(nm) {
    average_precision(train$Class, cv_preds[, nm])
  }, numeric(1))
  best_single_name <- names(which.max(cv_ap))
  paired_test <- bootstrap_ensemble_difference(
    test$Class, ens$test, te_preds[, best_single_name],
    B = CONFIG$BOOTSTRAP_REPS, seed = seed + 900000L)
  paired_test$Experiment <- name
  paired_test$Best_Single_CV_Selected <- best_single_name
  paired_test$Ensemble_Logistic_Selected <- ens$logit_model
  paired_test$Ensemble_Forest_Selected <- ens$forest_model
  paired_test$Optimal_Logistic_Weight <- ens$weight
  write_csv(paired_test, paste0(name, "__ensemble_vs_single_bootstrap.csv"))

  if (is_real) {
    score_df <- data.frame(
      Truth = as.integer(as.character(test$Class)),
      Logistic_Baseline = te_preds[, candidate_name("Penalized_Logistic", "baseline")],
      RF_Baseline = te_preds[, candidate_name("Random_Forest", "baseline")],
      Best_Single = te_preds[, best_single_name],
      Ensemble = ens$test
    )
    write_csv(score_df, paste0(name, "__heldout_predictions.csv"))
    plot_real_curves(score_df, name)
  }
  log_message("FINISH: ", name, " | best single by CV AP: ",
              best_single_name, " | ensemble logistic share: ", ens$weight)
  list(scores = scores, ensemble = paired_test,
       train_rate = mean(train$Class == "1"),
       test_rate = mean(test$Class == "1"))
}

# ============================ 8. PLOTTING =====================================
curve_data <- function(y, p, curve = c("PR", "ROC"), max_points = 1400L) {
  curve <- match.arg(curve)
  y <- as.integer(y)
  ord <- order(p, decreasing = TRUE)
  yy <- y[ord]; pp <- p[ord]
  ends <- which(!duplicated(pp, fromLast = TRUE))
  # Decimation is for display only, all data are used in AP/AUC calculation.
  kept <- unique(c(seq(1L, length(ends),
                       by = max(1L, floor(length(ends) / max_points))),
                   length(ends)))
  indices <- ends[kept]
  ctp <- cumsum(yy)
  tp <- ctp[indices]
  fp <- indices - tp
  if (curve == "PR") {
    data.frame(X = c(0, tp / sum(y)),
               Y = c(1, tp / indices))
  } else {
    data.frame(X = c(0, fp / sum(y == 0)),
               Y = c(0, tp / sum(y == 1)))
  }
}

plot_real_curves <- function(preds, experiment_name) {
  names_to_plot <- setdiff(names(preds), "Truth")
  curves <- list()
  for (name in names_to_plot) for (typ in c("PR", "ROC")) {
    cdf <- curve_data(preds$Truth, preds[[name]], typ)
    cdf$Model <- name; cdf$Curve <- typ
    curves[[length(curves) + 1L]] <- cdf
  }
  curves <- do.call(rbind, curves)
  for (typ in c("PR", "ROC")) {
    d <- curves[curves$Curve == typ, , drop = FALSE]
    fig <- ggplot2::ggplot(d, ggplot2::aes(X, Y, colour = Model)) +
      ggplot2::geom_line(linewidth = 0.8) +
      ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::labs(
        title = paste(experiment_name, typ, "curve (held-out test)"),
        x = if (typ == "PR") "Recall" else "False positive rate",
        y = if (typ == "PR") "Precision" else "True positive rate"
      )
    ggplot2::ggsave(file.path(CONFIG$PLOT_CURVE_DIR,
                              paste0(experiment_name, "__", typ, ".png")),
                    fig, width = 9, height = 5.5, dpi = 200)
  }
}

plot_overview <- function(all_scores) {
  # One row per model/remedy, at its prespecified 0.5 cutoff (fair ranking
  # comparison not confounded by operating-point tuning).
  z <- all_scores[all_scores$Operating_Rule == "default_0.5" &
                    all_scores$Classifier %in% CLASSIFIERS, , drop = FALSE]
  if (!nrow(z)) return(invisible(NULL))
  z$Model_Remedy <- paste(z$Classifier, z$Remedy, sep = " / ")
  p <- ggplot2::ggplot(z, ggplot2::aes(x = Prevalence * 100,
                                      y = PR_AUC_AP, colour = Model_Remedy)) +
    ggplot2::geom_point(size = 1.5) + ggplot2::geom_line() +
    ggplot2::facet_wrap(~ Experiment, scales = "free_y") +
    ggplot2::labs(title = "Held-out average precision across experiments",
                  x = "Observed test fraud prevalence (%)",
                  y = "PR-AUC (Average Precision)") +
    ggplot2::theme_minimal(base_size = 10)
  ggplot2::ggsave(file.path(CONFIG$PLOT_FINAL_DIR, "all_experiments__AP.png"),
                  p, width = 16, height = 11, dpi = 160)

  s <- all_scores[all_scores$Operating_Rule %in% c("default_0.5", "CV_tuned_F1") &
                    all_scores$Classifier %in% CLASSIFIERS, , drop = FALSE]
  s$Variant <- paste(s$Remedy, s$Operating_Rule, sep = " / ")
  q <- ggplot2::ggplot(s, ggplot2::aes(x = Sensitivity, y = Precision,
                                      colour = Variant, shape = Classifier)) +
    ggplot2::geom_point(alpha = 0.8, size = 2) +
    ggplot2::facet_wrap(~ Experiment) +
    ggplot2::labs(title = "Precision-recall operating points on untouched tests",
                  x = "Sensitivity / Recall", y = "Precision") +
    ggplot2::theme_minimal(base_size = 10)
  ggplot2::ggsave(file.path(CONFIG$PLOT_FINAL_DIR, "all_experiments__operating_points.png"),
                  q, width = 16, height = 11, dpi = 160)
}


# ============================== EDA BEFORE FITTING ============================
# EDA is descriptive and never selects model features, thresholds, or parameters.
# It is performed on the original population sample BEFORE splitting/resampling.
plot_simulation_eda <- function(df, scenario_name) {
  d <- df
  fig1 <- ggplot2::ggplot(d, ggplot2::aes(x = Class, fill = Class)) +
    ggplot2::geom_bar(width = .66, show.legend = FALSE) +
    ggplot2::scale_y_log10() +
    ggplot2::labs(title = paste(scenario_name, "class counts"),
                  subtitle = "Log scale exposes rare positives", x = "Observed outcome",
                  y = "Transaction count (log scale)") +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(file.path(CONFIG$PLOT_EDA_DIR,
                  paste0(scenario_name, "__EDA_class_counts.png")),
                  fig1, width = 7.4, height = 4.6, dpi = 170)
  long <- data.frame(X1 = d$X1, Class = d$Class)
  fig2 <- ggplot2::ggplot(long, ggplot2::aes(x = X1, fill = Class)) +
    ggplot2::geom_density(alpha = .35, adjust = 1.2) +
    ggplot2::labs(title = paste(scenario_name, "X1 by outcome"),
                  subtitle = "Class overlap is retained despite intercept changes") +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(file.path(CONFIG$PLOT_EDA_DIR,
                  paste0(scenario_name, "__EDA_X1_density.png")),
                  fig2, width = 7.4, height = 4.6, dpi = 170)
}

plot_simulation_design <- function(grid) {
  d <- grid
  d$Expected_Positives <- d$N * d$Prevalence
  fig <- ggplot2::ggplot(d, ggplot2::aes(x = factor(N), y = Expected_Positives,
                                fill = factor(Prevalence))) +
    ggplot2::geom_col(position = "dodge") +
    ggplot2::scale_y_log10() +
    ggplot2::labs(title = "Simulation design: fraction versus absolute count",
                  subtitle = "Same fraction can imply very different minority information",
                  x = "Sample size", y = "Expected positive cases (log scale)",
                  fill = "Target prevalence") +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(file.path(CONFIG$PLOT_EDA_DIR, "simulation__EDA_design.png"),
                  fig, width = 9, height = 5.2, dpi = 180)
}

# ============================== EXPERIMENT DESIGN =============================
# QUICK CHECK: reduce SIM_REPETITIONS to 1, SIM_SIZES to 2000L,
# ADD_LARGE_RARE_SCENARIO to FALSE, and RF_TREES to 50.
CONFIG$SIM_PREVALENCES <- c(0.50, 0.20, 0.05, 0.01)
CONFIG$SIM_SIZES <- c(2000L, 20000L)
CONFIG$SIM_REPETITIONS <- 3L  # increase to 10-30 for final Monte Carlo inference
CONFIG$ADD_LARGE_RARE_SCENARIO <- TRUE

set.seed(CONFIG$SEED)
MC_X <- matrix(rnorm(100000L * length(BETA)), ncol = length(BETA))
MC_ETA <- drop(MC_X %*% BETA)
SIM_INTERCEPTS <- vapply(CONFIG$SIM_PREVALENCES, get_intercept, numeric(1))
names(SIM_INTERCEPTS) <- as.character(CONFIG$SIM_PREVALENCES)

grid <- expand.grid(Prevalence = CONFIG$SIM_PREVALENCES,
                    N = CONFIG$SIM_SIZES,
                    Repeat = seq_len(CONFIG$SIM_REPETITIONS))
if (CONFIG$ADD_LARGE_RARE_SCENARIO) {
  grid <- rbind(grid, data.frame(Prevalence = 0.01, N = 100000L,
                        Repeat = seq_len(CONFIG$SIM_REPETITIONS)))
}
write_csv(grid, "simulation__design_grid.csv")
write_compressed_data(grid, "simulation_design.csv.gz", CONFIG$DESIGN_DATA_DIR)
plot_simulation_design(unique(grid[c("Prevalence", "N")]))
write_csv(data.frame(Target_Prevalence = CONFIG$SIM_PREVALENCES,
            Intercept = unname(SIM_INTERCEPTS),
            Beta_X1 = BETA["X1"], Beta_X2 = BETA["X2"],
            Beta_X3 = BETA["X3"], Beta_X4 = BETA["X4"]),
          "simulation__population_coefficients.csv")

all_results <- list(); all_ensembles <- list(); scenario_rows <- list()
for (j in seq_len(nrow(grid))) {
  rate <- grid$Prevalence[j]; n <- grid$N[j]
  seed <- CONFIG$SEED + 101L * j
  obj <- simulate_dataset(n, rate, seed)
  d <- obj$data
  # This is exploratory EDA only; any transforms must be learned from train.
  name <- sprintf("sim_n%d_p%02d_rep%02d", n, round(rate * 100), grid$Repeat[j])
if (grid$Repeat[j] == 1L) plot_simulation_eda(d, name)
# Store an independent, reproducible dataset for each Monte Carlo repeat.
# The oracle P(Y=1|X) is for audit only; it must NEVER become a predictor.
if (isTRUE(CONFIG$SAVE_SIMULATED_DATA)) {
  stored_data <- d
  if (isTRUE(CONFIG$SAVE_TRUE_PROBABILITY)) {
    stored_data$Oracle_True_Probability_DO_NOT_MODEL <- obj$truth
  }
  write_compressed_data(stored_data, paste0(name, ".csv.gz"),
                        CONFIG$GENERATED_DATA_DIR)
  rm(stored_data)
}
  # Use an indexed copy to recover precisely the oracle test probabilities.
  indexed <- d; indexed$.source_row <- seq_len(nrow(d))
  split_idx <- stratified_split(indexed, CONFIG$TEST_FRACTION, seed + 1L)
  ids_test <- split_idx$test$.source_row
  split_idx$train$.source_row <- split_idx$test$.source_row <- NULL
  result <- experiment(split_idx$train, split_idx$test, name,
            seed = seed + 10L, target_prevalence = rate,
            true_test_prob = obj$truth[ids_test], is_real = FALSE)
  all_results[[j]] <- result$scores
  all_ensembles[[j]] <- result$ensemble
  scenario_rows[[j]] <- data.frame(Experiment = name, N = n,
    Repeat = grid$Repeat[j], Target_Prevalence = rate,
    Observed_Prevalence = mean(d$Class == "1"),
    Observed_Positives = sum(d$Class == "1"),
    Train_Positives = sum(split_idx$train$Class == "1"),
    Test_Positives = sum(split_idx$test$Class == "1"),
    Population_Intercept = obj$intercept)
  write_csv(do.call(rbind, all_results), "simulation__all_metrics.csv")
  write_csv(do.call(rbind, all_ensembles), "simulation__ensemble_comparisons.csv")
  write_csv(do.call(rbind, scenario_rows), "simulation__scenario_summary.csv")
}
final_results <- do.call(rbind, all_results)
# Per-replicate comparisons are paired by exactly the SAME simulated test data.
# Do NOT interpret repeated rows within one replicate as independent samples.
base <- subset(final_results, Classifier %in% CLASSIFIERS & Remedy == "baseline" &
                            Operating_Rule == "default_0.5")
candidates <- subset(final_results, Classifier %in% CLASSIFIERS &
                     !(Remedy == "baseline" & Operating_Rule == "default_0.5"))
comp <- merge(candidates, base[, c("Experiment", "Classifier", "F1", "PR_AUC_AP",
                "Sensitivity", "Precision", "Balanced_Accuracy")],
              by = c("Experiment", "Classifier"), suffixes = c("", "_Base"))
for (metric in c("F1", "PR_AUC_AP", "Sensitivity", "Precision", "Balanced_Accuracy")) {
  comp[[paste0("Delta_", metric)]] <- comp[[metric]] - comp[[paste0(metric, "_Base")]]
}
write_csv(comp, "simulation__remedy_deltas.csv")
# Summarize across independent simulation replications rather than showing
# one crowded panel for every repetition.
vis <- subset(final_results, Classifier %in% CLASSIFIERS &
                            Operating_Rule == "default_0.5")
mean_ap <- aggregate(PR_AUC_AP ~ Target_Prevalence + Train_N + Classifier + Remedy,
                     data = vis, FUN = mean)
mean_ap$Training_Size <- factor(mean_ap$Train_N,
              levels = sort(unique(mean_ap$Train_N)))
fig_summary <- ggplot2::ggplot(mean_ap,
    ggplot2::aes(x = Target_Prevalence * 100, y = PR_AUC_AP,
                 colour = Remedy, group = Remedy)) +
  ggplot2::geom_line(linewidth = .7) + ggplot2::geom_point(size = 2) +
  ggplot2::facet_grid(Classifier ~ Training_Size, scales = "free_y") +
  ggplot2::labs(title = "Mean held-out average precision across Monte Carlo runs",
                subtitle = "Scores use the untouched 0.5-threshold probability ranking",
                x = "Target positive prevalence (%)", y = "Average precision (AP)") +
  ggplot2::theme_minimal(base_size = 11)
ggplot2::ggsave(file.path(CONFIG$PLOT_FINAL_DIR, "simulation__AP_summary.png"),
                fig_summary, width = 12.6, height = 7.8, dpi = 180)
write_csv(mean_ap, "simulation__aggregated_AP.csv")
plot_overview(final_results)

# ==================== 10. FINAL PRESENTATION FIGURES =========================
# No plots use training-set resampling for metrics or choose the best method by
# looking at held-out test labels. All figures are DESCRIPTIVE comparisons.
# Monte Carlo summaries are means/SD over independent replicate datasets.

save_final_figure <- function(p, name, width = 11, height = 7) {
  ggplot2::ggsave(file.path(CONFIG$PLOT_FINAL_DIR, name), p,
                  width = width, height = height, dpi = 300, bg = "white")
}

publication_theme <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = 15),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "bottom",
      strip.text = ggplot2::element_text(face = "bold")
    )
}

plot_final_simulation_results <- function(scores, paired_deltas, comparison, meta) {
  base_models <- scores[scores$Classifier %in% CLASSIFIERS, , drop = FALSE]
  default <- base_models[base_models$Operating_Rule == "default_0.5", , drop = FALSE]
  default$Train_Size <- factor(default$Train_N, levels = sort(unique(default$Train_N)))

  # One value per scenario / replicate; group means are NOT pooled test labels.
  grouping <- c("Target_Prevalence", "Train_Size", "Classifier", "Remedy")
  aggr <- stats::aggregate(default[c("PR_AUC_AP", "ROC_AUC", "Balanced_Accuracy")],
                           default[grouping], mean, na.rm = TRUE)
  names(aggr)[names(aggr) == "PR_AUC_AP"] <- "Mean_AP"
  names(aggr)[names(aggr) == "ROC_AUC"] <- "Mean_ROC_AUC"
  names(aggr)[names(aggr) == "Balanced_Accuracy"] <- "Mean_Balanced_Accuracy"
  write_csv(aggr, "simulation__final_plot_summary.csv")

  metric_curves <- list(
    list(variable = "Mean_AP", title = "Average precision across prevalence and sample size",
         y = "Held-out average precision", file = "01_Average_Precision_by_Prevalence.png"),
    list(variable = "Mean_ROC_AUC", title = "ROC-AUC across prevalence and sample size",
         y = "Held-out ROC-AUC", file = "02_ROC_AUC_by_Prevalence.png"),
    list(variable = "Mean_Balanced_Accuracy", title = "Balanced accuracy across prevalence and sample size",
         y = "Held-out balanced accuracy", file = "03_Balanced_Accuracy_by_Prevalence.png")
  )
  for (spec in metric_curves) {
    p <- ggplot2::ggplot(aggr,
             ggplot2::aes(x = Target_Prevalence * 100, y = .data[[spec$variable]],
                          colour = Remedy, group = Remedy)) +
      ggplot2::geom_line(linewidth = 0.8) + ggplot2::geom_point(size = 2) +
      ggplot2::facet_grid(Classifier ~ Train_Size, scales = "free_y") +
      ggplot2::scale_x_continuous(breaks = c(1, 5, 20, 50)) +
      ggplot2::labs(title = spec$title,
                    subtitle = "Points are Monte Carlo means; unresampled independent held-out tests",
                    x = "Target positive prevalence (%)", y = spec$y,
                    colour = "Training remedy") + publication_theme()
    save_final_figure(p, spec$file, width = 13, height = 7.5)
  }

  # Threshold-only comparisons on baseline learner: rankings are unchanged.
  th <- base_models[base_models$Remedy == "baseline" &
             base_models$Operating_Rule %in% c("default_0.5", "CV_tuned_F1"), , drop = FALSE]
  th$Train_Size <- factor(th$Train_N, levels = sort(unique(th$Train_N)))
  thsum <- stats::aggregate(th["F1"],
          th[c("Target_Prevalence", "Train_Size", "Classifier", "Operating_Rule")],
          mean, na.rm = TRUE)
  write_csv(thsum, "simulation__final_threshold_comparison.csv")
  pt <- ggplot2::ggplot(thsum,
           ggplot2::aes(x = Target_Prevalence * 100, y = F1,
                        colour = Operating_Rule, group = Operating_Rule)) +
    ggplot2::geom_line(linewidth = 0.8) + ggplot2::geom_point(size = 2.2) +
    ggplot2::facet_grid(Classifier ~ Train_Size, scales = "free_y") +
    ggplot2::scale_x_continuous(breaks = c(1, 5, 20, 50)) +
    ggplot2::labs(title = "Does threshold tuning improve test-set F1?",
                  subtitle = "Same baseline risk scores; threshold chosen using training OOF only",
                  x = "Target prevalence (%)", y = "Held-out F1", colour = "Operating rule") +
    publication_theme()
  save_final_figure(pt, "04_Default_vs_CV_Tuned_Threshold_F1.png", 13, 7.5)

  # Paired same-dataset remedy minus unmodified baseline effects.
  d <- paired_deltas[paired_deltas$Operating_Rule == "default_0.5" &
                       paired_deltas$Remedy != "baseline", , drop = FALSE]
  d$Train_Size <- factor(d$Train_N, levels = sort(unique(d$Train_N)))
  ds <- stats::aggregate(d["Delta_PR_AUC_AP"],
                          d[c("Target_Prevalence", "Train_Size", "Classifier", "Remedy")],
                          mean, na.rm = TRUE)
  write_csv(ds, "simulation__final_paired_AP_improvement.csv")
  pd <- ggplot2::ggplot(ds,
          ggplot2::aes(x = Target_Prevalence * 100, y = Delta_PR_AUC_AP,
                       colour = Remedy, group = Remedy)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "grey45") +
    ggplot2::geom_line(linewidth = 0.8) + ggplot2::geom_point(size = 2) +
    ggplot2::facet_grid(Classifier ~ Train_Size, scales = "free_y") +
    ggplot2::scale_x_continuous(breaks = c(1, 5, 20, 50)) +
    ggplot2::labs(title = "Paired improvement in average precision over baseline",
                  subtitle = "Positive = improvement on exactly the same held-out observations",
                  x = "Target prevalence (%)", y = "Remedy AP minus baseline AP",
                  colour = "Remedy") + publication_theme()
  save_final_figure(pd, "05_Paired_Remedy_AP_Improvement.png", 13, 7.5)

  # An operating-point visualization, not a single-metric model ranking.
  point_d <- base_models[base_models$Operating_Rule == "CV_tuned_F1", , drop = FALSE]
  pts <- stats::aggregate(point_d[c("Precision", "Sensitivity")],
         point_d[c("Target_Prevalence", "Train_N", "Classifier", "Remedy")],
         mean, na.rm = TRUE)
  pts$Scenario <- paste0("n=", format(pts$Train_N, scientific = FALSE),
                         " · ", 100 * pts$Target_Prevalence, "%")
  write_csv(pts, "simulation__final_precision_recall_points.csv")
  pp <- ggplot2::ggplot(pts,
          ggplot2::aes(x = Sensitivity, y = Precision, colour = Remedy,
                       shape = Classifier)) +
    ggplot2::geom_point(size = 2.4, alpha = 0.9) +
    ggplot2::facet_wrap(~ Scenario, ncol = 3) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(title = "Precision–recall trade-off at CV-tuned F1 thresholds",
                  subtitle = "Each point is the mean held-out operating point across independent runs",
                  x = "Sensitivity (recall)", y = "Precision") + publication_theme()
  save_final_figure(pp, "06_Precision_Recall_Tradeoff.png", 13.5, 9.0)

  # Distinguish prevalence fraction from the ABSOLUTE information available.
  cb <- default[default$Remedy == "baseline", , drop = FALSE]
  pc <- ggplot2::ggplot(cb,
          ggplot2::aes(x = Train_Positive_Count, y = PR_AUC_AP,
                       colour = factor(Target_Prevalence * 100))) +
    ggplot2::geom_point(alpha = 0.75, size = 2.1) +
    ggplot2::scale_x_log10() +
    ggplot2::facet_wrap(~ Classifier, nrow = 1) +
    ggplot2::labs(title = "Minority sample count versus held-out average precision",
                  subtitle = "Points are individual Monte Carlo repeats, not pooled aggregates",
                  x = "Number of minority cases in training (log scale)",
                  y = "Held-out average precision", colour = "Target prevalence (%)") +
    publication_theme()
  save_final_figure(pc, "07_Minority_Count_vs_Average_Precision.png", 12, 5.5)

  # Formal ensemble comparison uses OOF-selected single learner as comparator.
  ens <- merge(comparison, meta[c("Experiment", "Target_Prevalence", "N")],
               by = "Experiment", all.x = TRUE)
  ens$Scenario <- paste0("n=", ens$N, ", ", ens$Target_Prevalence * 100, "%")
  write_csv(ens, "simulation__final_ensemble_plot_data.csv")
  ens$Label <- paste0(ens$Scenario, " · ", ens$Experiment)
  pe <- ggplot2::ggplot(ens,
          ggplot2::aes(x = Label, y = Ensemble_minus_Single_AP)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "grey45") +
    ggplot2::geom_pointrange(ggplot2::aes(ymin = Bootstrap_CI_Lower,
                                        ymax = Bootstrap_CI_Upper),
                             colour = "#147D92", linewidth = 0.55) +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "Ensemble minus CV-selected best single model",
                  subtitle = "Paired, stratified bootstrap CIs for each test-set replicate (descriptive)",
                  x = "Scenario / replicate", y = "Test average-precision difference") +
    publication_theme() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 7))
  # Scenario repetitions retain distinct factor labels on the vertical axis.
  save_final_figure(pe, "08_Ensemble_vs_Best_Single_AP.png", 12.5,
                    max(7, min(17, 0.32 * nrow(ens) + 2.5)))
}

plot_final_simulation_results(final_results, comp,
                              do.call(rbind, all_ensembles),
                              do.call(rbind, scenario_rows))

writeLines(capture.output(str(CONFIG)), file.path(CONFIG$METADATA_DIR, "simulation__configuration.txt"))
writeLines(capture.output(sessionInfo()), file.path(CONFIG$METADATA_DIR, "simulation__sessionInfo.txt"))
log_message("Simulation finished. Tables: ", CONFIG$TABLE_DIR,
            " | Final figures: ", CONFIG$PLOT_FINAL_DIR,
            " | Generated data: ", CONFIG$GENERATED_DATA_DIR)

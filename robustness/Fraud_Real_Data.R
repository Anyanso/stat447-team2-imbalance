################################################################################
# REVISION 3: LOCAL DOWNLOADS DATA AND SIMULATION RESULTS ORGANIZATION
# In RStudio open your .Rproj; then source this file from that project directory.
# All generated artifacts are stored under results/simulation/fraud_real_data/.
################################################################################
################################################################################
# STAT 847 -- Team 2 | University of Saskatchewan | Dr. Li Xing | Fall 2026
# RESEARCH QUESTION: Can class-imbalance remedies improve prediction?
# This file is SELF-CONTAINED: no dependency on the other R script.
# FILE LOCATIONS: results/simulation/fraud_real_data/ under project root.
# REPRODUCIBILITY: fixed seeds, source code, data provenance and sessionInfo.
# CRITICAL: Keep the entire held-out TEST set untouched and at natural prevalence.
################################################################################
options(warn = 1)
set.seed(847L)

# FILE 2 OF 3: SOURCE-DOWNLOADED REAL FRAUD DATA, EDA, AND MODELING.
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

# Open the RStudio .Rproj first. The project directory is the working directory.
CONFIG$PROJECT_ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

# IMPORTANT: This points to the CSV you already downloaded. No Kaggle CLI/API.
# For example: "data/creditcard.csv" (forward slashes in R).
# Change this one setting if your dataset is elsewhere.
CONFIG$LOCAL_CSV <- "data/creditcard.csv"

# A self-contained named subfolder of the SIMULATION results directory.
# It cannot overwrite simulation tables or plots.
CONFIG$OUTPUT_DIR <- file.path(CONFIG$PROJECT_ROOT, "results", "simulation", "fraud_real_data")
CONFIG$TABLE_DIR <- file.path(CONFIG$OUTPUT_DIR, "tables")
CONFIG$METADATA_DIR <- file.path(CONFIG$OUTPUT_DIR, "metadata")
CONFIG$DATA_DIR <- file.path(CONFIG$OUTPUT_DIR, "data", "raw")
CONFIG$PROCESSED_DATA_DIR <- file.path(CONFIG$OUTPUT_DIR, "data", "processed")
CONFIG$SPLIT_DIR <- file.path(CONFIG$OUTPUT_DIR, "data", "splits")
CONFIG$PLOT_EDA_DIR <- file.path(CONFIG$OUTPUT_DIR, "plots", "eda")
CONFIG$PLOT_FINAL_DIR <- file.path(CONFIG$OUTPUT_DIR, "plots", "final")
CONFIG$PLOT_CURVE_DIR <- file.path(CONFIG$OUTPUT_DIR, "plots", "curves")
CONFIG$SAVE_PROCESSED_DATA <- TRUE
CONFIG$SAVE_RAW_COPY <- TRUE  # preserve the exact input used with the outputs

for (path in c(CONFIG$TABLE_DIR, CONFIG$METADATA_DIR, CONFIG$DATA_DIR,
               CONFIG$PROCESSED_DATA_DIR, CONFIG$SPLIT_DIR,
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

# ======================= 9. LOCAL DOWNLOADED FRAUD DATA =======================
load_local_fraud <- function() {
  csv_path <- path.expand(CONFIG$LOCAL_CSV)
  if (!file.exists(csv_path)) {
    stop("Local fraud CSV not found: ", csv_path,
         "\nSet CONFIG$LOCAL_CSV to your actual downloaded creditcard.csv path, ",
         "e.g. C:/Users/HP/Downloads/creditcard.csv. No Kaggle access is required.")
  }
  if (file.info(csv_path)$size <= 0) stop("Local CSV is empty: ", csv_path)
  log_message("Reading downloaded local file: ", normalizePath(csv_path, winslash = "/"))

  # Save a copy of the original CSV next to the study outputs for provenance.
  if (isTRUE(CONFIG$SAVE_RAW_COPY)) {
    dest <- file.path(CONFIG$DATA_DIR, "creditcard.csv")
    if (normalizePath(csv_path, winslash = "/", mustWork = TRUE) !=
        normalizePath(dest, winslash = "/", mustWork = FALSE)) {
      if (!file.copy(csv_path, dest, overwrite = TRUE)) {
        stop("Could not save a copy of the source CSV in ", CONFIG$DATA_DIR)
      }
    }
  }

  d <- as.data.frame(data.table::fread(csv_path, showProgress = FALSE))
  expected <- c("Time", paste0("V", 1:28), "Amount", "Class")
  if (!all(expected %in% names(d))) {
    stop("Unexpected local dataset columns; expected Time, V1...V28, Amount, Class.")
  }
  if (nrow(d) != 284807L || sum(d$Class == 1L, na.rm = TRUE) != 492L) {
    stop("Unexpected dataset contents: expected 284,807 rows and 492 frauds. ",
         "Verify that your local CSV is the original ULB creditcard.csv dataset.")
  }
  if (anyNA(d)) stop("Unexpected missing values in local dataset")
  # Predeclared transformation, NOT selected using test outcomes.
  d$Amount <- log1p(d$Amount)
  d$Class <- factor(as.character(d$Class), levels = c("0", "1"))
  log_message("Verified local ULB dataset: ", nrow(d), " transactions, ",
              sum(d$Class == "1"), " frauds")
  d
}

# =============================== EDA BEFORE MODELING ==========================
# All plots inspect ORIGINAL records; no resampling or model fit occurs here.
# EDA uses labels only descriptively, NEVER for test-specific tuning.
plot_fraud_eda <- function(d) {
  class_tbl <- as.data.frame(table(d$Class))
  names(class_tbl) <- c("Class", "Count")
  class_tbl$Percent <- 100 * class_tbl$Count / nrow(d)
  write_csv(class_tbl, "fraud__EDA_class_frequency.csv")
  missing_tbl <- data.frame(Variable = names(d),
                            Missing = vapply(d, function(x) sum(is.na(x)), numeric(1)))
  write_csv(missing_tbl, "fraud__EDA_missingness.csv")
  amount_summary <- data.frame(
      Class = c("Legitimate", "Fraud"),
      N = c(sum(d$Class == "0"), sum(d$Class == "1")),
      Mean_Amount = c(mean(d$Amount[d$Class == "0"]), mean(d$Amount[d$Class == "1"])),
      Median_Amount = c(median(d$Amount[d$Class == "0"]), median(d$Amount[d$Class == "1"])),
      P95_Amount = c(unname(quantile(d$Amount[d$Class == "0"], .95)),
                     unname(quantile(d$Amount[d$Class == "1"], .95))))
  write_csv(amount_summary, "fraud__EDA_amount_summary.csv")
  duplicate_flag <- duplicated(d)
  write_csv(data.frame(Original_N = nrow(d),
                       Exact_Duplicate_Count = sum(duplicate_flag),
                       Exact_Duplicate_Frauds = sum(duplicate_flag & d$Class == "1"),
                       N_After_Deduplication = sum(!duplicate_flag)),
            "fraud__EDA_duplicate_audit.csv")

  p1 <- ggplot2::ggplot(class_tbl, ggplot2::aes(x = Class, y = Count, fill = Class)) +
    ggplot2::geom_col(width = .6, show.legend = FALSE) +
    ggplot2::scale_y_log10() +
    ggplot2::geom_text(ggplot2::aes(label = format(Count, big.mark = ",")),
                       vjust = -0.3, size = 4) +
    ggplot2::labs(title = "ULB fraud: the extreme base-rate problem",
                   subtitle = "Original, unsampled transaction counts (log axis)",
                   x = "0 = legitimate | 1 = fraud", y = "Count (log scale)") +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(file.path(CONFIG$PLOT_EDA_DIR,"fraud__EDA_class_balance.png"),
                  p1, width = 8, height = 4.8, dpi = 180)

  p2 <- ggplot2::ggplot(d, ggplot2::aes(x = log1p(Amount), fill = Class)) +
    ggplot2::geom_density(alpha = .35, adjust = 1.5) +
    ggplot2::labs(title = "Amount distributions by fraud status",
                  subtitle = "Log(1 + Amount) shown for heavy right tail",
                  x = "log(1 + transaction amount)", y = "Density") +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(file.path(CONFIG$PLOT_EDA_DIR,"fraud__EDA_amount_density.png"),
                  p2, width = 8, height = 4.8, dpi = 180)

  d$Hour <- floor(d$Time / 3600)
  hourly <- aggregate(as.integer(as.character(d$Class)), list(Hour = d$Hour),
                      FUN = function(z) c(N = length(z), Fraud = sum(z)))
  hourly <- data.frame(Hour = hourly$Hour, N = hourly$x[, "N"],
                       Fraud = hourly$x[, "Fraud"])
  hourly$Rate <- hourly$Fraud / hourly$N
  write_csv(hourly, "fraud__EDA_hourly_prevalence.csv")
  p3 <- ggplot2::ggplot(hourly, ggplot2::aes(Hour, Rate * 100)) +
    ggplot2::geom_line(linewidth = .8) + ggplot2::geom_point(size = 1.0) +
    ggplot2::labs(title = "Observed fraud prevalence across elapsed hours",
                  subtitle = "Descriptive only; no deployment drift conclusion from two days",
                  x = "Elapsed hour from earliest transaction", y = "Fraud (%)") +
    ggplot2::theme_minimal(base_size = 12)
  ggplot2::ggsave(file.path(CONFIG$PLOT_EDA_DIR,"fraud__EDA_hourly_fraud_rate.png"),
                  p3, width = 8, height = 4.8, dpi = 180)

  selected <- c("V1", "V4", "V10", "V14")
  long <- do.call(rbind, lapply(selected, function(v) {
    data.frame(Feature = v, Value = d[[v]], Class = d$Class)
  }))
  p4 <- ggplot2::ggplot(long, ggplot2::aes(x = Value, fill = Class)) +
    ggplot2::geom_density(alpha = .35) +
    ggplot2::facet_wrap(~Feature, scales = "free", ncol = 2) +
    ggplot2::labs(title = "Illustrative anonymized PCA features by class",
                  subtitle = "Not a feature-selection procedure",
                  x = "Feature value", y = "Density") +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(file.path(CONFIG$PLOT_EDA_DIR,"fraud__EDA_feature_distributions.png"),
                  p4, width = 9, height = 6.6, dpi = 170)
}

# ================================ REAL DATA STUDY =============================
# Local downloaded CSV; schema and source-count checks occur automatically.
# No internet, Kaggle CLI, Kaggle account, or token is needed.
# IMPORTANT: This pipeline is computationally expensive; begin with random
# split only if necessary, and reenable chronological holdout for the report.
CONFIG$RUN_RANDOM_SPLIT <- TRUE
CONFIG$RUN_CHRONOLOGICAL_SPLIT <- TRUE
CONFIG$REMOVE_EXACT_DUPLICATES <- TRUE  # protect random split against identical-row overlap

fraud <- load_local_fraud()
# 'Amount' is log1p-transformed by load_local_fraud(), so EDA needs
# ORIGINAL amount: invert it for descriptive plotting and descriptive stats.
fraud_eda <- fraud
fraud_eda$Amount <- expm1(fraud_eda$Amount)
plot_fraud_eda(fraud_eda)
rm(fraud_eda); gc(verbose = FALSE)

# Exact duplicate rows can otherwise occur on both sides of a random split.
# Deduplication is performed on the ORIGINAL observations, BEFORE the split;
# original raw data totals remain logged and EDA reflects the full source.
if (CONFIG$REMOVE_EXACT_DUPLICATES) {
  n_raw <- nrow(fraud)
  fraud <- fraud[!duplicated(fraud), , drop = FALSE]
  rownames(fraud) <- NULL
  log_message("Removed ", n_raw - nrow(fraud), " exact duplicate rows before partitioning; retained ",
              nrow(fraud), " transactions and ", sum(fraud$Class == "1"), " frauds.")
}

# The cleaned model-ready data are saved separately from ORIGINAL data.
# Amount has a prespecified log1p transformation; saved original CSV is unchanged.
if (isTRUE(CONFIG$SAVE_PROCESSED_DATA)) {
  write_compressed_data(fraud, "creditcard_model_ready.csv.gz",
                        CONFIG$PROCESSED_DATA_DIR)
}
# Temporary row ID for split reproducibility. Remove it before model fitting!
fraud$.source_row <- seq_len(nrow(fraud))

save_split_and_remove_id <- function(sp, experiment_name) {
  for (part in c("train", "test")) {
    ids <- sp[[part]]$.source_row
    write_compressed_data(data.frame(Processed_Row_ID = ids),
                          paste0(experiment_name, "__", part, "_rows.csv.gz"),
                          CONFIG$SPLIT_DIR)
    sp[[part]]$.source_row <- NULL
  }
  sp
}

results <- list(); ensembles <- list(); split_meta <- list()

if (CONFIG$RUN_RANDOM_SPLIT) {
  sp <- stratified_split(fraud, CONFIG$TEST_FRACTION, CONFIG$SEED + 300L)
  sp <- save_split_and_remove_id(sp, "fraud_stratified_random")
  split_meta[[length(split_meta) + 1L]] <- data.frame(
    Experiment = "fraud_stratified_random", Train_N = nrow(sp$train),
    Test_N = nrow(sp$test), Train_Positives = sum(sp$train$Class == "1"),
    Test_Positives = sum(sp$test$Class == "1"),
    Train_Prevalence = mean(sp$train$Class == "1"),
    Test_Prevalence = mean(sp$test$Class == "1"),
    Chronological_Boundary = NA_real_)
  res <- experiment(sp$train, sp$test, "fraud_stratified_random",
                    seed = CONFIG$SEED + 301L, is_real = TRUE,
                    target_prevalence = 492 / 284807)
  results[[length(results) + 1L]] <- res$scores
  ensembles[[length(ensembles) + 1L]] <- res$ensemble
}
if (CONFIG$RUN_CHRONOLOGICAL_SPLIT) {
  sp <- chronological_split(fraud, CONFIG$TEST_FRACTION)
  sp <- save_split_and_remove_id(sp, "fraud_chronological_holdout")
  split_meta[[length(split_meta) + 1L]] <- data.frame(
    Experiment = "fraud_chronological_holdout", Train_N = nrow(sp$train),
    Test_N = nrow(sp$test), Train_Positives = sum(sp$train$Class == "1"),
    Test_Positives = sum(sp$test$Class == "1"),
    Train_Prevalence = mean(sp$train$Class == "1"),
    Test_Prevalence = mean(sp$test$Class == "1"),
    Chronological_Boundary = max(sp$train$Time))
  # Do not pretend that random 5-fold training CV is forward-chaining CV.
  # The final test IS strictly later than training; inner training CV is
  # stratified random and should be described honestly in the report.
  res <- experiment(sp$train, sp$test, "fraud_chronological_holdout",
                    seed = CONFIG$SEED + 401L, is_real = TRUE,
                    target_prevalence = 492 / 284807)
  results[[length(results) + 1L]] <- res$scores
  ensembles[[length(ensembles) + 1L]] <- res$ensemble
}
if (!length(results)) stop("Enable at least one fraud split")
write_csv(do.call(rbind, split_meta), "fraud__split_summary.csv")
resall <- do.call(rbind, results)
write_csv(resall, "fraud__all_metrics.csv")
write_csv(do.call(rbind, ensembles), "fraud__ensemble_comparisons.csv")
selected <- do.call(rbind, lapply(split(resall, resall$Experiment), function(x) {
  # Select on TRAINING OOF AP, not test AP. Rule for operational cutoff is
  # separately tuned by OOF F1 only (test F1 never selects).
  x <- x[x$Classifier %in% c(CLASSIFIERS, "Ensemble") &
         x$Operating_Rule == "CV_tuned_F1", , drop = FALSE]
  x[order(-x$CV_PR_AUC_AP, -x$CV_F1_Selected), , drop = FALSE][1, , drop = FALSE]
}))
write_csv(selected, "fraud__CV_selected_models.csv")

# ===================== FINAL FRAUD RESULTS VISUALS ============================
# Figures reflect only UNRESAMPLED holdouts. Selection is by training OOF scores.
save_final_fraud_plot <- function(p, name, width = 10, height = 6) {
  ggplot2::ggsave(file.path(CONFIG$PLOT_FINAL_DIR, name), p,
                  width = width, height = height, dpi = 300, bg = "white")
}
fraud_theme <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", size = 15),
                   panel.grid.minor = ggplot2::element_blank(),
                   legend.position = "bottom",
                   strip.text = ggplot2::element_text(face = "bold"))
}
plot_final_fraud_results <- function(result_rows, chosen) {
  fixed <- result_rows[result_rows$Operating_Rule == "default_0.5" &
             result_rows$Classifier %in% CLASSIFIERS, , drop = FALSE]
  fixed$Remedy <- factor(fixed$Remedy, levels = REMEDIES)
  p1 <- ggplot2::ggplot(fixed,
          ggplot2::aes(x = Remedy, y = PR_AUC_AP, fill = Classifier)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.7), width = 0.65) +
    ggplot2::facet_wrap(~ Experiment, scales = "free_y") +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "Fraud prediction: held-out average precision",
                  subtitle = "Ranking score on natural-prevalence test sets; threshold-independent",
                  x = "Training remedy", y = "Average precision (AP)", fill = "Classifier") +
    fraud_theme()
  save_final_fraud_plot(p1, "01_Fraud_AP_Model_Comparison.png", 12, 6.5)

  p2 <- ggplot2::ggplot(fixed,
          ggplot2::aes(x = Remedy, y = ROC_AUC, fill = Classifier)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.7), width = 0.65) +
    ggplot2::facet_wrap(~ Experiment) + ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(title = "Held-out ROC-AUC by classifier and remedy",
                  x = "Training remedy", y = "ROC-AUC", fill = "Classifier") + fraud_theme()
  save_final_fraud_plot(p2, "02_Fraud_ROC_AUC_Model_Comparison.png", 12, 6.5)

  th <- result_rows[result_rows$Classifier %in% CLASSIFIERS &
         result_rows$Remedy == "baseline" &
         result_rows$Operating_Rule %in% c("default_0.5", "CV_tuned_F1"), , drop = FALSE]
  p3 <- ggplot2::ggplot(th,
          ggplot2::aes(x = Classifier, y = F1, fill = Operating_Rule)) +
    ggplot2::geom_col(position = "dodge", width = 0.65) +
    ggplot2::facet_wrap(~ Experiment) +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::labs(title = "Unmodified model: default versus CV-tuned threshold",
                  subtitle = "Thresholds learned only from out-of-fold training predictions",
                  x = "Classifier", y = "Held-out F1", fill = "Decision rule") + fraud_theme()
  save_final_fraud_plot(p3, "03_Fraud_Baseline_Threshold_Effect_on_F1.png", 12, 6)

  po <- result_rows[result_rows$Classifier %in% CLASSIFIERS &
       result_rows$Operating_Rule == "CV_tuned_F1", , drop = FALSE]
  po$Remedy <- factor(po$Remedy, levels = REMEDIES)
  p4 <- ggplot2::ggplot(po,
          ggplot2::aes(x = Sensitivity, y = Precision,
                       colour = Remedy, shape = Classifier)) +
    ggplot2::geom_point(size = 3.5, alpha = 0.9) +
    ggplot2::facet_wrap(~ Experiment) +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    ggplot2::labs(title = "Fraud investigation trade-off: recall versus precision",
                  subtitle = "Each point uses its independently CV-tuned F1 threshold",
                  x = "Sensitivity / recall", y = "Precision") + fraud_theme()
  save_final_fraud_plot(p4, "04_Fraud_Precision_Recall_Tradeoff.png", 12, 6)

  p5 <- ggplot2::ggplot(fixed,
          ggplot2::aes(x = Remedy, y = Brier, fill = Classifier)) +
    ggplot2::geom_col(position = "dodge", width = 0.65) +
    ggplot2::facet_wrap(~ Experiment, scales = "free_y") +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "Brier score: quality of predicted probabilities",
                  subtitle = "Lower is better; resampling/weights can hurt calibration",
                  x = "Training remedy", y = "Mean squared probability error", fill = "Classifier") +
    fraud_theme()
  save_final_fraud_plot(p5, "05_Fraud_Brier_Score.png", 12, 6.5)

  # Confusion matrices for the models selected using TRAINING CV only.
  cc <- do.call(rbind, lapply(seq_len(nrow(chosen)), function(i) {
    r <- chosen[i, , drop = FALSE]
    data.frame(Experiment = r$Experiment,
      Actual = c("Fraud", "Legitimate", "Fraud", "Legitimate"),
      Predicted = c("Fraud", "Legitimate", "Legitimate", "Fraud"),
      Count = c(r$TP, r$TN, r$FN, r$FP),
      Selected_Model = paste(r$Classifier, r$Remedy, sep = " / "))
  }))
  write_csv(cc, "fraud__selected_confusion_matrix_cells.csv")
  p6 <- ggplot2::ggplot(cc,
          ggplot2::aes(x = Predicted, y = Actual, fill = Count)) +
    ggplot2::geom_tile(colour = "white", linewidth = 1) +
    ggplot2::geom_text(ggplot2::aes(label = format(Count, big.mark = ",", trim = TRUE)),
                       size = 4, colour = "black") +
    ggplot2::facet_wrap(~ Experiment) +
    ggplot2::scale_fill_gradient(low = "#E7F5F6", high = "#11859B") +
    ggplot2::labs(title = "Confusion matrices: training-CV-selected models",
                  subtitle = "Held-out test counts, not training performance",
                  x = "Predicted class", y = "True class", fill = "Count") + fraud_theme() +
    ggplot2::theme(legend.position = "none")
  save_final_fraud_plot(p6, "06_Fraud_CV_Selected_Confusion_Matrices.png", 11, 5.5)

  # Ensemble weights were selected on training OOF AP, NOT test outcomes.
  weight_rows <- lapply(unique(result_rows$Experiment), function(exp_name) {
    file <- file.path(CONFIG$TABLE_DIR, paste0(exp_name, "__ensemble_weight_search.csv"))
    if (file.exists(file)) data.table::fread(file) else NULL
  })
  weight_rows <- Filter(Negate(is.null), weight_rows)
  if (length(weight_rows)) {
    ww <- as.data.frame(data.table::rbindlist(weight_rows, fill = TRUE))
    p7 <- ggplot2::ggplot(ww,
            ggplot2::aes(x = Logistic_Weight, y = CV_AP)) +
      ggplot2::geom_line(linewidth = 0.9, colour = "#147D92") +
      ggplot2::geom_point(size = 2.0, colour = "#147D92") +
      ggplot2::geom_vline(ggplot2::aes(xintercept = Selected_Weight),
                           linetype = "dashed", colour = "#C96A2B") +
      ggplot2::facet_wrap(~ Experiment, scales = "free_y") +
      ggplot2::labs(title = "STAT 847 ensemble: cross-validated mixing ratio",
                    subtitle = "Vertical dashed line: weight chosen using training OOF average precision",
                    x = "Logistic share (random forest share = 1 − weight)",
                    y = "Out-of-fold average precision") + fraud_theme()
    save_final_fraud_plot(p7, "07_Fraud_Ensemble_Weight_Search.png", 12, 6)
  }
}
plot_final_fraud_results(resall, selected)

writeLines(capture.output(str(CONFIG)), file.path(CONFIG$METADATA_DIR, "fraud__configuration.txt"))
writeLines(capture.output(sessionInfo()), file.path(CONFIG$METADATA_DIR, "fraud__sessionInfo.txt"))
log_message("Fraud study finished. Tables: ", CONFIG$TABLE_DIR,
            " | Final figures: ", CONFIG$PLOT_FINAL_DIR,
            " | Local input: ", CONFIG$LOCAL_CSV,
            " | Saved original data: ", CONFIG$DATA_DIR,
            " | Processed data: ", CONFIG$PROCESSED_DATA_DIR)

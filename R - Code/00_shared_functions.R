# STAT 847 Team 2 - shared model and evaluation functions
# These functions are used by both the simulated and real fraud experiments.
# Run the numbered analysis scripts from the RStudio project directory.

make_settings <- function() {
  fast <- identical(Sys.getenv("STAT847_FAST_CHECK", "0"), "1")
  list(
    seed = 847L,
    test_fraction = 0.20,
    cv_folds = 5L,
    inner_folds = 3L,
    lambdas = c(0.005, 0.02, 0.08, 0.32),
    trees = if (fast) 30L else 150L,
    bootstrap_runs = if (fast) 25L else 200L,
    repeats = if (fast) 1L else 3L,
    sizes = if (fast) 2000L else c(2000L, 20000L),
    rates = c(0.50, 0.20, 0.05, 0.01),
    large_rare_scenario = !fast,
    remedies = c("baseline", "oversample", "undersample", "smote", "weighted"),
    models = c("Penalized_Logistic", "Random_Forest"),
    over_ratio = 0.25,
    majority_to_minority = 4L,
    ensemble_weights = seq(0.1, 0.9, by = 0.1),
    threads = 1L,
    fast_check = fast
  )
}

check_packages <- function() {
  needed <- c("data.table", "ggplot2", "glmnet", "ranger", "themis")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Install the missing R packages and rerun: install.packages(c(",
         paste(sprintf('"%s"', missing), collapse = ", "), "))")
  }
  invisible(TRUE)
}

make_folders <- function(base, parts) {
  for (p in parts) dir.create(file.path(base, p), recursive = TRUE, showWarnings = FALSE)
  invisible(base)
}

save_table <- function(df, folder, name) {
  data.table::fwrite(df, file.path(folder, "tables", name))
}

save_plot <- function(plot, folder, section, name, width = 10, height = 6, dpi = 200) {
  ggplot2::ggsave(file.path(folder, "plots", section, name), plot,
                  width = width, height = height, dpi = dpi, bg = "white")
}

# AP is non-interpolated average precision (not trapezoidal PR-AUC).
# Tied probabilities are handled as groups, so their order does not matter.
get_ap <- function(y, prob) {
  y <- as.integer(as.character(y))
  if (sum(y == 1L) == 0L) return(NA_real_)
  ind <- order(prob, decreasing = TRUE)
  yy <- y[ind]
  pp <- prob[ind]
  last <- which(!duplicated(pp, fromLast = TRUE))
  tp <- cumsum(yy)[last]
  recall <- tp / sum(yy == 1L)
  precision <- tp / last
  sum(diff(c(0, recall)) * precision)
}

get_auc <- function(y, prob) {
  y <- as.integer(as.character(y))
  n1 <- sum(y == 1L); n0 <- sum(y == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  ranks <- rank(prob, ties.method = "average")
  (sum(ranks[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

clip_prob <- function(p) pmin(1 - 1e-12, pmax(1e-12, as.numeric(p)))

get_scores <- function(y, prob, cutoff = 0.5) {
  y <- as.integer(as.character(y))
  prob <- clip_prob(prob)
  prediction <- as.integer(prob >= cutoff)
  tp <- sum(y == 1L & prediction == 1L)
  tn <- sum(y == 0L & prediction == 0L)
  fp <- sum(y == 0L & prediction == 1L)
  fn <- sum(y == 1L & prediction == 0L)
  sensitivity <- if (tp + fn == 0L) NA_real_ else tp / (tp + fn)
  specificity <- if (tn + fp == 0L) NA_real_ else tn / (tn + fp)
  precision <- if (tp + fp == 0L) 0 else tp / (tp + fp)
  f1 <- if (2 * tp + fp + fn == 0L) 0 else 2 * tp / (2 * tp + fp + fn)
  data.frame(
    N = length(y), Positives = sum(y == 1L), Prevalence = mean(y == 1L),
    Threshold = cutoff, Accuracy = mean(y == prediction),
    Sensitivity = sensitivity, Specificity = specificity, Precision = precision,
    F1 = f1, Balanced_Accuracy = mean(c(sensitivity, specificity)),
    ROC_AUC = get_auc(y, prob), PR_AUC_AP = get_ap(y, prob),
    Brier = mean((prob - y)^2),
    Log_Loss = -mean(y * log(prob) + (1 - y) * log(1 - prob)),
    TP = tp, FP = fp, TN = tn, FN = fn, Flagged_Count = tp + fp
  )
}

# Select F1/F2 thresholds from training out-of-fold scores, never test labels.
choose_cutoff <- function(y, prob, beta = 1) {
  y <- as.integer(as.character(y))
  prob <- clip_prob(prob)
  if (sum(y == 1L) == 0L) return(0.5)
  ind <- order(prob, decreasing = TRUE)
  yy <- y[ind]; pp <- prob[ind]
  last <- which(!duplicated(pp, fromLast = TRUE))
  tp <- cumsum(yy)[last]
  fp <- last - tp
  fn <- sum(yy) - tp
  b2 <- beta^2
  scores <- (1 + b2) * tp / pmax(1, (1 + b2) * tp + b2 * fn + fp)
  pp[last[which.max(scores)]]
}

make_folds <- function(y, k, s) {
  if (min(table(y)) < k) stop("Not enough cases of one class for ", k, " folds")
  set.seed(s)
  fold_id <- integer(length(y))
  for (class_name in unique(as.character(y))) {
    rows <- sample(which(as.character(y) == class_name))
    fold_id[rows] <- rep(seq_len(k), length.out = length(rows))
  }
  fold_id
}

split_random <- function(data, s, test_fraction = 0.20) {
  set.seed(s)
  is_test <- rep(FALSE, nrow(data))
  for (cl in levels(data$Class)) {
    rows <- which(data$Class == cl)
    n_test <- max(1L, round(length(rows) * test_fraction))
    is_test[sample(rows, n_test)] <- TRUE
  }
  list(train = data[!is_test, , drop = FALSE],
       test = data[is_test, , drop = FALSE])
}

# All means and SDs come from the current training partition.
scale_first <- function(train) {
  cols <- setdiff(names(train), "Class")
  mu <- vapply(train[cols], mean, numeric(1))
  sd_values <- vapply(train[cols], sd, numeric(1))
  sd_values[!is.finite(sd_values) | sd_values < 1e-10] <- 1
  out <- train
  for (v in cols) out[[v]] <- (out[[v]] - mu[v]) / sd_values[v]
  list(data = out, average = mu, sd = sd_values, cols = cols)
}

scale_other <- function(data, info) {
  out <- data
  for (v in info$cols) out[[v]] <- (out[[v]] - info$average[v]) / info$sd[v]
  out
}

# Alter training samples only. Validation and test sets retain natural prevalence.
change_sample <- function(data, method, s, cfg) {
  if (method %in% c("baseline", "weighted")) return(data)
  set.seed(s)
  yes <- which(data$Class == "1")
  no <- which(data$Class == "0")
  if (length(yes) < 2L || length(no) < 2L) {
    stop("Not enough observations of both classes for ", method)
  }
  if (method == "oversample") {
    want <- max(length(yes), ceiling(cfg$over_ratio * length(no)))
    extra <- if (want > length(yes)) sample(yes, want - length(yes), replace = TRUE) else integer()
    ans <- data[c(seq_len(nrow(data)), extra), , drop = FALSE]
  } else if (method == "undersample") {
    keep <- min(length(no), cfg$majority_to_minority * length(yes))
    ans <- data[c(yes, sample(no, keep)), , drop = FALSE]
  } else if (method == "smote") {
    if (length(yes) / length(no) >= cfg$over_ratio) return(data)
    ans <- as.data.frame(themis::smote(data, var = "Class",
                        k = min(5L, length(yes) - 1L), over_ratio = cfg$over_ratio))
  } else {
    stop("Unknown sampling method: ", method)
  }
  ans$Class <- factor(as.character(ans$Class), levels = c("0", "1"))
  ans <- ans[sample.int(nrow(ans)), , drop = FALSE]
  rownames(ans) <- NULL
  ans
}

get_weights <- function(train, original, method) {
  if (method != "weighted") return(rep(1, nrow(train)))
  w <- sqrt(sum(original$Class == "0") / sum(original$Class == "1"))
  ifelse(train$Class == "1", w, 1)
}

# For penalized logistic regression, lambda is tuned using INNER training CV.
# Each inner split independently learns scaling and any sampling treatment.
fit_one <- function(train, model_name, method, s, cfg) {
  if (!identical(levels(train$Class), c("0", "1"))) stop("Class levels must be 0, 1")
  cols <- setdiff(names(train), "Class")
  if (anyNA(train) || !all(vapply(train[cols], is.numeric, logical(1)))) {
    stop("Training predictors must be numeric and nonmissing")
  }
  if (model_name == "Penalized_Logistic") {
    k <- min(cfg$inner_folds, as.integer(min(table(train$Class))))
    if (k < 2L) stop("Not enough minority observations for inner validation")
    fold_id <- make_folds(train$Class, k, s + 600L)
    errors <- matrix(NA_real_, nrow = k, ncol = length(cfg$lambdas))
    for (j in seq_len(k)) {
      tr <- train[fold_id != j, , drop = FALSE]
      va <- train[fold_id == j, , drop = FALSE]
      sc <- scale_first(tr)
      tr2 <- change_sample(sc$data, method, s + j, cfg)
      model <- glmnet::glmnet(
        as.matrix(tr2[sc$cols]), as.integer(as.character(tr2$Class)),
        family = "binomial", alpha = 0, lambda = cfg$lambdas,
        weights = get_weights(tr2, tr, method),
        standardize = FALSE, maxit = 100000L)
      va2 <- scale_other(va, sc)
      p <- as.matrix(predict(model, newx = as.matrix(va2[sc$cols]),
                             s = cfg$lambdas, type = "response"))
      yy <- as.integer(as.character(va$Class))
      for (h in seq_along(cfg$lambdas)) {
        pp <- clip_prob(p[, h])
        errors[j, h] <- -mean(yy * log(pp) + (1 - yy) * log(1 - pp))
      }
    }
    picked <- cfg$lambdas[which.min(colMeans(errors))]
    sc <- scale_first(train)
    train2 <- change_sample(sc$data, method, s + 100L, cfg)
    model <- glmnet::glmnet(
      as.matrix(train2[sc$cols]), as.integer(as.character(train2$Class)),
      family = "binomial", alpha = 0, lambda = cfg$lambdas,
      weights = get_weights(train2, train, method),
      standardize = FALSE, maxit = 100000L)
    return(list(kind = model_name, fit = model, scale = sc, lambda = picked))
  }
  if (model_name == "Random_Forest") {
    sc <- scale_first(train)
    train2 <- change_sample(sc$data, method, s + 100L, cfg)
    class_weights <- NULL
    if (method == "weighted") {
      w <- sqrt(sum(train$Class == "0") / sum(train$Class == "1"))
      class_weights <- c("0" = 1, "1" = w)
    }
    model <- ranger::ranger(
      Class ~ ., data = train2, probability = TRUE, num.trees = cfg$trees,
      mtry = max(1L, floor(sqrt(length(sc$cols)))),
      min.node.size = 10L, class.weights = class_weights,
      num.threads = cfg$threads, seed = s, write.forest = TRUE)
    return(list(kind = model_name, fit = model, scale = sc))
  }
  stop("Unknown model: ", model_name)
}

get_predictions <- function(model, newdata, cfg) {
  x <- scale_other(newdata, model$scale)
  if (model$kind == "Penalized_Logistic") {
    return(clip_prob(predict(model$fit, newx = as.matrix(x[model$scale$cols]),
                             s = model$lambda, type = "response")))
  }
  clip_prob(predict(model$fit, data = x, num.threads = cfg$threads)$predictions[, "1"])
}

# Curve points are thinned for plotting only; AP/AUC always use all observations.
curve_points <- function(y, prob, type, max_points = 1200L) {
  y <- as.integer(as.character(y))
  ind <- order(prob, decreasing = TRUE)
  y <- y[ind]; prob <- prob[ind]
  last <- which(!duplicated(prob, fromLast = TRUE))
  step <- max(1L, floor(length(last) / max_points))
  last <- unique(c(last[seq(1L, length(last), by = step)], tail(last, 1L)))
  tp <- cumsum(y)[last]
  fp <- last - tp
  if (type == "PR") {
    data.frame(x = c(0, tp / sum(y)), y = c(1, tp / last))
  } else {
    data.frame(x = c(0, fp / sum(y == 0L)), y = c(0, tp / sum(y == 1L)))
  }
}

# Fit and compare models for one train/test split. All model, threshold, and
# ensemble choices are made on training OOF predictions, not test outcomes.
compare_models <- function(train, test, experiment, out_folder, s, cfg,
                           target = NA_real_, real_data = FALSE, true_prob = NULL) {
  cat("\nStarting:", experiment, "\n")
  f <- make_folds(train$Class, cfg$cv_folds, s)
  methods <- expand.grid(Model = cfg$models, Method = cfg$remedies,
                         stringsAsFactors = FALSE)
  labels <- paste(methods$Model, methods$Method, sep = "__")
  p_cv <- matrix(NA_real_, nrow(train), length(labels), dimnames = list(NULL, labels))
  p_test <- matrix(NA_real_, nrow(test), length(labels), dimnames = list(NULL, labels))

  for (i in seq_along(labels)) {
    cat("  Fitting", labels[i], "\n")
    for (j in seq_len(cfg$cv_folds)) {
      model <- fit_one(train[f != j, , drop = FALSE],
                       methods$Model[i], methods$Method[i], s + i * 100L + j, cfg)
      p_cv[f == j, i] <- get_predictions(model, train[f == j, , drop = FALSE], cfg)
    }
    model <- fit_one(train, methods$Model[i], methods$Method[i],
                     s + 20000L + i, cfg)
    p_test[, i] <- get_predictions(model, test, cfg)
    rm(model)
  }
  if (anyNA(p_cv) || anyNA(p_test)) stop("Some predictions were not produced")

  rows <- list()
  add_result <- function(model, method, rule, cutoff, cv_prob, test_prob) {
    cv <- get_scores(train$Class, cv_prob, cutoff)
    te <- get_scores(test$Class, test_prob, cutoff)
    rows[[length(rows) + 1L]] <<- data.frame(
      Experiment = experiment, Target_Prevalence = target,
      Train_N = nrow(train), Test_N = nrow(test),
      Train_Positive_Count = sum(train$Class == "1"),
      Test_Positive_Count = sum(test$Class == "1"),
      Classifier = model, Remedy = method, Operating_Rule = rule,
      Threshold = cutoff, CV_ROC_AUC = cv$ROC_AUC,
      CV_PR_AUC_AP = cv$PR_AUC_AP, CV_F1_Selected = cv$F1,
      te[, setdiff(names(te), "Threshold"), drop = FALSE]
    )
  }

  add_result("Majority_Class", "predict_all_negative", "all_negative", 0.5,
             rep(0, nrow(train)), rep(0, nrow(test)))
  for (i in seq_along(labels)) {
    add_result(methods$Model[i], methods$Method[i], "default_0.5", 0.5,
               p_cv[, i], p_test[, i])
    th <- choose_cutoff(train$Class, p_cv[, i], beta = 1)
    add_result(methods$Model[i], methods$Method[i], "CV_tuned_F1", th,
               p_cv[, i], p_test[, i])
    if (methods$Method[i] == "baseline") {
      th2 <- choose_cutoff(train$Class, p_cv[, i], beta = 2)
      add_result(methods$Model[i], methods$Method[i], "CV_tuned_F2", th2,
                 p_cv[, i], p_test[, i])
    }
  }

  # STAT 847 ensemble: choose both component remedies and mixing ratio by OOF AP.
  ap_cv <- vapply(seq_along(labels), function(i) get_ap(train$Class, p_cv[, i]), numeric(1))
  a_set <- which(methods$Model == "Penalized_Logistic")
  b_set <- which(methods$Model == "Random_Forest")
  a <- a_set[which.max(ap_cv[a_set])]
  b <- b_set[which.max(ap_cv[b_set])]
  grid_ap <- vapply(cfg$ensemble_weights, function(w) {
    get_ap(train$Class, w * p_cv[, a] + (1 - w) * p_cv[, b])
  }, numeric(1))
  w_best <- cfg$ensemble_weights[which.max(grid_ap)]
  blend_cv <- w_best * p_cv[, a] + (1 - w_best) * p_cv[, b]
  blend_test <- w_best * p_test[, a] + (1 - w_best) * p_test[, b]
  add_result("Ensemble", "CV_optimized_blend", "default_0.5", 0.5,
             blend_cv, blend_test)
  th <- choose_cutoff(train$Class, blend_cv, beta = 1)
  add_result("Ensemble", "CV_optimized_blend", "CV_tuned_F1", th,
             blend_cv, blend_test)

  if (!is.null(true_prob)) {
    # Oracle is a reference in simulations, not an eligible learner.
    add_result("Oracle_True_P", "simulation_truth", "default_0.5", 0.5,
               rep(mean(train$Class == "1"), nrow(train)), true_prob)
  }
  result <- do.call(rbind, rows)
  save_table(result, out_folder, paste0(experiment, "__test_metrics.csv"))
  weight_search <- data.frame(
    Experiment = experiment, Logistic_Weight = cfg$ensemble_weights,
    RF_Weight = 1 - cfg$ensemble_weights, CV_AP = grid_ap,
    Selected_Weight = w_best, Logistic_Model = labels[a], RF_Model = labels[b]
  )
  save_table(weight_search, out_folder, paste0(experiment, "__ensemble_weight_search.csv"))

  # Paired class-stratified test bootstrap, after all selections are frozen.
  best_i <- which.max(ap_cv)
  test_y <- as.integer(as.character(test$Class))
  neg <- which(test_y == 0L); pos <- which(test_y == 1L)
  set.seed(s + 90000L)
  ap_diffs <- replicate(cfg$bootstrap_runs, {
    ind <- c(sample(neg, length(neg), replace = TRUE),
             sample(pos, length(pos), replace = TRUE))
    get_ap(test_y[ind], blend_test[ind]) - get_ap(test_y[ind], p_test[ind, best_i])
  })
  boot <- data.frame(
    Experiment = experiment, Best_Single_CV_Selected = labels[best_i],
    Selected_Logistic = labels[a], Selected_Forest = labels[b],
    Optimal_Logistic_Weight = w_best,
    Ensemble_minus_Single_AP = get_ap(test_y, blend_test) - get_ap(test_y, p_test[, best_i]),
    Bootstrap_CI_Lower = unname(quantile(ap_diffs, .025)),
    Bootstrap_CI_Upper = unname(quantile(ap_diffs, .975)),
    Bootstrap_Pr_Difference_Positive = mean(ap_diffs > 0),
    Bootstrap_Replicates = cfg$bootstrap_runs
  )
  save_table(boot, out_folder, paste0(experiment, "__ensemble_vs_single_bootstrap.csv"))

  if (real_data) {
    predictions <- data.frame(
      Truth = test_y,
      Logistic_Baseline = p_test[, "Penalized_Logistic__baseline"],
      RF_Baseline = p_test[, "Random_Forest__baseline"],
      Best_Single = p_test[, best_i], Ensemble = blend_test
    )
    save_table(predictions, out_folder, paste0(experiment, "__heldout_predictions.csv"))
    for (type in c("PR", "ROC")) {
      points <- lapply(names(predictions)[-1L], function(name) {
        d <- curve_points(predictions$Truth, predictions[[name]], type)
        d$Model <- name
        d
      })
      plot_data <- do.call(rbind, points)
      save_table(plot_data, out_folder, paste0(experiment, "__", type, "_curve_points.csv"))
      p <- ggplot2::ggplot(plot_data, ggplot2::aes(x, y, colour = Model)) +
        ggplot2::geom_line(linewidth = .8) +
        ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::labs(title = paste(experiment, type, "curve (held-out test)"),
                      x = if (type == "PR") "Recall" else "False-positive rate",
                      y = if (type == "PR") "Precision" else "True-positive rate")
      save_plot(p, out_folder, "curves", paste0(experiment, "__", type, ".png"), 9, 5.5)
    }
  }
  list(scores = result, bootstrap = boot, weight_search = weight_search)
}

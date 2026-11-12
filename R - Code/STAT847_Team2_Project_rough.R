set.seed(847)
options(warn = 1)

# packages I have been using
packages <- c("data.table", "ggplot2", "glmnet", "ranger", "themis")
for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}

seed <- 847
folds <- 5
inner_folds <- 3
lambdas <- c(0.005, 0.02, 0.08, 0.32)
trees <- 150
bootstrap_runs <- 200
repeats <- 3     
sizes <- c(2000, 20000)
rates <- c(0.50, 0.20, 0.05, 0.01)
remedies <- c("baseline", "oversample", "undersample", "smote", "weighted")
models <- c("Penalized_Logistic", "Random_Forest")

# Kaggle Data
csv_file <- "C:/Users/HP/Downloads/creditcard.csv"

root <- normalizePath(getwd(), winslash = "/")
sim_folder <- file.path(root, "results", "simulation")
fraud_folder <- file.path(sim_folder, "fraud_real_data")
compare_folder <- file.path(sim_folder, "combined_results")

# Setting up the output
for (folder in c(
    file.path(sim_folder, "tables"), file.path(sim_folder, "data", "generated"),
    file.path(sim_folder, "data", "design"), file.path(sim_folder, "plots", "eda"),
    file.path(sim_folder, "plots", "final"), file.path(sim_folder, "metadata"),
    file.path(fraud_folder, "tables"), file.path(fraud_folder, "data", "raw"),
    file.path(fraud_folder, "data", "processed"), file.path(fraud_folder, "data", "splits"),
    file.path(fraud_folder, "plots", "eda"), file.path(fraud_folder, "plots", "curves"),
    file.path(fraud_folder, "plots", "final"), file.path(fraud_folder, "metadata"),
    compare_folder)) {
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
}

# I use AP for the precision-recall summary, not trapezoid PR-AUC
get_ap <- function(y, prob) {
  y <- as.integer(as.character(y))
  if (!sum(y == 1)) return(NA_real_)
  ind <- order(prob, decreasing = TRUE)
  y <- y[ind]
  prob <- prob[ind]
  at <- which(!duplicated(prob, fromLast = TRUE))
  tp <- cumsum(y)[at]
  recall <- tp / sum(y)
  precision <- tp / at
  sum(diff(c(0, recall)) * precision)
}

get_auc <- function(y, prob) {
  y <- as.integer(as.character(y))
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  ranks <- rank(prob, ties.method = "average")
  (sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

get_scores <- function(y, prob, cutoff = 0.5) {
  y <- as.integer(as.character(y))
  prob <- pmin(1 - 1e-12, pmax(1e-12, as.numeric(prob)))
  prediction <- as.integer(prob >= cutoff)
  tp <- sum(y == 1 & prediction == 1)
  tn <- sum(y == 0 & prediction == 0)
  fp <- sum(y == 0 & prediction == 1)
  fn <- sum(y == 1 & prediction == 0)
  sensitivity <- if (tp + fn == 0) NA_real_ else tp / (tp + fn)
  specificity <- if (tn + fp == 0) NA_real_ else tn / (tn + fp)
  precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
  f1 <- if (2 * tp + fp + fn == 0) 0 else 2 * tp / (2 * tp + fp + fn)
  data.frame(
    N = length(y), Positives = sum(y), Prevalence = mean(y), Threshold = cutoff,
    Accuracy = mean(y == prediction), Sensitivity = sensitivity,
    Specificity = specificity, Precision = precision, F1 = f1,
    Balanced_Accuracy = mean(c(sensitivity, specificity)),
    ROC_AUC = get_auc(y, prob), PR_AUC_AP = get_ap(y, prob),
    Brier = mean((prob - y)^2),
    Log_Loss = -mean(y * log(prob) + (1 - y) * log(1 - prob)),
    TP = tp, FP = fp, TN = tn, FN = fn, Flagged_Count = tp + fp
  )
}

# selecting the F1 or F2 threshold on training predictions only
choose_cutoff <- function(y, prob, beta = 1) {
  y <- as.integer(as.character(y))
  ind <- order(prob, decreasing = TRUE)
  y <- y[ind]; prob <- prob[ind]
  at <- which(!duplicated(prob, fromLast = TRUE))
  tp <- cumsum(y)[at]; fp <- at - tp; fn <- sum(y) - tp
  b2 <- beta^2
  values <- (1 + b2) * tp / pmax(1, (1 + b2) * tp + b2 * fn + fp)
  prob[at[which.max(values)]]
}

# stratifying both the train/test split and the CV folds
make_folds <- function(y, k, s) {
  set.seed(s)
  if (min(table(y)) < k) stop("Not enough minority observations for this many folds")
  fold_id <- integer(length(y))
  for (class_name in unique(as.character(y))) {
    rows <- sample(which(as.character(y) == class_name))
    fold_id[rows] <- rep(seq_len(k), length.out = length(rows))
  }
  fold_id
}

split_random <- function(data, s) {
  set.seed(s)
  test_row <- rep(FALSE, nrow(data))
  for (cl in c("0", "1")) {
    rows <- which(data$Class == cl)
    test_row[sample(rows, max(1, round(length(rows) * 0.2)))] <- TRUE
  }
  list(train = data[!test_row, , drop = FALSE], test = data[test_row, , drop = FALSE])
}

# scaling must be learned from the current training part, not the test data
scale_first <- function(train) {
  cols <- setdiff(names(train), "Class")
  mu <- sapply(train[cols], mean)
  sd_values <- sapply(train[cols], sd)
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

# just changing the training sample (never CV validation or test rows)
change_sample <- function(data, way, s) {
  if (way %in% c("baseline", "weighted")) return(data)
  set.seed(s)
  yes <- which(data$Class == "1")
  no <- which(data$Class == "0")
  if (length(yes) < 2 || length(no) < 2) stop("Not enough samples in both classes")
  if (way == "oversample") {
    want <- max(length(yes), ceiling(0.25 * length(no)))
    extra <- if (want > length(yes)) sample(yes, want - length(yes), replace = TRUE) else integer()
    ans <- data[c(seq_len(nrow(data)), extra), , drop = FALSE]
  } else if (way == "undersample") {
    keep <- min(length(no), 4 * length(yes))
    ans <- data[c(yes, sample(no, keep)), , drop = FALSE]
  } else if (way == "smote") {
    if (length(yes) / length(no) >= 0.25) return(data)
    ans <- as.data.frame(themis::smote(data, var = "Class",
                                      k = min(5, length(yes) - 1), over_ratio = 0.25))
  } else {
    stop("Unknown sampling method")
  }
  ans$Class <- factor(as.character(ans$Class), levels = c("0", "1"))
  ans <- ans[sample.int(nrow(ans)), , drop = FALSE]
  rownames(ans) <- NULL
  ans
}

get_weights <- function(train, original, way) {
  if (way != "weighted") return(rep(1, nrow(train)))
  w <- sqrt(sum(original$Class == "0") / sum(original$Class == "1"))
  ifelse(train$Class == "1", w, 1)
}

# I keep the actual fitting here so I can use the same steps for every scenario
# ridge lambda is checked using a smaller CV inside the current training set
fit_one <- function(train, model_name, way, s) {
  if (model_name == "Penalized_Logistic") {
    k <- min(inner_folds, as.integer(min(table(train$Class))))
    if (k < 2) stop("Need at least two positives for inner CV")
    fold_id <- make_folds(train$Class, k, s + 600)
    errors <- matrix(NA_real_, nrow = k, ncol = length(lambdas))
    for (j in seq_len(k)) {
      tr <- train[fold_id != j, , drop = FALSE]
      va <- train[fold_id == j, , drop = FALSE]
      sc <- scale_first(tr)
      tr2 <- change_sample(sc$data, way, s + j)
      wt <- get_weights(tr2, tr, way)
      m <- glmnet::glmnet(as.matrix(tr2[sc$cols]),
                          as.integer(as.character(tr2$Class)), family = "binomial",
                          alpha = 0, lambda = lambdas, weights = wt,
                          standardize = FALSE, maxit = 100000)
      va2 <- scale_other(va, sc)
      pp <- as.matrix(predict(m, newx = as.matrix(va2[sc$cols]),
                              s = lambdas, type = "response"))
      yy <- as.integer(as.character(va$Class))
      for (h in seq_along(lambdas)) {
        ph <- pmin(1 - 1e-12, pmax(1e-12, pp[, h]))
        errors[j, h] <- -mean(yy * log(ph) + (1 - yy) * log(1 - ph))
      }
    }
    picked <- lambdas[which.min(colMeans(errors))]
    sc <- scale_first(train)
    train2 <- change_sample(sc$data, way, s + 100)
    m <- glmnet::glmnet(as.matrix(train2[sc$cols]),
                        as.integer(as.character(train2$Class)), family = "binomial",
                        alpha = 0, lambda = picked,
                        weights = get_weights(train2, train, way),
                        standardize = FALSE, maxit = 100000)
    return(list(kind = model_name, fit = m, scale = sc, lambda = picked))
  }
  sc <- scale_first(train)
  train2 <- change_sample(sc$data, way, s + 100)
  class_weights <- NULL
  if (way == "weighted") {
    w <- sqrt(sum(train$Class == "0") / sum(train$Class == "1"))
    class_weights <- c("0" = 1, "1" = w)
  }
  m <- ranger::ranger(Class ~ ., data = train2, probability = TRUE,
                      num.trees = trees, mtry = max(1, floor(sqrt(length(sc$cols)))),
                      min.node.size = 10, class.weights = class_weights,
                      num.threads = 1, seed = s)
  list(kind = model_name, fit = m, scale = sc)
}

get_predictions <- function(model, newdata) {
  x <- scale_other(newdata, model$scale)
  if (model$kind == "Penalized_Logistic") {
    as.numeric(predict(model$fit, newx = as.matrix(x[model$scale$cols]),
                       s = model$lambda, type = "response"))
  } else {
    as.numeric(predict(model$fit, data = x, num.threads = 1)$predictions[, "1"])
  }
}

# drawing ROC and PR from the held-out probabilities
curve_points <- function(y, prob, type) {
  y <- as.integer(as.character(y))
  ind <- order(prob, decreasing = TRUE)
  y <- y[ind]; prob <- prob[ind]
  at <- which(!duplicated(prob, fromLast = TRUE))
  at <- unique(c(at[seq(1, length(at), by = max(1, floor(length(at)/1200)))], tail(at, 1)))
  tp <- cumsum(y)[at]
  fp <- at - tp
  if (type == "PR") {
    data.frame(x = c(0, tp / sum(y)), y = c(1, tp / at))
  } else {
    data.frame(x = c(0, fp / sum(y == 0)), y = c(0, tp / sum(y == 1)))
  }
}

# this does one experiment on one fixed train/test split
# selection and threshold choices all happen using training OOF predictions
compare_models <- function(train, test, experiment, out_folder, s,
                           target = NA_real_, real_data = FALSE, true_prob = NULL) {
  cat("\nStarting", experiment, "\n")
  f <- make_folds(train$Class, folds, s)
  methods <- expand.grid(Model = models, Method = remedies, stringsAsFactors = FALSE)
  p_cv <- matrix(NA_real_, nrow(train), nrow(methods))
  p_test <- matrix(NA_real_, nrow(test), nrow(methods))
  labels <- paste(methods$Model, methods$Method, sep = "__")
  colnames(p_cv) <- colnames(p_test) <- labels

  for (i in seq_len(nrow(methods))) {
    cat("  ", labels[i], "\n")
    for (j in seq_len(folds)) {
      train_piece <- train[f != j, , drop = FALSE]
      valid_piece <- train[f == j, , drop = FALSE]
      m <- fit_one(train_piece, methods$Model[i], methods$Method[i], s + i*100 + j)
      p_cv[f == j, i] <- get_predictions(m, valid_piece)
    }
    m <- fit_one(train, methods$Model[i], methods$Method[i], s + 20000 + i)
    p_test[, i] <- get_predictions(m, test)
    rm(m)
  }

  # I want both the ordinary cutoff and a tuned one for comparison
  rows <- list()
  add_result <- function(model, method, rule, cutoff, cv_prob, test_prob) {
    cv_s <- get_scores(train$Class, cv_prob, cutoff)
    ts <- get_scores(test$Class, test_prob, cutoff)
    z <- data.frame(
      Experiment = experiment, Target_Prevalence = target,
      Train_N = nrow(train), Test_N = nrow(test),
      Train_Positive_Count = sum(train$Class == "1"),
      Test_Positive_Count = sum(test$Class == "1"),
      Classifier = model, Remedy = method, Operating_Rule = rule,
      Threshold = cutoff, CV_ROC_AUC = cv_s$ROC_AUC,
      CV_PR_AUC_AP = cv_s$PR_AUC_AP, CV_F1_Selected = cv_s$F1,
      ts[, setdiff(names(ts), "Threshold"), drop = FALSE]
    )
    rows[[length(rows) + 1]] <<- z
  }

  add_result("Majority_Class", "predict_all_negative", "all_negative", 0.5,
             rep(0, nrow(train)), rep(0, nrow(test)))
  for (i in seq_along(labels)) {
    add_result(methods$Model[i], methods$Method[i], "default_0.5", 0.5,
               p_cv[, i], p_test[, i])
    threshold <- choose_cutoff(train$Class, p_cv[, i], beta = 1)
    add_result(methods$Model[i], methods$Method[i], "CV_tuned_F1", threshold,
               p_cv[, i], p_test[, i])
    if (methods$Method[i] == "baseline") {
      threshold2 <- choose_cutoff(train$Class, p_cv[, i], beta = 2)
      add_result(methods$Model[i], methods$Method[i], "CV_tuned_F2", threshold2,
                 p_cv[, i], p_test[, i])
    }
  }

  # mixing one logistic model and one random forest -- STAT 847 part
  ap_cv <- sapply(seq_along(labels), function(i) get_ap(train$Class, p_cv[, i]))
  logistic_i <- which(methods$Model == "Penalized_Logistic")
  forest_i <- which(methods$Model == "Random_Forest")
  a <- logistic_i[which.max(ap_cv[logistic_i])]
  b <- forest_i[which.max(ap_cv[forest_i])]
  weights_grid <- seq(0.1, 0.9, 0.1)
  grid_ap <- sapply(weights_grid, function(w) get_ap(train$Class,
                                        w * p_cv[, a] + (1-w) * p_cv[, b]))
  best_w <- weights_grid[which.max(grid_ap)]
  blend_cv <- best_w * p_cv[, a] + (1-best_w) * p_cv[, b]
  blend_test <- best_w * p_test[, a] + (1-best_w) * p_test[, b]
  add_result("Ensemble", "CV_optimized_blend", "default_0.5", 0.5,
             blend_cv, blend_test)
  th <- choose_cutoff(train$Class, blend_cv, beta = 1)
  add_result("Ensemble", "CV_optimized_blend", "CV_tuned_F1", th,
             blend_cv, blend_test)

  if (!is.null(true_prob)) {
    # only for simulation: the oracle is not a trainable candidate
    add_result("Oracle_True_P", "simulation_truth", "default_0.5", 0.5,
               rep(mean(train$Class == "1"), nrow(train)), true_prob)
  }

  result <- do.call(rbind, rows)
  write.csv(result, file.path(out_folder, "tables", paste0(experiment, "__test_metrics.csv")), row.names = FALSE)
  search_weights <- data.frame(Experiment = experiment, Logistic_Weight = weights_grid,
                               RF_Weight = 1 - weights_grid,
                               CV_AP = grid_ap,
                               Selected_Weight = best_w,
                               Logistic_Model = labels[a], RF_Model = labels[b])
  write.csv(search_weights, file.path(out_folder, "tables", paste0(experiment, "__ensemble_weight_search.csv")), row.names = FALSE)

  # best individual model chosen from training OOF only
  best_i <- which.max(ap_cv)
  test_y <- as.integer(as.character(test$Class))
  neg <- which(test_y == 0); pos <- which(test_y == 1)
  set.seed(s + 90000)
  ap_diffs <- replicate(bootstrap_runs, {
    ind <- c(sample(neg, length(neg), replace = TRUE),
             sample(pos, length(pos), replace = TRUE))
    get_ap(test_y[ind], blend_test[ind]) - get_ap(test_y[ind], p_test[ind, best_i])
  })
  boot <- data.frame(Experiment = experiment, Best_Single_CV_Selected = labels[best_i],
                     Selected_Logistic = labels[a], Selected_Forest = labels[b],
                     Optimal_Logistic_Weight = best_w,
                     Ensemble_minus_Single_AP = get_ap(test_y, blend_test) - get_ap(test_y, p_test[, best_i]),
                     Bootstrap_CI_Lower = unname(quantile(ap_diffs, .025)),
                     Bootstrap_CI_Upper = unname(quantile(ap_diffs, .975)),
                     Bootstrap_Pr_Difference_Positive = mean(ap_diffs > 0),
                     Bootstrap_Replicates = bootstrap_runs)
  write.csv(boot, file.path(out_folder, "tables", paste0(experiment, "__ensemble_vs_single_bootstrap.csv")), row.names = FALSE)

  # save more detailed data for the fraud tests so plots can be remade later
  if (real_data) {
    p_out <- data.frame(Truth = test_y,
                        Logistic_Baseline = p_test[, "Penalized_Logistic__baseline"],
                        RF_Baseline = p_test[, "Random_Forest__baseline"],
                        Best_Single = p_test[, best_i], Ensemble = blend_test)
    write.csv(p_out, file.path(out_folder, "tables", paste0(experiment, "__heldout_predictions.csv")), row.names = FALSE)
    for (type in c("PR", "ROC")) {
      plots <- list()
      for (nm in names(p_out)[-1]) {
        d <- curve_points(p_out$Truth, p_out[[nm]], type)
        d$Model <- nm
        plots[[length(plots) + 1]] <- d
      }
      plotting_data <- do.call(rbind, plots)
      p <- ggplot2::ggplot(plotting_data, ggplot2::aes(x, y, colour = Model)) +
        ggplot2::geom_line(linewidth = .75) + ggplot2::theme_minimal() +
        ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
        ggplot2::labs(title = paste(experiment, type),
                      x = if (type == "PR") "Recall" else "False positive rate",
                      y = if (type == "PR") "Precision" else "True positive rate")
      ggplot2::ggsave(file.path(out_folder, "plots", "curves", paste0(experiment, "__", type, ".png")),
                      p, width = 9, height = 5, dpi = 180)
    }
  }
  list(scores = result, bootstrap = boot)
}

# trying the simulated data now
# I am changing the intercept, not the predictor slopes
betas <- c(X1 = 1.25, X2 = -1.0, X3 = 0.8, X4 = 0)
set.seed(seed)
mc_x <- matrix(rnorm(100000 * 4), ncol = 4)
eta <- as.numeric(mc_x %*% betas)
intercepts <- sapply(rates, function(r) uniroot(function(b0) mean(plogis(b0 + eta)) - r,
                                                c(-20, 20), tol = 1e-10)$root)
names(intercepts) <- as.character(rates)

sim_grid <- expand.grid(Prevalence = rates, N = sizes, Repeat = seq_len(repeats))
sim_grid <- rbind(sim_grid, data.frame(Prevalence = .01, N = 100000, Repeat = seq_len(repeats)))
write.csv(sim_grid, file.path(sim_folder, "tables", "simulation__design_grid.csv"), row.names = FALSE)
write.csv(data.frame(Prevalence = rates, Intercept = intercepts,
                     beta1 = betas[1], beta2 = betas[2], beta3 = betas[3], beta4 = betas[4]),
          file.path(sim_folder, "tables", "simulation__population_coefficients.csv"), row.names = FALSE)

all_sim <- list(); sim_boot <- list(); sim_details <- list()
for (i in seq_len(nrow(sim_grid))) {
  n <- sim_grid$N[i]; rate <- sim_grid$Prevalence[i]; rep_no <- sim_grid$Repeat[i]
  name <- sprintf("sim_n%d_p%02d_rep%02d", n, round(100 * rate), rep_no)
  s <- seed + 101*i
  set.seed(s)
  x <- matrix(rnorm(n * 4), ncol = 4)
  colnames(x) <- names(betas)
  real_p <- plogis(as.numeric(intercepts[as.character(rate)] + x %*% betas))
  y <- rbinom(n, 1, real_p)
  d <- as.data.frame(x)
  d$Class <- factor(as.character(y), levels = c("0", "1"))
  data.table::fwrite(d, file.path(sim_folder, "data", "generated", paste0(name, ".csv.gz")), compress = "gzip")

  if (rep_no == 1) {
    p <- ggplot2::ggplot(d, ggplot2::aes(Class, fill = Class)) +
      ggplot2::geom_bar(show.legend = FALSE) + ggplot2::scale_y_log10() +
      ggplot2::theme_minimal() + ggplot2::labs(title = paste(name, "class counts"))
    ggplot2::ggsave(file.path(sim_folder, "plots", "eda", paste0(name, "__counts.png")),
                    p, width = 7, height = 4, dpi = 160)
    p <- ggplot2::ggplot(d, ggplot2::aes(X1, fill = Class)) +
      ggplot2::geom_density(alpha = .4) + ggplot2::theme_minimal() +
      ggplot2::labs(title = paste(name, "X1 distribution"))
    ggplot2::ggsave(file.path(sim_folder, "plots", "eda", paste0(name, "__X1.png")),
                    p, width = 7, height = 4, dpi = 160)
  }

  # I kept the source-row position only to line up the true probabilities
  d$.id <- seq_len(nrow(d))
  parts <- split_random(d, s + 1)
  test_id <- parts$test$.id
  parts$train$.id <- NULL
  parts$test$.id <- NULL
  result <- compare_models(parts$train, parts$test, name, sim_folder, s + 10,
                           target = rate, true_prob = real_p[test_id])
  all_sim[[i]] <- result$scores
  sim_boot[[i]] <- result$bootstrap
  sim_details[[i]] <- data.frame(Experiment = name, N = n, Repeat = rep_no,
    Target_Prevalence = rate, Observed_Prevalence = mean(y),
    Observed_Positives = sum(y), Train_Positives = sum(parts$train$Class == "1"),
    Test_Positives = sum(parts$test$Class == "1"), Population_Intercept = intercepts[as.character(rate)])
  write.csv(do.call(rbind, all_sim), file.path(sim_folder, "tables", "simulation__all_metrics.csv"), row.names = FALSE)
  write.csv(do.call(rbind, sim_boot), file.path(sim_folder, "tables", "simulation__ensemble_comparisons.csv"), row.names = FALSE)
  write.csv(do.call(rbind, sim_details), file.path(sim_folder, "tables", "simulation__scenario_summary.csv"), row.names = FALSE)
}

sim_results <- do.call(rbind, all_sim)
sim_info <- do.call(rbind, sim_details)
# comparing remedies with baseline on exactly the same test observations
base <- subset(sim_results, Classifier %in% models & Remedy == "baseline" & Operating_Rule == "default_0.5")
other <- subset(sim_results, Classifier %in% models & !(Remedy == "baseline" & Operating_Rule == "default_0.5"))
changes <- merge(other, base[, c("Experiment", "Classifier", "F1", "PR_AUC_AP", "Sensitivity", "Precision", "Balanced_Accuracy")],
                 by = c("Experiment", "Classifier"), suffixes = c("", "_Baseline"))
for (metric in c("F1", "PR_AUC_AP", "Sensitivity", "Precision", "Balanced_Accuracy")) {
  changes[[paste0("Delta_", metric)]] <- changes[[metric]] - changes[[paste0(metric, "_Baseline")]]
}
write.csv(changes, file.path(sim_folder, "tables", "simulation__remedy_deltas.csv"), row.names = FALSE)

# some working figures. I'll decide which ones belong in the report afterwards
sim_plot <- subset(sim_results, Classifier %in% models & Operating_Rule == "default_0.5")
averages <- aggregate(cbind(PR_AUC_AP, ROC_AUC, Balanced_Accuracy) ~
                        Target_Prevalence + Train_N + Classifier + Remedy,
                      data = sim_plot, FUN = mean)
write.csv(averages, file.path(sim_folder, "tables", "simulation__plot_averages.csv"), row.names = FALSE)
for (metric in c("PR_AUC_AP", "ROC_AUC", "Balanced_Accuracy")) {
  d <- averages
  d$Result <- d[[metric]]
  p <- ggplot2::ggplot(d, ggplot2::aes(Target_Prevalence * 100, Result,
                                      colour = Remedy, group = Remedy)) +
    ggplot2::geom_line() + ggplot2::geom_point() +
    ggplot2::facet_grid(Classifier ~ Train_N) + ggplot2::theme_minimal() +
    ggplot2::scale_x_continuous(breaks = c(1, 5, 20, 50)) +
    ggplot2::labs(x = "Target fraud prevalence (%)", y = metric, title = paste(metric, "in simulations"))
  ggplot2::ggsave(file.path(sim_folder, "plots", "final", paste0("sim_", metric, ".png")),
                  p, width = 11, height = 7, dpi = 190)
}

tuned <- subset(sim_results, Classifier %in% models & Remedy == "baseline" &
                  Operating_Rule %in% c("default_0.5", "CV_tuned_F1"))
p <- ggplot2::ggplot(tuned, ggplot2::aes(factor(Target_Prevalence), F1, colour = Operating_Rule)) +
  ggplot2::stat_summary(fun = mean, geom = "point", size = 2) +
  ggplot2::stat_summary(fun = mean, geom = "line", aes(group = Operating_Rule)) +
  ggplot2::facet_grid(Classifier ~ Train_N) + ggplot2::theme_minimal() +
  ggplot2::labs(title = "Threshold tuning and F1", x = "Target prevalence")
ggplot2::ggsave(file.path(sim_folder, "plots", "final", "sim_threshold_F1.png"), p,
                width = 11, height = 7, dpi = 190)

p <- ggplot2::ggplot(changes[changes$Operating_Rule == "default_0.5", ],
                     ggplot2::aes(Target_Prevalence * 100, Delta_PR_AUC_AP,
                                  colour = Remedy, group = Remedy)) +
  ggplot2::stat_summary(fun = mean, geom = "line") +
  ggplot2::stat_summary(fun = mean, geom = "point") +
  ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
  ggplot2::facet_grid(Classifier ~ Train_N) + ggplot2::theme_minimal() +
  ggplot2::labs(title = "Change in AP compared with baseline", x = "Target prevalence (%)")
ggplot2::ggsave(file.path(sim_folder, "plots", "final", "sim_remedy_AP_change.png"), p,
                width = 11, height = 7, dpi = 190)

# I also want a plot of the actual number of positive examples
pos_data <- subset(sim_results, Classifier %in% models & Remedy == "baseline" &
                   Operating_Rule == "default_0.5")
p <- ggplot2::ggplot(pos_data, ggplot2::aes(Train_Positive_Count, PR_AUC_AP,
                         colour = factor(Target_Prevalence * 100))) +
  ggplot2::geom_point(alpha = .7) + ggplot2::scale_x_log10() +
  ggplot2::facet_wrap(~Classifier) + ggplot2::theme_minimal() +
  ggplot2::labs(title = "Number of positive training examples and test AP",
                x = "Positive training examples (log scale)", y = "Held-out AP",
                colour = "Target prevalence (%)")
ggplot2::ggsave(file.path(sim_folder, "plots", "final", "sim_number_of_positives.png"),
                p, width = 10, height = 5, dpi = 180)

# mean recall and precision for the operating rules tuned using training CV
pr <- subset(sim_results, Classifier %in% models & Operating_Rule == "CV_tuned_F1")
pr$Scenario <- paste(pr$Train_N, "train,", 100 * pr$Target_Prevalence, "%")
p <- ggplot2::ggplot(pr, ggplot2::aes(Sensitivity, Precision, colour = Remedy,
                                      shape = Classifier)) +
  ggplot2::geom_point(alpha = .65) + ggplot2::facet_wrap(~Scenario) +
  ggplot2::theme_minimal() +
  ggplot2::labs(title = "Precision and recall after threshold tuning")
ggplot2::ggsave(file.path(sim_folder, "plots", "final", "sim_precision_recall_points.png"),
                p, width = 12, height = 8, dpi = 180)

# ensemble improvements with bootstrap intervals; one row per simulation repeat
bdf <- do.call(rbind, sim_boot)
bdf <- merge(bdf, sim_info[, c("Experiment", "N", "Target_Prevalence")], by = "Experiment")
write.csv(bdf, file.path(sim_folder, "tables", "simulation__ensemble_plot_data.csv"), row.names = FALSE)
p <- ggplot2::ggplot(bdf, ggplot2::aes(Experiment, Ensemble_minus_Single_AP)) +
  ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
  ggplot2::geom_pointrange(ggplot2::aes(ymin = Bootstrap_CI_Lower,
                                        ymax = Bootstrap_CI_Upper)) +
  ggplot2::coord_flip() + ggplot2::theme_minimal() +
  ggplot2::labs(title = "Ensemble minus best CV-selected single learner",
                y = "Difference in held-out AP", x = "Simulation repeat")
ggplot2::ggsave(file.path(sim_folder, "plots", "final", "sim_ensemble_vs_single.png"),
                p, width = 12, height = 10, dpi = 180)

# time to try the real fraud data
# the original ULB dataset should have 284807 rows and 492 fraud cases
if (!file.exists(csv_file)) stop(paste("Could not find the csv file:", csv_file))
fraud <- as.data.frame(data.table::fread(csv_file))
required <- c("Time", paste0("V", 1:28), "Amount", "Class")
if (!all(required %in% names(fraud))) stop("This CSV does not have the original ULB columns")
if (nrow(fraud) != 284807 || sum(fraud$Class == 1, na.rm = TRUE) != 492) {
  stop("Check the dataset: expected 284807 rows and 492 fraud transactions")
}
if (anyNA(fraud) || !all(fraud$Class %in% c(0, 1))) stop("Fraud dataset has missing or invalid values")
file.copy(csv_file, file.path(fraud_folder, "data", "raw", "creditcard.csv"), overwrite = TRUE)
fraud$Class <- factor(as.character(fraud$Class), levels = c("0", "1"))

# a few EDA things before changing the amount variable
counts <- as.data.frame(table(fraud$Class))
names(counts) <- c("Class", "Count")
counts$Percent <- 100 * counts$Count / nrow(fraud)
write.csv(counts, file.path(fraud_folder, "tables", "fraud__EDA_class_counts.csv"), row.names = FALSE)
dups <- data.frame(Original_N = nrow(fraud), Exact_Duplicates = sum(duplicated(fraud)),
                   Fraud_Duplicates = sum(duplicated(fraud) & fraud$Class == "1"))
write.csv(dups, file.path(fraud_folder, "tables", "fraud__EDA_duplicate_check.csv"), row.names = FALSE)
p <- ggplot2::ggplot(counts, ggplot2::aes(Class, Count, fill = Class)) +
  ggplot2::geom_col(show.legend = FALSE) + ggplot2::scale_y_log10() + ggplot2::theme_minimal()
ggplot2::ggsave(file.path(fraud_folder, "plots", "eda", "fraud_class_counts.png"), p,
                width = 7, height = 4.5, dpi = 180)
p <- ggplot2::ggplot(fraud, ggplot2::aes(log1p(Amount), fill = Class)) +
  ggplot2::geom_density(alpha = .4) + ggplot2::theme_minimal() +
  ggplot2::labs(title = "Transaction amounts by fraud class")
ggplot2::ggsave(file.path(fraud_folder, "plots", "eda", "fraud_amounts.png"), p,
                width = 7, height = 4.5, dpi = 180)

fraud$Amount <- log1p(fraud$Amount)
fraud <- fraud[!duplicated(fraud), , drop = FALSE]
rownames(fraud) <- NULL
# the scaled data are only created inside each training fold
# this transformation was decided before seeing the results
write.csv(data.frame(Data = "original creditcard.csv", Raw_N = 284807,
                     Raw_Fraud = 492, After_Removing_Duplicates = nrow(fraud),
                     After_Removing_Duplicates_Fraud = sum(fraud$Class == "1")),
          file.path(fraud_folder, "tables", "fraud__dataset_summary.csv"), row.names = FALSE)
data.table::fwrite(fraud, file.path(fraud_folder, "data", "processed", "creditcard_model_ready.csv.gz"), compress = "gzip")

fraud$.id <- seq_len(nrow(fraud))
# I want to try both a regular random split and a later-in-time test
all_fraud <- list(); all_fraud_boot <- list(); split_info <- list()
for (kind in c("fraud_stratified_random", "fraud_chronological_holdout")) {
  if (kind == "fraud_stratified_random") {
    sp <- split_random(fraud, seed + 300)
  } else {
    ordered <- fraud[order(fraud$Time, seq_len(nrow(fraud))), , drop = FALSE]
    ntrain <- floor(0.80 * nrow(ordered))
    sp <- list(train = ordered[seq_len(ntrain), , drop = FALSE],
               test = ordered[seq.int(ntrain + 1, nrow(ordered)), , drop = FALSE])
  }
  for (part in c("train", "test")) {
    data.table::fwrite(data.frame(Processed_Row_ID = sp[[part]]$.id),
      file.path(fraud_folder, "data", "splits", paste0(kind, "__", part, "_rows.csv.gz")), compress = "gzip")
  }
  split_info[[kind]] <- data.frame(Experiment = kind, Train_N = nrow(sp$train),
    Test_N = nrow(sp$test), Train_Positives = sum(sp$train$Class == "1"),
    Test_Positives = sum(sp$test$Class == "1"),
    Train_Prevalence = mean(sp$train$Class == "1"), Test_Prevalence = mean(sp$test$Class == "1"),
    Chronological_Boundary = if (kind == "fraud_chronological_holdout") max(sp$train$Time) else NA_real_)
  sp$train$.id <- NULL; sp$test$.id <- NULL
  out <- compare_models(sp$train, sp$test, kind, fraud_folder, seed + 400 + length(all_fraud),
                        target = 492 / 284807, real_data = TRUE)
  all_fraud[[kind]] <- out$scores
  all_fraud_boot[[kind]] <- out$bootstrap
}
fraud_results <- do.call(rbind, all_fraud)
write.csv(do.call(rbind, split_info), file.path(fraud_folder, "tables", "fraud__split_summary.csv"), row.names = FALSE)
write.csv(fraud_results, file.path(fraud_folder, "tables", "fraud__all_metrics.csv"), row.names = FALSE)
write.csv(do.call(rbind, all_fraud_boot), file.path(fraud_folder, "tables", "fraud__ensemble_comparisons.csv"), row.names = FALSE)

# best selected using CV columns, just reporting the test metrics
selected <- subset(fraud_results, Classifier %in% c(models, "Ensemble") & Operating_Rule == "CV_tuned_F1")
selected <- do.call(rbind, lapply(split(selected, selected$Experiment), function(d) {
  d[order(-d$CV_PR_AUC_AP, -d$CV_F1_Selected), , drop = FALSE][1, , drop = FALSE]
}))
write.csv(selected, file.path(fraud_folder, "tables", "fraud__CV_selected_models.csv"), row.names = FALSE)

# plots for the fraud results
fraud_fig <- subset(fraud_results, Classifier %in% models & Operating_Rule == "default_0.5")
for (metric in c("PR_AUC_AP", "ROC_AUC", "Brier")) {
  df <- fraud_fig
  df$Value <- df[[metric]]
  p <- ggplot2::ggplot(df, ggplot2::aes(Remedy, Value, fill = Classifier)) +
    ggplot2::geom_col(position = "dodge") + ggplot2::coord_flip() +
    ggplot2::facet_wrap(~Experiment, scales = "free_y") +
    ggplot2::theme_minimal() + ggplot2::labs(title = paste("Fraud:", metric))
  ggplot2::ggsave(file.path(fraud_folder, "plots", "final", paste0("fraud_", metric, ".png")),
                  p, width = 11, height = 6, dpi = 190)
}
threshold_df <- subset(fraud_results, Classifier %in% models & Remedy == "baseline" &
  Operating_Rule %in% c("default_0.5", "CV_tuned_F1"))
p <- ggplot2::ggplot(threshold_df, ggplot2::aes(Classifier, F1, fill = Operating_Rule)) +
  ggplot2::geom_col(position = "dodge") + ggplot2::facet_wrap(~ Experiment) +
  ggplot2::theme_minimal() + ggplot2::labs(title = "Fraud: default versus tuned F1")
ggplot2::ggsave(file.path(fraud_folder, "plots", "final", "fraud_threshold_F1.png"),
                p, width = 11, height = 6, dpi = 190)

# visualizing the chosen model confusion matrix
cm <- do.call(rbind, lapply(seq_len(nrow(selected)), function(i) {
  x <- selected[i, ]
  data.frame(Experiment = x$Experiment, Actual = c("Fraud", "Fraud", "Legitimate", "Legitimate"),
             Predicted = c("Fraud", "Legitimate", "Fraud", "Legitimate"),
             Count = c(x$TP, x$FN, x$FP, x$TN))
}))
write.csv(cm, file.path(fraud_folder, "tables", "fraud__confusion_matrix_cells.csv"), row.names = FALSE)
p <- ggplot2::ggplot(cm, ggplot2::aes(Predicted, Actual, fill = Count)) +
  ggplot2::geom_tile() + ggplot2::geom_text(ggplot2::aes(label = Count)) +
  ggplot2::facet_wrap(~Experiment) + ggplot2::theme_minimal() +
  ggplot2::labs(title = "Fraud: held-out confusion matrices")
ggplot2::ggsave(file.path(fraud_folder, "plots", "final", "fraud_confusion_matrices.png"),
                p, width = 11, height = 5.5, dpi = 190)

# more figures to see how the methods compare at their chosen cutoffs
pr_real <- subset(fraud_results, Classifier %in% models & Operating_Rule == "CV_tuned_F1")
p <- ggplot2::ggplot(pr_real, ggplot2::aes(Sensitivity, Precision,
                                            colour = Remedy, shape = Classifier)) +
  ggplot2::geom_point(size = 3) + ggplot2::facet_wrap(~Experiment) +
  ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  ggplot2::theme_minimal() +
  ggplot2::labs(title = "Fraud: precision versus sensitivity at tuned thresholds")
ggplot2::ggsave(file.path(fraud_folder, "plots", "final", "fraud_precision_recall_tradeoff.png"),
                p, width = 11, height = 6, dpi = 180)

weight_files <- list.files(file.path(fraud_folder, "tables"),
                           pattern = "ensemble_weight_search\\.csv$", full.names = TRUE)
all_weights <- do.call(rbind, lapply(weight_files, function(f) read.csv(f)))
if (!is.null(all_weights) && nrow(all_weights)) {
  p <- ggplot2::ggplot(all_weights, ggplot2::aes(Logistic_Weight, CV_AP)) +
    ggplot2::geom_line() + ggplot2::geom_point() +
    ggplot2::geom_vline(ggplot2::aes(xintercept = Selected_Weight),
                        linetype = "dashed") +
    ggplot2::facet_wrap(~ Experiment) + ggplot2::theme_minimal() +
    ggplot2::labs(title = "Fraud: CV choice of ensemble mixing weight")
  ggplot2::ggsave(file.path(fraud_folder, "plots", "final", "fraud_ensemble_weights.png"),
                  p, width = 10, height = 5, dpi = 180)
}

# last thing -- tables to compare the simulation with the real data
# they are very different prevalences so I'm NOT pooling their labels or test scores
needed_cols <- c("Experiment", "Classifier", "Remedy", "Operating_Rule", "Train_N", "Test_N",
                 "Prevalence", "Sensitivity", "Specificity", "Precision", "F1",
                 "Balanced_Accuracy", "ROC_AUC", "PR_AUC_AP")
sim_final <- subset(sim_results, Operating_Rule == "default_0.5")[, needed_cols]
fraud_final <- subset(fraud_results, Operating_Rule == "default_0.5")[, needed_cols]
sim_final$Study <- "Simulation"
fraud_final$Study <- "Fraud"
write.csv(rbind(sim_final, fraud_final),
          file.path(compare_folder, "both_studies_default_threshold.csv"), row.names = FALSE)
write.csv(averages, file.path(compare_folder, "simulation_average_metrics.csv"), row.names = FALSE)
write.csv(fraud_final, file.path(compare_folder, "fraud_holdout_metrics.csv"), row.names = FALSE)

writeLines(capture.output(sessionInfo()), file.path(sim_folder, "metadata", "R_session_info.txt"))
writeLines(capture.output(sessionInfo()), file.path(fraud_folder, "metadata", "R_session_info.txt"))
writeLines(capture.output(list(seed = seed, folds = folds, inner_folds = inner_folds,
      rates = rates, sizes = sizes, repeats = repeats, lambdas = lambdas, trees = trees)),
      file.path(sim_folder, "metadata", "settings.txt"))
writeLines(c("Dataset: original ULB credit card fraud data", paste("Source file:", csv_file),
      "Fraud class = 1. Test sets were not resampled.",
      "For chronological holdout, inner and outer training CV use random stratified folds, NOT forward chaining."),
      file.path(fraud_folder, "metadata", "notes.txt"))
cat("\nI finished the simulation and fraud runs.\n")
cat("Simulation results:", sim_folder, "\n")
cat("Fraud results:", fraud_folder, "\n")
cat("Comparison files:", compare_folder, "\n")

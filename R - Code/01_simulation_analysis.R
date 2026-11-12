# STAT 847 Team 2 - Part 1: simulation
# Question: what happens to prediction when the positive class is rare?
# Source this file from the project root in RStudio.

source("00_shared_functions.R", local = FALSE)
check_packages()
options(warn = 1)
cfg <- make_settings()
set.seed(cfg$seed)

sim_folder <- file.path(getwd(), "results", "simulation")
make_folders(sim_folder, c(
  "tables", "data/generated", "data/design", "data/splits",
  "plots/eda", "plots/final", "plots/curves", "metadata"
))

# 1. Data-generating model
# Keep the slopes and predictors fixed; only change the intercept to set prevalence.
betas <- c(X1 = 1.25, X2 = -1.0, X3 = 0.8, X4 = 0)
set.seed(cfg$seed)
mc_x <- matrix(rnorm(100000L * length(betas)), ncol = length(betas))
linear_part <- as.numeric(mc_x %*% betas)
intercepts <- vapply(cfg$rates, function(rate) {
  uniroot(function(b0) mean(plogis(b0 + linear_part)) - rate,
          interval = c(-20, 20), tol = 1e-10)$root
}, numeric(1))
names(intercepts) <- as.character(cfg$rates)

sim_grid <- expand.grid(
  Prevalence = cfg$rates, N = cfg$sizes, Repeat = seq_len(cfg$repeats)
)
if (cfg$large_rare_scenario) {
  sim_grid <- rbind(sim_grid, data.frame(
    Prevalence = 0.01, N = 100000L, Repeat = seq_len(cfg$repeats)
  ))
}
save_table(sim_grid, sim_folder, "simulation__design_grid.csv")
save_table(data.frame(
  Prevalence = cfg$rates, Intercept = intercepts,
  beta1 = unname(betas[1]), beta2 = unname(betas[2]),
  beta3 = unname(betas[3]), beta4 = unname(betas[4])
), sim_folder, "simulation__population_coefficients.csv")
writeLines(c(
  "Data-generating model: Bernoulli(expit(beta0 + 1.25*X1 - 1.0*X2 + 0.8*X3 + 0*X4))",
  "Predictors X1-X4: independent standard normal", "Intercept calibrated for each target prevalence",
  "The population model and slopes stay constant across scenarios."
), file.path(sim_folder, "data", "design", "simulation_design_notes.txt"))
rm(mc_x, linear_part)

# 2. Run simulation experiments
all_sim <- list()
sim_boot <- list()
sim_details <- list()
for (i in seq_len(nrow(sim_grid))) {
  n <- sim_grid$N[i]
  rate <- sim_grid$Prevalence[i]
  repeat_id <- sim_grid$Repeat[i]
  name <- sprintf("sim_n%d_p%02d_rep%02d", n, round(100 * rate), repeat_id)
  s <- cfg$seed + 101L * i
  cat("\nSimulation", i, "of", nrow(sim_grid), ":", name, "\n")

  set.seed(s)
  x <- matrix(rnorm(n * length(betas)), ncol = length(betas))
  colnames(x) <- names(betas)
  true_p <- plogis(as.numeric(intercepts[as.character(rate)] + x %*% betas))
  y <- rbinom(n, 1, true_p)
  d <- as.data.frame(x)
  d$Class <- factor(as.character(y), levels = c("0", "1"))
  data.table::fwrite(d, file.path(sim_folder, "data", "generated", paste0(name, ".csv.gz")),
                     compress = "gzip")

  # Descriptive figures on original unsampled observations, first repeat only.
  if (repeat_id == 1L) {
    p1 <- ggplot2::ggplot(d, ggplot2::aes(Class, fill = Class)) +
      ggplot2::geom_bar(show.legend = FALSE) + ggplot2::scale_y_log10() +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::labs(title = paste(name, "class counts"), y = "Count (log scale)")
    save_plot(p1, sim_folder, "eda", paste0(name, "__counts.png"), 7, 4)
    p2 <- ggplot2::ggplot(d, ggplot2::aes(X1, fill = Class)) +
      ggplot2::geom_density(alpha = 0.4) + ggplot2::theme_minimal(base_size = 12) +
      ggplot2::labs(title = paste(name, "X1 distributions"))
    save_plot(p2, sim_folder, "eda", paste0(name, "__X1.png"), 7, 4)
  }

  d$.id <- seq_len(nrow(d))
  parts <- split_random(d, s + 1L, cfg$test_fraction)
  test_ids <- parts$test$.id
  for (part in c("train", "test")) {
    data.table::fwrite(data.frame(Original_Row_ID = parts[[part]]$.id),
                       file.path(sim_folder, "data", "splits",
                                 paste0(name, "__", part, "_rows.csv.gz")), compress = "gzip")
    parts[[part]]$.id <- NULL
  }
  comparison <- compare_models(parts$train, parts$test, name, sim_folder,
                               s + 10L, cfg, target = rate,
                               true_prob = true_p[test_ids])
  all_sim[[i]] <- comparison$scores
  sim_boot[[i]] <- comparison$bootstrap
  sim_details[[i]] <- data.frame(
    Experiment = name, N = n, Repeat = repeat_id,
    Target_Prevalence = rate, Observed_Prevalence = mean(y),
    Observed_Positives = sum(y), Train_Positives = sum(parts$train$Class == "1"),
    Test_Positives = sum(parts$test$Class == "1"),
    Population_Intercept = unname(intercepts[as.character(rate)])
  )
  # Saving after each scenario makes it possible to inspect partial progress.
  save_table(do.call(rbind, all_sim), sim_folder, "simulation__all_metrics.csv")
  save_table(do.call(rbind, sim_boot), sim_folder, "simulation__ensemble_comparisons.csv")
  save_table(do.call(rbind, sim_details), sim_folder, "simulation__scenario_summary.csv")
}

sim_results <- do.call(rbind, all_sim)
sim_info <- do.call(rbind, sim_details)

# 3. Compare each remedy against its own baseline on the SAME held-out test.
base <- subset(sim_results, Classifier %in% cfg$models & Remedy == "baseline" &
                 Operating_Rule == "default_0.5")
other <- subset(sim_results, Classifier %in% cfg$models &
                  !(Remedy == "baseline" & Operating_Rule == "default_0.5"))
changes <- merge(other,
                 base[, c("Experiment", "Classifier", "F1", "PR_AUC_AP",
                          "Sensitivity", "Precision", "Balanced_Accuracy")],
                 by = c("Experiment", "Classifier"), suffixes = c("", "_Baseline"))
for (metric in c("F1", "PR_AUC_AP", "Sensitivity", "Precision", "Balanced_Accuracy")) {
  changes[[paste0("Delta_", metric)]] <- changes[[metric]] -
    changes[[paste0(metric, "_Baseline")]]
}
save_table(changes, sim_folder, "simulation__remedy_deltas.csv")

# 4. Final simulation figures
# These use held-out predictions; the averages combine Monte Carlo repeats.
sim_plot <- subset(sim_results, Classifier %in% cfg$models &
                     Operating_Rule == "default_0.5")
averages <- aggregate(cbind(PR_AUC_AP, ROC_AUC, Balanced_Accuracy) ~
                        Target_Prevalence + Train_N + Classifier + Remedy,
                      data = sim_plot, FUN = mean)
save_table(averages, sim_folder, "simulation__plot_averages.csv")
for (metric in c("PR_AUC_AP", "ROC_AUC", "Balanced_Accuracy")) {
  plot_data <- averages
  plot_data$Result <- plot_data[[metric]]
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(Target_Prevalence * 100, Result,
                                                colour = Remedy, group = Remedy)) +
    ggplot2::geom_line() + ggplot2::geom_point() +
    ggplot2::facet_grid(Classifier ~ Train_N) + ggplot2::theme_minimal(base_size = 11) +
    ggplot2::scale_x_continuous(breaks = c(1, 5, 20, 50)) +
    ggplot2::labs(title = paste("Simulation:", metric),
                  x = "Target positive prevalence (%)", y = metric)
  save_plot(p, sim_folder, "final", paste0("sim_", metric, ".png"), 11, 7)
}

# Compare an ordinary 0.5 cutoff against a training-OOF-tuned F1 cutoff.
tuned <- subset(sim_results, Classifier %in% cfg$models & Remedy == "baseline" &
                  Operating_Rule %in% c("default_0.5", "CV_tuned_F1"))
p <- ggplot2::ggplot(tuned, ggplot2::aes(factor(Target_Prevalence), F1,
                                         colour = Operating_Rule,
                                         group = Operating_Rule)) +
  ggplot2::stat_summary(fun = mean, geom = "point", size = 2) +
  ggplot2::stat_summary(fun = mean, geom = "line") +
  ggplot2::facet_grid(Classifier ~ Train_N) +
  ggplot2::theme_minimal(base_size = 11) +
  ggplot2::labs(title = "Threshold tuning and held-out F1", x = "Target prevalence", y = "F1")
save_plot(p, sim_folder, "final", "sim_threshold_F1.png", 11, 7)

# How much does AP change from the unmodified learner?
delta_plot <- changes[changes$Operating_Rule == "default_0.5", , drop = FALSE]
p <- ggplot2::ggplot(delta_plot, ggplot2::aes(Target_Prevalence * 100, Delta_PR_AUC_AP,
                                               colour = Remedy, group = Remedy)) +
  ggplot2::stat_summary(fun = mean, geom = "line") +
  ggplot2::stat_summary(fun = mean, geom = "point") +
  ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
  ggplot2::facet_grid(Classifier ~ Train_N) +
  ggplot2::theme_minimal(base_size = 11) +
  ggplot2::labs(title = "Change in held-out AP relative to baseline",
                x = "Target prevalence (%)", y = "Difference in AP")
save_plot(p, sim_folder, "final", "sim_remedy_AP_change.png", 11, 7)

# Minority count and minority prevalence are different aspects of imbalance.
positive_data <- subset(sim_results, Classifier %in% cfg$models & Remedy == "baseline" &
                          Operating_Rule == "default_0.5")
p <- ggplot2::ggplot(positive_data, ggplot2::aes(Train_Positive_Count, PR_AUC_AP,
                                                  colour = factor(Target_Prevalence * 100))) +
  ggplot2::geom_point(alpha = 0.7) + ggplot2::scale_x_log10() +
  ggplot2::facet_wrap(~ Classifier) + ggplot2::theme_minimal(base_size = 11) +
  ggplot2::labs(title = "Number of training positives versus held-out AP",
                x = "Number of positive training observations (log scale)",
                y = "Average precision", colour = "Target prevalence (%)")
save_plot(p, sim_folder, "final", "sim_number_of_positives.png", 10, 5)

pr <- subset(sim_results, Classifier %in% cfg$models & Operating_Rule == "CV_tuned_F1")
pr$Scenario <- paste(pr$Train_N, "train,", 100 * pr$Target_Prevalence, "%")
p <- ggplot2::ggplot(pr, ggplot2::aes(Sensitivity, Precision,
                                       colour = Remedy, shape = Classifier)) +
  ggplot2::geom_point(alpha = 0.65) + ggplot2::facet_wrap(~ Scenario) +
  ggplot2::theme_minimal(base_size = 11) +
  ggplot2::labs(title = "Precision and recall at training-tuned thresholds")
save_plot(p, sim_folder, "final", "sim_precision_recall_points.png", 12, 8)

bdf <- merge(do.call(rbind, sim_boot),
             sim_info[, c("Experiment", "N", "Target_Prevalence")], by = "Experiment")
save_table(bdf, sim_folder, "simulation__ensemble_plot_data.csv")
p <- ggplot2::ggplot(bdf, ggplot2::aes(Experiment, Ensemble_minus_Single_AP)) +
  ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
  ggplot2::geom_pointrange(ggplot2::aes(ymin = Bootstrap_CI_Lower,
                                        ymax = Bootstrap_CI_Upper)) +
  ggplot2::coord_flip() + ggplot2::theme_minimal(base_size = 11) +
  ggplot2::labs(title = "Ensemble minus best CV-selected single learner",
                y = "Difference in held-out AP", x = "Simulation experiment")
save_plot(p, sim_folder, "final", "sim_ensemble_vs_single.png", 12, 10)

writeLines(capture.output(str(cfg)), file.path(sim_folder, "metadata", "settings.txt"))
writeLines(capture.output(sessionInfo()), file.path(sim_folder, "metadata", "R_session_info.txt"))
cat("\nSimulation finished. Results:", sim_folder, "\n")

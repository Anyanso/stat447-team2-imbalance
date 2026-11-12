# STAT 847 Team 2 - Part 2: original ULB credit-card fraud data
# Read the CSV already on the computer, not through Kaggle or another API.
# Source from the RStudio project root. Run 01 separately first if desired.

source("00_shared_functions.R", local = FALSE)
check_packages()
options(warn = 1)
cfg <- make_settings()

# EDIT THIS LOCATION if the original creditcard.csv is not in Downloads.
# You may also set the environment variable STAT847_FRAUD_CSV before running.
csv_file <- Sys.getenv("STAT847_FRAUD_CSV",
                       unset = "C:/Users/HP/Downloads/creditcard.csv")

sim_folder <- file.path(getwd(), "results", "simulation")
fraud_folder <- file.path(sim_folder, "fraud_real_data")
make_folders(fraud_folder, c(
  "tables", "data/raw", "data/processed", "data/splits",
  "plots/eda", "plots/curves", "plots/final", "metadata"
))

# 1. Validate and preserve the ORIGINAL local dataset
if (!file.exists(csv_file)) stop("Cannot find creditcard.csv at: ", csv_file,
                                 "\nEdit csv_file near the top of this script.")
fraud <- as.data.frame(data.table::fread(csv_file, showProgress = TRUE))
required <- c("Time", paste0("V", 1:28), "Amount", "Class")
if (!setequal(names(fraud), required)) {
  stop("The CSV must have exactly the original ULB columns: Time, V1-V28, Amount, Class")
}
fraud <- fraud[, required, drop = FALSE]
if (!all(vapply(fraud, is.numeric, logical(1)))) stop("All original CSV columns should be numeric")
if (anyNA(fraud) || any(!is.finite(as.matrix(fraud)))) stop("The original data contain missing/nonfinite values")
if (!all(fraud$Class %in% c(0, 1))) stop("The response Class must be 0 or 1")
if (nrow(fraud) != 284807L || sum(fraud$Class == 1L) != 492L) {
  stop("Unexpected dataset: the original ULB CSV has 284,807 rows and 492 frauds")
}
if (any(fraud$Amount < 0)) stop("Transaction amounts cannot be negative")

original_copy <- file.path(fraud_folder, "data", "raw", "creditcard.csv")
if (!file.copy(csv_file, original_copy, overwrite = TRUE)) {
  stop("Unable to copy the original CSV into the results directory")
}
fraud$Class <- factor(as.character(fraud$Class), levels = c("0", "1"))

# 2. Describe imbalance BEFORE preprocessing or any resampling
counts <- as.data.frame(table(fraud$Class))
names(counts) <- c("Class", "Count")
counts$Percent <- 100 * counts$Count / nrow(fraud)
save_table(counts, fraud_folder, "fraud__EDA_class_counts.csv")
missing <- data.frame(Variable = names(fraud),
                      Missing = vapply(fraud, function(z) sum(is.na(z)), integer(1)))
save_table(missing, fraud_folder, "fraud__EDA_missingness.csv")
dups <- data.frame(Original_N = nrow(fraud),
                   Exact_Duplicates = sum(duplicated(fraud)),
                   Fraud_Duplicates = sum(duplicated(fraud) & fraud$Class == "1"))
save_table(dups, fraud_folder, "fraud__EDA_duplicate_check.csv")
amount_summary <- data.frame(
  Class = c("Legitimate", "Fraud"),
  N = c(sum(fraud$Class == "0"), sum(fraud$Class == "1")),
  Mean_Amount = c(mean(fraud$Amount[fraud$Class == "0"]),
                  mean(fraud$Amount[fraud$Class == "1"])),
  Median_Amount = c(median(fraud$Amount[fraud$Class == "0"]),
                    median(fraud$Amount[fraud$Class == "1"]))
)
save_table(amount_summary, fraud_folder, "fraud__EDA_amount_summary.csv")

p <- ggplot2::ggplot(counts, ggplot2::aes(Class, Count, fill = Class)) +
  ggplot2::geom_col(show.legend = FALSE) + ggplot2::scale_y_log10() +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::labs(title = "Original ULB credit-card transactions by class",
                x = "Class: 0 legitimate, 1 fraud", y = "Count (log scale)")
save_plot(p, fraud_folder, "eda", "fraud_class_counts.png", 7, 4.5)

p <- ggplot2::ggplot(fraud, ggplot2::aes(log1p(Amount), fill = Class)) +
  ggplot2::geom_density(alpha = 0.4) + ggplot2::theme_minimal(base_size = 12) +
  ggplot2::labs(title = "Original transaction amounts by fraud class",
                x = "log(1 + original amount)")
save_plot(p, fraud_folder, "eda", "fraud_amounts.png", 7, 4.5)

# Extra descriptive visuals; no variables are chosen using test outcomes.
fraud$Elapsed_Hour <- floor(fraud$Time / 3600)
hourly <- aggregate(as.integer(as.character(fraud$Class)),
                    by = list(Hour = fraud$Elapsed_Hour),
                    FUN = function(z) c(N = length(z), Fraud = sum(z)))
hourly <- data.frame(Hour = hourly$Hour, N = hourly$x[, "N"],
                     Fraud = hourly$x[, "Fraud"])
hourly$Fraud_Rate <- hourly$Fraud / hourly$N
save_table(hourly, fraud_folder, "fraud__EDA_hourly_prevalence.csv")
p <- ggplot2::ggplot(hourly, ggplot2::aes(Hour, Fraud_Rate * 100)) +
  ggplot2::geom_line() + ggplot2::geom_point(size = 1) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::labs(title = "Descriptive fraud prevalence by elapsed hour",
                x = "Hours since first transaction", y = "Fraud (%)")
save_plot(p, fraud_folder, "eda", "fraud_hourly_prevalence.png", 8, 4.5)
fraud$Elapsed_Hour <- NULL

# 3. Pre-specified transformation and duplicate protection
# Original amounts remain intact in data/raw/creditcard.csv.
fraud$Amount <- log1p(fraud$Amount)
fraud <- fraud[!duplicated(fraud), , drop = FALSE]
rownames(fraud) <- NULL
save_table(data.frame(
  Data = "Original ULB creditcard.csv", Raw_N = 284807L, Raw_Fraud = 492L,
  After_Removing_Duplicates = nrow(fraud),
  After_Removing_Duplicates_Fraud = sum(fraud$Class == "1")
), fraud_folder, "fraud__dataset_summary.csv")
data.table::fwrite(fraud, file.path(fraud_folder, "data", "processed",
                                    "creditcard_model_ready.csv.gz"), compress = "gzip")

# 4. Evaluate both an ordinary stratified split and a later-in-time test
# The chronological TEST is later than training, but CV within training is
# stratified random: it is NOT rolling-origin/forward-chaining validation.
fraud$.id <- seq_len(nrow(fraud))
all_fraud <- list()
all_fraud_boot <- list()
split_info <- list()
for (kind in c("fraud_stratified_random", "fraud_chronological_holdout")) {
  if (kind == "fraud_stratified_random") {
    sp <- split_random(fraud, cfg$seed + 300L, cfg$test_fraction)
  } else {
    ordered <- fraud[order(fraud$Time, seq_len(nrow(fraud))), , drop = FALSE]
    ntrain <- floor((1 - cfg$test_fraction) * nrow(ordered))
    sp <- list(train = ordered[seq_len(ntrain), , drop = FALSE],
               test = ordered[seq.int(ntrain + 1L, nrow(ordered)), , drop = FALSE])
  }
  for (part in c("train", "test")) {
    data.table::fwrite(data.frame(Processed_Row_ID = sp[[part]]$.id),
                       file.path(fraud_folder, "data", "splits",
                                 paste0(kind, "__", part, "_rows.csv.gz")), compress = "gzip")
  }
  split_info[[kind]] <- data.frame(
    Experiment = kind, Train_N = nrow(sp$train), Test_N = nrow(sp$test),
    Train_Positives = sum(sp$train$Class == "1"),
    Test_Positives = sum(sp$test$Class == "1"),
    Train_Prevalence = mean(sp$train$Class == "1"),
    Test_Prevalence = mean(sp$test$Class == "1"),
    Chronological_Boundary = if (kind == "fraud_chronological_holdout")
      max(sp$train$Time) else NA_real_
  )
  sp$train$.id <- NULL
  sp$test$.id <- NULL
  comparison <- compare_models(sp$train, sp$test, kind, fraud_folder,
                               cfg$seed + 400L + length(all_fraud), cfg,
                               target = 492 / 284807, real_data = TRUE)
  all_fraud[[kind]] <- comparison$scores
  all_fraud_boot[[kind]] <- comparison$bootstrap
  save_table(do.call(rbind, all_fraud), fraud_folder, "fraud__all_metrics.csv")
  save_table(do.call(rbind, all_fraud_boot), fraud_folder, "fraud__ensemble_comparisons.csv")
  save_table(do.call(rbind, split_info), fraud_folder, "fraud__split_summary.csv")
}

fraud_results <- do.call(rbind, all_fraud)

# 5. Report a candidate chosen ONLY using out-of-fold training AP/F1.
selected <- subset(fraud_results,
                   Classifier %in% c(cfg$models, "Ensemble") &
                     Operating_Rule == "CV_tuned_F1")
selected <- do.call(rbind, lapply(split(selected, selected$Experiment), function(d) {
  d[order(-d$CV_PR_AUC_AP, -d$CV_F1_Selected), , drop = FALSE][1, , drop = FALSE]
}))
save_table(selected, fraud_folder, "fraud__CV_selected_models.csv")

# 6. Figures from untouched, natural-prevalence held-out observations
fraud_fig <- subset(fraud_results,
                    Classifier %in% cfg$models & Operating_Rule == "default_0.5")
for (metric in c("PR_AUC_AP", "ROC_AUC", "Brier")) {
  df <- fraud_fig
  df$Value <- df[[metric]]
  p <- ggplot2::ggplot(df, ggplot2::aes(Remedy, Value, fill = Classifier)) +
    ggplot2::geom_col(position = "dodge") + ggplot2::coord_flip() +
    ggplot2::facet_wrap(~ Experiment, scales = "free_y") +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::labs(title = paste("Fraud:", metric), y = metric)
  save_plot(p, fraud_folder, "final", paste0("fraud_", metric, ".png"), 11, 6)
}

threshold_df <- subset(fraud_results,
                       Classifier %in% cfg$models & Remedy == "baseline" &
                         Operating_Rule %in% c("default_0.5", "CV_tuned_F1"))
p <- ggplot2::ggplot(threshold_df, ggplot2::aes(Classifier, F1, fill = Operating_Rule)) +
  ggplot2::geom_col(position = "dodge") + ggplot2::facet_wrap(~ Experiment) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::labs(title = "Fraud: default versus training-CV-tuned F1")
save_plot(p, fraud_folder, "final", "fraud_threshold_F1.png", 11, 6)

cm <- do.call(rbind, lapply(seq_len(nrow(selected)), function(i) {
  x <- selected[i, , drop = FALSE]
  data.frame(Experiment = x$Experiment,
             Actual = c("Fraud", "Fraud", "Legitimate", "Legitimate"),
             Predicted = c("Fraud", "Legitimate", "Fraud", "Legitimate"),
             Count = c(x$TP, x$FN, x$FP, x$TN))
}))
save_table(cm, fraud_folder, "fraud__confusion_matrix_cells.csv")
p <- ggplot2::ggplot(cm, ggplot2::aes(Predicted, Actual, fill = Count)) +
  ggplot2::geom_tile(colour = "white") +
  ggplot2::geom_text(ggplot2::aes(label = Count)) +
  ggplot2::facet_wrap(~ Experiment) + ggplot2::theme_minimal(base_size = 12) +
  ggplot2::labs(title = "Fraud: confusion matrices of CV-selected models")
save_plot(p, fraud_folder, "final", "fraud_confusion_matrices.png", 11, 5.5)

pr_real <- subset(fraud_results,
                  Classifier %in% cfg$models & Operating_Rule == "CV_tuned_F1")
p <- ggplot2::ggplot(pr_real, ggplot2::aes(Sensitivity, Precision,
                                            colour = Remedy, shape = Classifier)) +
  ggplot2::geom_point(size = 3) + ggplot2::facet_wrap(~ Experiment) +
  ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::labs(title = "Fraud: precision and recall at training-tuned cutoffs")
save_plot(p, fraud_folder, "final", "fraud_precision_recall_tradeoff.png", 11, 6)

weight_files <- list.files(file.path(fraud_folder, "tables"),
                           pattern = "ensemble_weight_search\\.csv$", full.names = TRUE)
all_weights <- do.call(rbind, lapply(weight_files, data.table::fread))
if (!is.null(all_weights) && nrow(all_weights)) {
  p <- ggplot2::ggplot(all_weights, ggplot2::aes(Logistic_Weight, CV_AP)) +
    ggplot2::geom_line() + ggplot2::geom_point() +
    ggplot2::geom_vline(ggplot2::aes(xintercept = Selected_Weight),
                        linetype = "dashed") +
    ggplot2::facet_wrap(~ Experiment) + ggplot2::theme_minimal(base_size = 12) +
    ggplot2::labs(title = "Fraud: selecting the ensemble mixing weight using OOF AP")
  save_plot(p, fraud_folder, "final", "fraud_ensemble_weights.png", 10, 5)
}

writeLines(capture.output(str(cfg)), file.path(fraud_folder, "metadata", "settings.txt"))
writeLines(capture.output(sessionInfo()), file.path(fraud_folder, "metadata", "R_session_info.txt"))
writeLines(c(
  "Dataset: original ULB credit-card transactions (284807 rows, 492 fraud records)",
  paste("Local source:", normalizePath(csv_file, winslash = "/")),
  "Original dataset copied to data/raw/creditcard.csv.",
  "Amount transformed with log1p and exact duplicates removed before splitting.",
  "Class 1 is fraud. Test sets preserve natural prevalence and are never resampled.",
  "Scaling, SMOTE and other resampling, inner lambda tuning are training-only.",
  "Chronological held-out test is later in time, but training CV is random stratified, not forward-chaining."
), file.path(fraud_folder, "metadata", "data_and_cv_notes.txt"))
cat("\nFraud study finished. Results:", fraud_folder, "\n")

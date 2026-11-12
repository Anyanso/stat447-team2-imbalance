# STAT 847 Team 2 - run both studies and make combined comparison files
# IMPORTANT: Open your RStudio project and run this script from its top folder.
# Use source("03_run_all.R") to run the complete analysis.

source("01_simulation_analysis.R", local = FALSE)
source("02_fraud_real_data_analysis.R", local = FALSE)

combined <- file.path(getwd(), "results", "simulation", "combined_results")
make_folders(combined, c("tables", "plots", "metadata"))

# Compare the same operating rule across studies. We do not pool their labels
# or treat different test prevalences as directly interchangeable.
needed_cols <- c(
  "Experiment", "Classifier", "Remedy", "Operating_Rule", "Train_N",
  "Test_N", "Prevalence", "Sensitivity", "Specificity", "Precision", "F1",
  "Balanced_Accuracy", "ROC_AUC", "PR_AUC_AP"
)
sim_final <- subset(sim_results, Operating_Rule == "default_0.5")[, needed_cols]
fraud_final <- subset(fraud_results, Operating_Rule == "default_0.5")[, needed_cols]
sim_final$Study <- "Simulation"
fraud_final$Study <- "Fraud"
combined_scores <- rbind(sim_final, fraud_final)
save_table(combined_scores, combined, "both_studies_default_threshold.csv")
save_table(averages, combined, "simulation_average_metrics.csv")
save_table(fraud_final, combined, "fraud_holdout_metrics.csv")

# A final display of AP by study. Facets emphasize the different settings.
combined_plot <- subset(combined_scores, Classifier %in% cfg$models)
combined_plot$Method <- paste(combined_plot$Classifier, combined_plot$Remedy, sep = " / ")
p <- ggplot2::ggplot(combined_plot,
                     ggplot2::aes(Prevalence * 100, PR_AUC_AP, colour = Method)) +
  ggplot2::geom_point(alpha = .65) +
  ggplot2::facet_wrap(~ Study, scales = "free_x") +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::labs(title = "Held-out AP: simulation and fraud analyses",
                subtitle = "Separate study panels; scores are not pooled",
                x = "Observed positive prevalence (%)", y = "Average precision")
ggplot2::ggsave(file.path(combined, "plots", "simulation_vs_fraud_AP.png"),
                p, width = 12, height = 6.5, dpi = 200, bg = "white")
writeLines(capture.output(sessionInfo()), file.path(combined, "metadata", "R_session_info.txt"))
cat("\nAll studies completed. Main results directory:", combined, "\n")

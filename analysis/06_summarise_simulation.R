# 06_summarise_simulation.R
# Summary tables and Figures 2-4 from results/sim_results.rds.
source("R/00_setup.R")
source("R/summarise.R")
results <- readRDS(file.path(RESULTS_DIR, "sim_results.rds"))

# make sure every replicate is there
reps <- unique(results[, c("setting", "rep.id")])
n.reps.total <- length(PREVALENCES) * length(SAMPLE_SIZES) * length(SEPARATION) * N_REPS
stopifnot(nrow(results) == n.reps.total * 13, nrow(reps) == n.reps.total,
          all(table(reps$setting) == N_REPS))
cat("Results complete:", n.reps.total, "replicates x 13 methods\n\n")

# mean, sd and MCSE of every metric
summ <- summarise_metrics(results)
saveRDS(summ, file.path(RESULTS_DIR, "sim_summary.rds"))
cat("Largest MCSE of any mean:", round(max(summ$mcse, na.rm = TRUE), 4), "\n\n")

# paired change from the baseline
paired <- rbind(paired_change(results, "roc.auc"),
                paired_change(results, "pr.auc"),
                paired_change(results, "sensitivity"),
                paired_change(results, "bal.accuracy"))
saveRDS(paired, file.path(RESULTS_DIR, "sim_paired.rds"))

# table for the report: change in ROC-AUC and in sensitivity,
# pooled over the two separations
pooled <- rbind(paired_change(results, "roc.auc", pool.sep = TRUE),
                paired_change(results, "sensitivity", pool.sep = TRUE))
tab <- pooled |>
  mutate(cell = sprintf("%+.3f (%.3f)", mean, mcse)) |>
  select(classifier, approach, n, prev, metric, cell) |>
  pivot_wider(names_from = prev, values_from = cell, names_prefix = "prev ") |>
  arrange(metric, classifier, n, approach)
write.csv(tab, file.path(RESULTS_DIR, "table_change_vs_baseline.csv"), row.names = FALSE)
cat("Change from baseline, mean (MCSE), pooled over separation:\n")
print(as.data.frame(tab), row.names = FALSE)

# Figure 2: change in ROC-AUC
fig2 <- plot_change(
  paired_change(results, "roc.auc", pool.sep = TRUE),
  ylab = "Change in ROC-AUC vs baseline",
  title = "Ranking barely changes, except the forest with under-sampling",
  subtitle = "Paired difference from the baseline on the same data sets; bars = +/- 2 MCSE; 200 replicates per point")
ggsave(file.path(FIGURES_DIR, "fig2_change_roc_auc.png"), fig2,
       width = 11, height = 6, dpi = 200, bg = "white")

# Figure 3: change in sensitivity
fig3 <- plot_change(
  paired_change(results, "sensitivity", pool.sep = TRUE),
  ylab = "Change in sensitivity vs baseline",
  title = "Sensitivity jumps for almost every approach",
  subtitle = "Paired difference from the baseline on the same data sets; bars = +/- 2 MCSE; 200 replicates per point")
ggsave(file.path(FIGURES_DIR, "fig3_change_sensitivity.png"), fig3,
       width = 11, height = 6, dpi = 200, bg = "white")

# Figure 4: PR-AUC gap to the Bayes benchmark
gap <- bayes_gap(results, "pr.auc") |>
  group_by(prev, n, classifier, approach) |>
  summarise(mean = mean(mean), mcse = sqrt(sum(mcse^2)) / n(), .groups = "drop")
gap$approach <- factor(APPROACH_LABELS[gap$approach],
                       levels = APPROACH_LABELS[c("baseline", "oversample", "undersample",
                                                  "smote", "weights", "threshold")])
gap$prev.lab <- factor(paste0(100 * gap$prev, "%"), levels = c("50%", "20%", "5%", "1%"))
gap$panel.col <- paste0("n = ", format(gap$n, big.mark = ","))
gap$panel.row <- ifelse(gap$classifier == "logistic", "Penalized logistic", "Random forest")
fig4 <- ggplot(gap, aes(prev.lab, mean, colour = approach, shape = approach, group = approach)) +
  geom_hline(yintercept = 0, colour = "grey45", linewidth = 0.4) +
  geom_line(linewidth = 0.6, position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = mean - 2 * mcse, ymax = mean + 2 * mcse), width = 0,
                linewidth = 0.5, position = position_dodge(width = 0.5)) +
  geom_point(size = 2.2, position = position_dodge(width = 0.5)) +
  facet_grid(panel.row ~ panel.col) +
  scale_colour_manual(values = c("#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")) +
  scale_shape_manual(values = c(1, 16, 17, 15, 18, 4)) +
  labs(x = "Minority prevalence", y = "PR-AUC gap to Bayes (lower is better)",
       colour = NULL, shape = NULL,
       title = "No approach closes the gap to the best possible ranking",
       subtitle = "Bayes PR-AUC minus method PR-AUC; bars = +/- 2 MCSE; 200 replicates per point") +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold"))
ggsave(file.path(FIGURES_DIR, "fig4_gap_pr_auc.png"), fig4,
       width = 11, height = 6, dpi = 200, bg = "white")

cat("\nSaved tables to results/ and figures 2-4 to figures/\n")

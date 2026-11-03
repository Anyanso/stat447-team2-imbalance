# =============================================================
# 09b_summarise_ensemble.R  --  Step 9 (STAT 847): does EasyEnsemble beat
# the best single approach, and at which imbalance ratios?
# Input : results/sim_results.rds, results/ens_sim_results.rds (simulation)
#         results/real_results.rds, results/ens_real_results.rds (real data)
# Output: results/table_ensemble_sim.csv, results/table_ensemble_real.csv,
#         figures/fig7_ensemble_vs_best.png
# =============================================================
source("R/00_setup.R")
source("R/summarise.R")
sim <- readRDS(file.path(RESULTS_DIR, "sim_results.rds"))
ens <- readRDS(file.path(RESULTS_DIR, "ens_sim_results.rds"))
cat("Ensemble replicates available:", nrow(unique(ens[, c("setting", "rep.id")])), "of 1600\n\n")

# ---- Block A: who is the "best single approach"? ----
# For each setting x classifier, the approach (of the 6) with the highest
# MEAN of the metric over its 100 replicates. Note this choice is made
# AFTER seeing the results, which flatters the single approaches a little --
# so it is a tough benchmark for the ensemble.
best_single <- function(metric) {
  m <- sim[sim$approach != "bayes", ] |>
    group_by(prev, n, sep.level, classifier, approach) |>
    summarise(value = mean(.data[[metric]], na.rm = TRUE), .groups = "drop") |>
    group_by(prev, n, sep.level, classifier) |>
    slice_max(value, n = 1, with_ties = FALSE) |>
    ungroup()
  m[, c("prev", "n", "sep.level", "classifier", "approach")]
}

# ---- Block B: PAIRED difference, ensemble minus best single approach ----
# Same replicate, same data -> replicate-by-replicate difference, then mean
# and MCSE (deck S.6). Positive = the ensemble wins.
vs_best <- function(metric) {
  best <- best_single(metric)
  b <- merge(sim, best, by = c("prev", "n", "sep.level", "classifier", "approach"))
  b <- b[, c("setting", "rep.id", "classifier", "approach", metric)]
  names(b)[4:5] <- c("best.approach", "best.value")
  d <- merge(ens[, c("prev", "n", "sep.level", "setting", "rep.id", "classifier", metric)],
             b, by = c("setting", "rep.id", "classifier"))
  d$diff <- d[[metric]] - d$best.value
  d |>
    group_by(prev, n, sep.level, classifier, best.approach) |>
    summarise(mean = mean(diff, na.rm = TRUE),
              mcse = sd(diff, na.rm = TRUE) / sqrt(sum(!is.na(diff))),
              B = n(), .groups = "drop") |>
    mutate(metric = metric,
           # A difference must be both CLEAR (beyond 2 MCSE, deck Mistake 2)
           # and LARGE ENOUGH TO MATTER (at least 0.005) to count.
           verdict = ifelse(abs(mean) < 0.005, "negligible (< 0.005)",
                     ifelse(mean - 2 * mcse > 0, "ensemble better",
                     ifelse(mean + 2 * mcse < 0, "best single better", "no clear difference"))))
}
res <- rbind(vs_best("roc.auc"), vs_best("pr.auc"), vs_best("bal.accuracy"))
out <- res |>
  mutate(change = sprintf("%+.3f (%.3f)", mean, mcse)) |>
  select(metric, classifier, n, sep.level, prev, best.approach, change, verdict) |>
  arrange(metric, classifier, n, sep.level, desc(prev))
write.csv(out, file.path(RESULTS_DIR, "table_ensemble_sim.csv"), row.names = FALSE)

cat("How often the ensemble beats / ties / loses to the best single approach\n")
cat("(16 settings x 2 classifiers per metric; counted only if beyond 2 MCSE and >= 0.005)\n")
print(table(res$metric, res$verdict))

# ---- Block C: Figure 7 -- ensemble minus best single, by prevalence ----
f7 <- res |>
  group_by(prev, n, classifier, metric) |>
  summarise(mean = mean(mean), mcse = sqrt(sum(mcse^2)) / n(), .groups = "drop")
f7$metric <- factor(c(roc.auc = "ROC-AUC", pr.auc = "PR-AUC",
                      bal.accuracy = "Balanced accuracy")[f7$metric],
                    levels = c("ROC-AUC", "PR-AUC", "Balanced accuracy"))
f7$prev.lab <- factor(paste0(100 * f7$prev, "%"), levels = c("50%", "20%", "5%", "1%"))
f7$classifier <- ifelse(f7$classifier == "logistic", "Penalized logistic", "Random forest")
f7$n.lab <- paste0("n = ", format(f7$n, big.mark = ","))
fig7 <- ggplot(f7, aes(prev.lab, mean, colour = n.lab, shape = n.lab, group = n.lab)) +
  geom_hline(yintercept = 0, colour = "grey45", linewidth = 0.4) +
  geom_line(linewidth = 0.6, position = position_dodge(width = 0.3)) +
  geom_errorbar(aes(ymin = mean - 2 * mcse, ymax = mean + 2 * mcse), width = 0,
                linewidth = 0.5, position = position_dodge(width = 0.3)) +
  geom_point(size = 2.2, position = position_dodge(width = 0.3)) +
  facet_grid(classifier ~ metric) +
  scale_colour_manual(values = c("#2a78d6", "#eb6834")) +
  scale_shape_manual(values = c(16, 17)) +
  labs(x = "Minority prevalence", y = "EasyEnsemble minus best single approach",
       colour = NULL, shape = NULL,
       title = "EasyEnsemble vs the best single approach in each setting",
       subtitle = "Above 0 = ensemble better; paired on the same data sets; bars = +/- 2 MCSE; pooled over separation") +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold"))
ggsave(file.path(FIGURES_DIR, "fig7_ensemble_vs_best.png"), fig7,
       width = 10, height = 6, dpi = 200, bg = "white")

# ---- Block D: real data, the same comparison over the 10 splits ----
real <- readRDS(file.path(RESULTS_DIR, "real_results.rds"))
ens.real <- readRDS(file.path(RESULTS_DIR, "ens_real_results.rds"))
real.out <- NULL
for (metric in c("roc.auc", "pr.auc", "bal.accuracy")) {
  for (cl in CLASSIFIERS) {
    r <- real[real$classifier == cl, ]
    means <- tapply(r[[metric]], r$approach, mean)
    best <- names(which.max(means))
    m <- merge(ens.real[ens.real$classifier == cl, c("split.id", metric)],
               r[r$approach == best, c("split.id", metric)], by = "split.id")
    d <- m[[2]] - m[[3]]
    real.out <- rbind(real.out, data.frame(
      metric = metric, classifier = cl, best.approach = best,
      ensemble = round(mean(m[[2]]), 3), best.single = round(mean(m[[3]]), 3),
      change = sprintf("%+.3f (%.3f)", mean(d), sd(d) / sqrt(length(d)))))
  }
}
write.csv(real.out, file.path(RESULTS_DIR, "table_ensemble_real.csv"), row.names = FALSE)
cat("\nReal data: ensemble vs best single approach (mean over 10 splits; SE)\n")
print(real.out, row.names = FALSE)

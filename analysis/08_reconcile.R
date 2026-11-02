# =============================================================
# 08_reconcile.R  --  Step 8: real-data summaries, and simulation vs real data
# Input : results/real_results.rds (Step 7), results/sim_results.rds (Step 5)
# Output: results/table_real.csv, results/table_reconcile.csv,
#         figures/fig5_real_change.png, figures/fig6_real_tradeoff.png
# =============================================================
source("R/00_setup.R")
source("R/summarise.R")
real <- readRDS(file.path(RESULTS_DIR, "real_results.rds"))
sim  <- readRDS(file.path(RESULTS_DIR, "sim_results.rds"))
stopifnot(nrow(real) == N_SPLITS * 13)

# ---- Block A: real-data averages over the 10 splits (mean and SE) ----
# With real data the "replicates" are the 10 random train/test splits.
methods <- real[real$classifier != "none", ]
tab.real <- methods |>
  group_by(classifier, approach) |>
  summarise(across(c(roc.auc, pr.auc, sensitivity, precision, f1, bal.accuracy, accuracy),
                   list(mean = ~ mean(.x, na.rm = TRUE),
                        se   = ~ sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x))))),
            .groups = "drop")
write.csv(tab.real, file.path(RESULTS_DIR, "table_real.csv"), row.names = FALSE)
null.row <- real[real$classifier == "none", ]
cat("'Always not fraud': accuracy", round(mean(null.row$accuracy), 4),
    "| sensitivity", mean(null.row$sensitivity), "\n\n")

# ---- Block B: PAIRED change from the baseline, split by split ----
# Same idea as Step 6: each approach and the baseline used the same split.
base <- methods[methods$approach == "baseline", c("split.id", "classifier", "roc.auc", "pr.auc")]
names(base)[3:4] <- c("b.roc", "b.pr")
d <- merge(methods[methods$approach != "baseline", ], base, by = c("split.id", "classifier"))
chg.real <- d |>
  group_by(classifier, approach) |>
  summarise(roc.mean = mean(roc.auc - b.roc), roc.se = sd(roc.auc - b.roc) / sqrt(n()),
            pr.mean  = mean(pr.auc - b.pr),   pr.se  = sd(pr.auc - b.pr) / sqrt(n()),
            .groups = "drop")

# ---- Block C: the matching simulation numbers ----
# Closest simulated setting: lowest prevalence (1%), largest n (20,000),
# high separation (real frauds are easy to rank: ROC-AUC about 0.95-0.98).
sim.match <- sim[sim$prev == 0.01 & sim$n == 20000 & sim$sep.level == "high", ]
sim.chg <- rbind(paired_change(sim.match, "roc.auc"), paired_change(sim.match, "pr.auc"))
sim.chg <- sim.chg |>
  select(classifier, approach, metric, mean, mcse) |>
  pivot_wider(names_from = metric, values_from = c(mean, mcse))

recon <- merge(chg.real, sim.chg, by = c("classifier", "approach"))
recon.out <- data.frame(
  classifier = recon$classifier,
  approach   = APPROACH_LABELS[recon$approach],
  `ROC-AUC change, simulation` = sprintf("%+.3f (%.3f)", recon$mean_roc.auc, recon$mcse_roc.auc),
  `ROC-AUC change, real`       = sprintf("%+.3f (%.3f)", recon$roc.mean, recon$roc.se),
  `PR-AUC change, simulation`  = sprintf("%+.3f (%.3f)", recon$mean_pr.auc, recon$mcse_pr.auc),
  `PR-AUC change, real`        = sprintf("%+.3f (%.3f)", recon$pr.mean, recon$pr.se),
  check.names = FALSE)
recon.out <- recon.out[order(recon.out$classifier, recon.out$approach), ]
write.csv(recon.out, file.path(RESULTS_DIR, "table_reconcile.csv"), row.names = FALSE)
cat("Change from baseline, mean (MCSE or SE): simulation (1%, n = 20,000, high sep.) vs real data\n")
print(recon.out, row.names = FALSE)

# ---- Block D: Figure 5 -- real data, change in ROC-AUC and PR-AUC ----
f5 <- rbind(data.frame(classifier = chg.real$classifier, approach = chg.real$approach,
                       metric = "ROC-AUC", mean = chg.real$roc.mean, se = chg.real$roc.se),
            data.frame(classifier = chg.real$classifier, approach = chg.real$approach,
                       metric = "PR-AUC", mean = chg.real$pr.mean, se = chg.real$pr.se))
f5$approach <- factor(APPROACH_LABELS[f5$approach],
                      levels = rev(APPROACH_LABELS[c("oversample", "undersample", "smote",
                                                     "weights", "threshold")]))
f5$metric <- factor(f5$metric, levels = c("ROC-AUC", "PR-AUC"))
f5$classifier <- ifelse(f5$classifier == "logistic", "Penalized logistic", "Random forest")
fig5 <- ggplot(f5, aes(mean, approach)) +
  geom_vline(xintercept = 0, colour = "grey45", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = mean - 2 * se, xmax = mean + 2 * se),
                 height = 0, linewidth = 0.6, colour = "#2a78d6") +
  geom_point(size = 2.4, colour = "#2a78d6") +
  facet_grid(classifier ~ metric) +
  labs(x = "Change vs baseline (same split)", y = NULL,
       title = "Real fraud data: resampling can lift ROC-AUC but rarely PR-AUC",
       subtitle = "Mean paired difference over 10 stratified 70/30 splits; bars = +/- 2 SE") +
  theme(panel.grid.minor = element_blank(), strip.text = element_text(face = "bold"))
ggsave(file.path(FIGURES_DIR, "fig5_real_change.png"), fig5,
       width = 9, height = 5, dpi = 200, bg = "white")

# ---- Block E: Figure 6 -- the sensitivity / precision trade-off ----
# Each point = one method at the cutoff it used. Up and right is better.
f6 <- methods |>
  group_by(classifier, approach) |>
  summarise(sensitivity = mean(sensitivity), precision = mean(precision, na.rm = TRUE),
            .groups = "drop")
f6$approach <- factor(APPROACH_LABELS[f6$approach],
                      levels = APPROACH_LABELS[c("baseline", "oversample", "undersample",
                                                 "smote", "weights", "threshold")])
f6$classifier <- ifelse(f6$classifier == "logistic", "Penalized logistic", "Random forest")
fig6 <- ggplot(f6, aes(sensitivity, precision, colour = classifier, shape = classifier)) +
  geom_point(size = 3) +
  # The five logistic approaches and forest under-sampling all land near
  # (90%, 6%); label that cluster once instead of six overlapping labels.
  geom_text(data = f6[!(f6$classifier == "Penalized logistic" & f6$approach != "Baseline") &
                        !(f6$classifier == "Random forest" & f6$approach == "Under-sampling"), ],
            aes(label = approach), nudge_y = 0.035, size = 3, show.legend = FALSE) +
  annotate("segment", x = 0.80, xend = 0.885, y = 0.22, yend = 0.085,
           colour = "grey45", linewidth = 0.3) +
  annotate("text", x = 0.80, y = 0.27, size = 3, colour = "grey25",
           label = "All five logistic approaches\n+ forest under-sampling") +
  scale_colour_manual(values = c("#2a78d6", "#eb6834")) +
  scale_x_continuous(limits = c(0.4, 1), labels = scales::percent) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(x = "Sensitivity (share of frauds caught)",
       y = "Precision (share of flags that are fraud)", colour = NULL, shape = NULL,
       title = "Catching more fraud costs precision -- except forest + SMOTE",
       subtitle = "Real data, mean over 10 splits, each method at its own cutoff") +
  theme(legend.position = "top", panel.grid.minor = element_blank())
ggsave(file.path(FIGURES_DIR, "fig6_real_tradeoff.png"), fig6,
       width = 8, height = 6, dpi = 200, bg = "white")
cat("\nSaved tables and figures 5-6\n")

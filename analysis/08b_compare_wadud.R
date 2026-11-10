# 08b_compare_wadud.R
# Robustness check: compares our simulation with Wadud's separate
# implementation (results/simulation/tables/). Makes Figure 8 for the
# report and Figure 8b for the slides.
source("R/00_setup.R")
his  <- read.csv("results/simulation/tables/simulation__all_metrics.csv")
ours <- readRDS(file.path(RESULTS_DIR, "sim_results.rds"))

# Wadud's study: change from its own baseline on the same data set
h <- his[his$Operating_Rule == "default_0.5" &
         his$Classifier %in% c("Penalized_Logistic", "Random_Forest"), ]
h$classifier <- ifelse(h$Classifier == "Penalized_Logistic", "logistic", "forest")
h$approach <- c(baseline = "baseline", oversample = "oversample", undersample = "undersample",
                smote = "smote", weighted = "weights")[h$Remedy]
h$n <- c(`1600` = 1000, `16000` = 20000, `80000` = 1e5)[as.character(h$Train_N)]  # nearest of ours
hb <- h[h$approach == "baseline", c("Experiment", "classifier", "ROC_AUC", "PR_AUC_AP", "Sensitivity")]
names(hb)[3:5] <- c("b.roc", "b.pr", "b.sens")
hd <- merge(h[h$approach != "baseline", ], hb, by = c("Experiment", "classifier"))
his.chg <- hd |>
  mutate(d.roc = ROC_AUC - b.roc, d.pr = PR_AUC_AP - b.pr, d.sens = Sensitivity - b.sens) |>
  group_by(prev = Target_Prevalence, n, classifier, approach) |>
  summarise(across(c(d.roc, d.pr, d.sens), list(m = mean, se = ~ sd(.x) / sqrt(n()))),
            .groups = "drop") |>
  mutate(study = "Second implementation (3 reps)")

# ours, high separation (closest in difficulty)
o <- ours[ours$sep.level == "high" &
          ours$approach %in% c("baseline", "oversample", "undersample", "smote", "weights"), ]
ob <- o[o$approach == "baseline", c("setting", "rep.id", "classifier", "roc.auc", "pr.auc", "sensitivity")]
names(ob)[4:6] <- c("b.roc", "b.pr", "b.sens")
od <- merge(o[o$approach != "baseline", ], ob, by = c("setting", "rep.id", "classifier"))
our.chg <- od |>
  mutate(d.roc = roc.auc - b.roc, d.pr = pr.auc - b.pr, d.sens = sensitivity - b.sens) |>
  group_by(prev, n, classifier, approach) |>
  summarise(across(c(d.roc, d.pr, d.sens), list(m = mean, se = ~ sd(.x) / sqrt(n()))),
            .groups = "drop") |>
  mutate(study = "Main study (100 reps)")

# agreement at n = 20,000, where both have enough test positives
both <- bind_rows(his.chg, our.chg) |> filter(n == 20000)
w <- both |>
  select(study, prev, classifier, approach, d.roc_m, d.pr_m, d.sens_m) |>
  pivot_wider(names_from = study, values_from = c(d.roc_m, d.pr_m, d.sens_m))
names(w) <- c("prev", "classifier", "approach",
              "roc.second", "roc.main", "pr.second", "pr.main", "sens.second", "sens.main")
write.csv(w, file.path(RESULTS_DIR, "table_robustness_agreement.csv"), row.names = FALSE)
sgn <- function(x) ifelse(abs(x) < 0.005, 0, sign(x))       # < 0.005 = "no change"
cat("Same direction (|change| < 0.005 = no change):",
    "ROC-AUC", round(mean(sgn(w$roc.second) == sgn(w$roc.main)), 2),
    "| PR-AUC", round(mean(sgn(w$pr.second) == sgn(w$pr.main)), 2), "\n")

# Figure 8 (report): change vs baseline in both studies
lab <- c(oversample = "Over-sampling", undersample = "Under-sampling",
         smote = "SMOTE", weights = "Class weights")
f <- both
f$approach <- factor(lab[f$approach], levels = lab)
f$classifier <- ifelse(f$classifier == "logistic", "Penalized logistic", "Random forest")
f$prev.lab <- factor(paste0(100 * f$prev, "%"), levels = c("50%", "20%", "5%", "1%"))
long <- bind_rows(
  f |> transmute(study, prev.lab, classifier, approach, metric = "ROC-AUC", m = d.roc_m, se = d.roc_se),
  f |> transmute(study, prev.lab, classifier, approach, metric = "PR-AUC", m = d.pr_m, se = d.pr_se))
long$metric <- factor(long$metric, levels = c("ROC-AUC", "PR-AUC"))
cols <- c("Main study (100 reps)" = "#2a78d6", "Second implementation (3 reps)" = "#eb6834")
fig8 <- ggplot(long, aes(prev.lab, m, colour = study, shape = study, group = study)) +
  geom_hline(yintercept = 0, colour = "grey45", linewidth = 0.4) +
  geom_line(linewidth = 0.5, position = position_dodge(width = 0.4)) +
  geom_errorbar(aes(ymin = m - 2 * se, ymax = m + 2 * se), width = 0, linewidth = 0.5,
                position = position_dodge(width = 0.4)) +
  geom_point(size = 2, position = position_dodge(width = 0.4)) +
  facet_grid(metric + classifier ~ approach, scales = "free_y") +
  scale_colour_manual(values = cols) + scale_shape_manual(values = c(16, 17)) +
  labs(x = "Minority prevalence", y = "Change vs baseline (same data set)",
       colour = NULL, shape = NULL,
       title = "Two independent simulations: resampling moves sensitivity, not ranking",
       subtitle = "n = 20,000; default 0.5 cutoff; bars = +/- 2 SE (second implementation: 3 replicates)") +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold"))
ggsave(file.path(FIGURES_DIR, "fig8_partner_comparison.png"), fig8,
       width = 11, height = 9, dpi = 170, bg = "white")

# Figure 8b (slides): our change against Wadud's, one dot per comparison;
# dots on the diagonal mean the two studies agree
sc <- bind_rows(
  w |> transmute(classifier, approach, metric = "ROC-AUC", main = roc.main, second = roc.second),
  w |> transmute(classifier, approach, metric = "PR-AUC", main = pr.main, second = pr.second))
sc$metric <- factor(sc$metric, levels = c("ROC-AUC", "PR-AUC"))
sc$classifier <- ifelse(sc$classifier == "logistic", "Penalized logistic", "Random forest")
lim <- range(c(sc$main, sc$second)) * 1.08
fig8b <- ggplot(sc, aes(main, second, colour = classifier, shape = classifier)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey55", linetype = "dashed") +
  geom_hline(yintercept = 0, colour = "grey80") + geom_vline(xintercept = 0, colour = "grey80") +
  geom_point(size = 3.2, alpha = 0.85) +
  facet_wrap(~ metric) + coord_equal(xlim = lim, ylim = lim) +
  scale_colour_manual(values = c("#2a78d6", "#eb6834")) +
  labs(x = "Change vs baseline, main study", y = "Change vs baseline, second implementation",
       colour = NULL, shape = NULL,
       title = "Do the two studies agree?",
       subtitle = "Each dot = one remedy x prevalence x classifier at n = 20,000; dashed line = perfect agreement") +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold"))
ggsave(file.path(FIGURES_DIR, "fig8b_agreement.png"), fig8b,
       width = 10, height = 5.6, dpi = 170, bg = "white")
cat("Saved figures 8 and 8b\n")

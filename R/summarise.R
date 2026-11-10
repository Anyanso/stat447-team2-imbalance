# summarise.R -- summaries and plot helper for the simulation (John Anyanso)
# Every average is reported with its Monte Carlo standard error,
# MCSE = sd / sqrt(B).

METRICS <- c("sensitivity", "specificity", "precision", "f1",
             "bal.accuracy", "accuracy", "roc.auc", "pr.auc")

# labels and order used in all tables and figures
APPROACH_LABELS <- c(baseline = "Baseline", oversample = "Over-sampling",
                     undersample = "Under-sampling", smote = "SMOTE",
                     weights = "Class weights", threshold = "Threshold",
                     bayes = "Bayes (truth)")

# Mean, sd and MCSE of each metric by setting x classifier x approach.
# Precision is undefined when nothing is flagged; those replicates are
# dropped from the mean and counted in n.flag.none.
summarise_metrics <- function(results) {
  long <- pivot_longer(results, all_of(METRICS),
                       names_to = "metric", values_to = "value")
  summ <- long |>
    group_by(prev, n, sep.level, classifier, approach, metric) |>
    summarise(mean = mean(value, na.rm = TRUE),
              sd   = sd(value, na.rm = TRUE),
              B    = sum(!is.na(value)),
              n.flag.none = sum(is.na(value)),
              .groups = "drop")
  summ$mcse <- summ$sd / sqrt(summ$B)
  summ
}

# Change from the baseline, paired by replicate. All approaches ran on the
# same data sets as the baseline, so the differences are much less noisy
# than comparing two averages.
paired_change <- function(results, metric, pool.sep = FALSE) {
  base <- results[results$approach == "baseline",
                  c("setting", "rep.id", "classifier", metric)]
  names(base)[4] <- "base.value"
  d <- merge(results[!results$approach %in% c("baseline", "bayes"), ], base,
             by = c("setting", "rep.id", "classifier"))
  d$change <- d[[metric]] - d$base.value

  groups <- c("prev", "n", "classifier", "approach",
              if (!pool.sep) "sep.level")
  d |>
    group_by(across(all_of(groups))) |>
    summarise(mean = mean(change, na.rm = TRUE),
              mcse = sd(change, na.rm = TRUE) / sqrt(sum(!is.na(change))),
              B    = sum(!is.na(change)),
              .groups = "drop") |>
    mutate(metric = metric)
}

# Gap to the Bayes benchmark: Bayes value minus method value, per replicate.
bayes_gap <- function(results, metric) {
  bay <- results[results$approach == "bayes", c("setting", "rep.id", metric)]
  names(bay)[3] <- "bayes.value"
  d <- merge(results[results$approach != "bayes", ], bay,
             by = c("setting", "rep.id"))
  d$gap <- d$bayes.value - d[[metric]]
  d |>
    group_by(prev, n, sep.level, classifier, approach) |>
    summarise(mean = mean(gap), mcse = sd(gap) / sqrt(n()), .groups = "drop") |>
    mutate(metric = metric)
}

# Change from baseline against prevalence, with +/- 2 MCSE bars.
# One panel per classifier and sample size.
plot_change <- function(chg, ylab, title, subtitle) {
  chg$approach <- factor(APPROACH_LABELS[chg$approach],
                         levels = APPROACH_LABELS[c("oversample", "undersample",
                                                    "smote", "weights", "threshold")])
  chg$prev.lab <- factor(paste0(100 * chg$prev, "%"),
                         levels = paste0(100 * sort(unique(chg$prev), decreasing = TRUE), "%"))
  chg$panel.col <- paste0("n = ", format(chg$n, big.mark = ","))
  chg$panel.row <- ifelse(chg$classifier == "logistic",
                          "Penalized logistic", "Random forest")
  ggplot(chg, aes(prev.lab, mean, colour = approach, shape = approach,
                  group = approach)) +
    geom_hline(yintercept = 0, colour = "grey45", linewidth = 0.4) +
    geom_line(linewidth = 0.6, position = position_dodge(width = 0.5)) +
    geom_errorbar(aes(ymin = mean - 2 * mcse, ymax = mean + 2 * mcse),
                  width = 0, linewidth = 0.5, position = position_dodge(width = 0.5)) +
    geom_point(size = 2.2, position = position_dodge(width = 0.5)) +
    facet_grid(panel.row ~ panel.col) +
    scale_colour_manual(values = c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")) +
    scale_shape_manual(values = c(16, 17, 15, 18, 4)) +
    labs(x = "Minority prevalence", y = ylab, colour = NULL, shape = NULL,
         title = title, subtitle = subtitle) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold"))
}

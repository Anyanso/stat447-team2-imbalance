# 01_check_gen_data.R
# Checks that gen_data() produces what we asked for, saves the beta0
# table and makes Figure 1.
source("R/00_setup.R")
set.seed(SEED)

# intercepts: one per prevalence x separation
beta0.tab <- make_beta0_table()
print(beta0.tab)
saveRDS(beta0.tab, file.path(RESULTS_DIR, "beta0_table.rds"))

# observed vs target prevalence, on a big sample so the check isn't noisy
beta0.tab$observed <- NA
for (i in 1:nrow(beta0.tab)) {
  d <- gen_data(200000, beta0.tab$prev[i], beta0.tab$sep[i], beta0.tab$beta0[i])
  beta0.tab$observed[i] <- mean(d$y == "1")
}
cat("\nTarget vs. observed prevalence (%):\n")
print(data.frame(sep = beta0.tab$sep.level,
                 target = 100 * beta0.tab$prev,
                 observed = round(100 * beta0.tab$observed, 2)))

# glm on a huge sample should give back sep * beta at every prevalence
# (only the intercept should move)
high <- beta0.tab[beta0.tab$sep.level == "high", ]
slopes <- matrix(NA, nrow(high), 6,
                 dimnames = list(paste0("prev = ", high$prev),
                                 c("(Intercept)", paste0("x", 1:5))))
for (i in 1:nrow(high)) {
  d <- gen_data(200000, high$prev[i], high$sep[i], high$beta0[i])
  glm.fit <- glm(y ~ x1 + x2 + x3 + x4 + x5, data = d, family = binomial)
  slopes[i, ] <- coef(glm.fit)
}
cat("\nTrue slopes (high separation):", SEPARATION["high"] * beta, "\n")
print(round(slopes, 2))

# expected number of positives in a training set
counts <- outer(PREVALENCES, SAMPLE_SIZES)
dimnames(counts) <- list(paste0("prev = ", PREVALENCES),
                         paste0("n = ", SAMPLE_SIZES))
cat("\nExpected number of minority cases in a training set:\n")
print(counts)

# ROC-AUC of the true probabilities = best possible ranking in each setting
beta0.tab$bayes.auc <- NA
for (i in 1:nrow(beta0.tab)) {
  d <- gen_data(200000, beta0.tab$prev[i], beta0.tab$sep[i], beta0.tab$beta0[i])
  pred <- prediction(d$p.true, d$y)
  beta0.tab$bayes.auc[i] <- performance(pred, "auc")@y.values[[1]]
}
cat("\nBayes ROC-AUC (best possible) by setting:\n")
print(round(xtabs(bayes.auc ~ sep.level + prev, data = beta0.tab), 3))

# same seed, same data
set.seed(SEED); a <- gen_data(50, 0.05, 1.5, beta0.tab$beta0[6])
set.seed(SEED); b <- gen_data(50, 0.05, 1.5, beta0.tab$beta0[6])
cat("\nSame seed gives identical data:", identical(a, b), "\n")

# Figure 1: same slopes in every panel, only beta0 changes
plot.df <- NULL
for (i in 1:nrow(beta0.tab)) {
  d <- gen_data(3000, beta0.tab$prev[i], beta0.tab$sep[i], beta0.tab$beta0[i])
  d$prev <- beta0.tab$prev[i]
  d$sep.level <- beta0.tab$sep.level[i]
  plot.df <- rbind(plot.df, d)
}
plot.df$prev.lab <- factor(paste0(100 * plot.df$prev, "% minority"),
                           levels = paste0(100 * PREVALENCES, "% minority"))
plot.df$sep.lab <- paste(plot.df$sep.level, "separation")
plot.df$class <- ifelse(plot.df$y == "1", "Minority (y = 1)", "Majority (y = 0)")
plot.df <- plot.df[order(plot.df$class), ]   # minority points drawn on top

fig <- ggplot(plot.df, aes(x1 - x2, x3 - x4, colour = class)) +
  geom_point(size = 0.7, alpha = 0.6) +
  facet_grid(sep.lab ~ prev.lab) +
  scale_colour_manual(values = c("Majority (y = 0)" = "#2a78d6",
                                 "Minority (y = 1)" = "#eb6834")) +
  labs(x = "x1 - x2", y = "x3 - x4", colour = NULL,
       title = "Lowering the intercept makes positives rarer, not harder to separate",
       subtitle = "3,000 simulated points per panel; the true slopes are identical in every panel") +
  theme(legend.position = "top", panel.grid.minor = element_blank())

ggsave(file.path(FIGURES_DIR, "fig1_simulated_imbalance.png"),
       fig, width = 9, height = 5, dpi = 200, bg = "white")
cat("Figure saved.\n")

# =============================================================
# 01_check_gen_data.R  --  Step 1: does the generator do what we asked?
# (Class simulation deck, Mistake 4: "check that your generator did
#  what you asked")
# Base-R syntax as in the ISLR (R edition) labs; ggplot for the figure.
# =============================================================
source("R/00_setup.R")
set.seed(SEED)

# ---- Check 1: the intercepts --------------------------------
# One beta0 per (prevalence, separation). The slopes are the same in every row.
beta0.tab <- make_beta0_table()
print(beta0.tab)
saveRDS(beta0.tab, file.path(RESULTS_DIR, "beta0_table.rds"))

# ---- Check 2: observed prevalence vs. target ----------------
# A BIG data set per setting, so the check itself is not noisy.
beta0.tab$observed <- NA
for (i in 1:nrow(beta0.tab)) {
  d <- gen_data(200000, beta0.tab$prev[i], beta0.tab$sep[i], beta0.tab$beta0[i])
  beta0.tab$observed[i] <- mean(d$y == "1")
}
cat("\nTarget vs. observed prevalence (%):\n")
print(data.frame(sep = beta0.tab$sep.level,
                 target = 100 * beta0.tab$prev,
                 observed = round(100 * beta0.tab$observed, 2)))

# ---- Check 3: are the TRUE slopes recovered? -----------------
# Ordinary logistic regression (ISLR 4.7.2, p. 172) on a huge sample.
# The slopes should match sep * beta at EVERY prevalence -- that is the
# whole point of moving only beta0.
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

# ---- Check 4: count vs. proportion of minority cases ---------
# The handout wants a small PROPORTION with a large COUNT, and the reverse.
counts <- outer(PREVALENCES, SAMPLE_SIZES)
dimnames(counts) <- list(paste0("prev = ", PREVALENCES),
                         paste0("n = ", SAMPLE_SIZES))
cat("\nExpected number of minority cases in a training set:\n")
print(counts)

# ---- Check 5: how hard is each setting? (Bayes benchmark preview) ---
# ROC-AUC of the TRUE probabilities = the best ranking any method can get.
# ROCR's prediction() / performance(): ISLR 9.6.3, p. 394.
beta0.tab$bayes.auc <- NA
for (i in 1:nrow(beta0.tab)) {
  d <- gen_data(200000, beta0.tab$prev[i], beta0.tab$sep[i], beta0.tab$beta0[i])
  pred <- prediction(d$p.true, d$y)
  beta0.tab$bayes.auc[i] <- performance(pred, "auc")@y.values[[1]]
}
cat("\nBayes ROC-AUC (best possible) by setting:\n")
print(round(xtabs(bayes.auc ~ sep.level + prev, data = beta0.tab), 3))

# ---- Check 6: reproducibility --------------------------------
set.seed(SEED); a <- gen_data(50, 0.05, 1.5, beta0.tab$beta0[6])
set.seed(SEED); b <- gen_data(50, 0.05, 1.5, beta0.tab$beta0[6])
cat("\nSame seed gives identical data:", identical(a, b), "\n")

# ---- Figure: what imbalance looks like ------------------------
# Same slopes, same predictors; only beta0 changes from panel to panel.
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
plot.df <- plot.df[order(plot.df$class), ]   # draw minority points on top

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

# 12_eda.R
# Exploratory plots for the report.
#   Figure 9:  simulated data -- how the true probabilities are spread at
#              each prevalence, and how few cases pass the 0.5 cutoff.
#   Figure 10: credit card data -- transaction amount by class, and the
#              fraud rate hour by hour over the two days.
# Also prints the numbers quoted in the text.
source("R/00_setup.R")
library(patchwork)
set.seed(SEED)

# ---- Simulated data ----
beta0.tab <- readRDS(file.path(RESULTS_DIR, "beta0_table.rds"))
sim.eda <- NULL
for (i in 1:nrow(beta0.tab)) {
  d <- gen_data(20000, beta0.tab$prev[i], beta0.tab$sep[i], beta0.tab$beta0[i])
  sim.eda <- rbind(sim.eda, data.frame(prev = beta0.tab$prev[i],
                                       sep.level = beta0.tab$sep.level[i],
                                       p.true = d$p.true, y = d$y))
}
sim.eda$prev.lab <- factor(paste0(100 * sim.eda$prev, "% positives"),
                           levels = c("50% positives", "20% positives",
                                      "5% positives", "1% positives"))
sim.eda$sep.lab <- paste(sim.eda$sep.level, "separation")

cat("Share of cases with true P(y = 1) above 0.5:\n")
print(aggregate(p.true ~ prev + sep.level, sim.eda, function(p) round(mean(p > 0.5), 4)))
cat("Share of positives whose true P(y = 1) is above 0.5:\n")
print(aggregate(p.true ~ prev + sep.level, sim.eda[sim.eda$y == "1", ],
                function(p) round(mean(p > 0.5), 3)))

fig9 <- ggplot(sim.eda, aes(p.true)) +
  geom_histogram(binwidth = 0.025, boundary = 0, fill = "#2a78d6", colour = "white",
                 linewidth = 0.1) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "#eb6834") +
  facet_grid(sep.lab ~ prev.lab, scales = "free_y") +
  labs(x = "True probability P(y = 1 | x)", y = "Number of cases (of 20,000)",
       title = "As positives get rarer, almost every case falls below the 0.5 cutoff",
       subtitle = "Dashed line = the usual 0.5 cutoff") +
  theme(panel.grid.minor = element_blank())
ggsave(file.path(FIGURES_DIR, "fig9_eda_simulated.png"), fig9,
       width = 10, height = 4.6, dpi = 200, bg = "white")

# ---- Credit card data ----
cc <- load_creditcard()
cc$class <- ifelse(cc$y == "1", "Fraud", "Normal")
cc$hour  <- floor(cc$Time / 3600)

cat("\nTransactions:", nrow(cc), " frauds:", sum(cc$y == "1"),
    " fraud rate:", round(100 * mean(cc$y == "1"), 3), "%\n")
cat("Median amount by class:\n")
print(tapply(cc$Amount, cc$class, median))
cat("Share of transactions with amount 0 by class:\n")
print(tapply(cc$Amount == 0, cc$class, mean))
cat("Share of amounts under 10 by class:\n")
print(tapply(cc$Amount < 10, cc$class, mean))

by.hour <- aggregate(cbind(n = 1, frauds = cc$y == "1") ~ hour, cc, sum)
by.hour$rate <- 100 * by.hour$frauds / by.hour$n
cat("\nHours with the highest fraud rate:\n")
print(head(by.hour[order(-by.hour$rate), ], 5))
cat("Transactions per hour, range:", range(by.hour$n), "\n")

p.amount <- ggplot(cc, aes(Amount + 1, fill = class)) +
  geom_density(alpha = 0.55, colour = NA, adjust = 1.2) +
  scale_x_log10(breaks = c(1, 10, 100, 1000, 10000),
                labels = c("0", "9", "99", "999", "9,999")) +
  scale_fill_manual(values = c(Fraud = "#eb6834", Normal = "#2a78d6")) +
  labs(x = "Transaction amount (log scale)", y = "Density", fill = NULL,
       title = "A. Amount by class") +
  theme(legend.position = "top", panel.grid.minor = element_blank())

p.hour <- ggplot(by.hour, aes(hour, rate)) +
  geom_col(aes(y = n / max(n) * max(rate)), fill = "grey85", width = 0.9) +
  geom_hline(yintercept = 100 * mean(cc$y == "1"), linetype = "dashed", colour = "grey40") +
  geom_line(colour = "#eb6834", linewidth = 0.6) +
  geom_point(colour = "#eb6834", size = 1.2) +
  scale_y_continuous(sec.axis = sec_axis(~ . / max(by.hour$rate) * max(by.hour$n),
                                         name = "Transactions (grey bars)",
                                         labels = scales::comma)) +
  labs(x = "Hour since the first transaction (two days)", y = "Fraud rate (%)",
       title = "B. Fraud rate by hour") +
  theme(panel.grid.minor = element_blank())

fig10 <- p.amount + p.hour + plot_layout(widths = c(1, 1.4))
ggsave(file.path(FIGURES_DIR, "fig10_eda_creditcard.png"), fig10,
       width = 10, height = 4, dpi = 200, bg = "white")
cat("Saved figures 9 and 10\n")

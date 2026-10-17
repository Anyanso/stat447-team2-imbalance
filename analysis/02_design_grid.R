# =============================================================
# 02_design_grid.R  --  Step 2: build and check the design grid
# =============================================================
source("R/00_setup.R")
set.seed(SEED)

# ---- Block A: the intercepts (same seed as Step 1, so same values) ----
beta0.tab <- make_beta0_table()

# ---- Block B: build the grid ----
design <- make_design(beta0.tab)
cat("Rows in the design grid:", nrow(design), "\n\n")
print(head(design, 3))

# ---- Block C: check the grid is complete ----
# Every one of the 16 settings must have exactly N_REPS replicates.
reps.per.setting <- table(design$setting)
cat("\nSettings:", length(reps.per.setting),
    "| replicates in each:", unique(as.vector(reps.per.setting)), "\n")
stopifnot(length(reps.per.setting) == 16,
          all(reps.per.setting == N_REPS),
          !any(is.na(design$beta0)))

# ---- Block D: how much work is this? ----
n.fits <- nrow(design) * length(APPROACHES) * length(CLASSIFIERS)
cat("Model fits in the full study:", n.fits, "\n")
cat("Rows of test data scored:", format(nrow(design) * N_TEST, big.mark = ",", scientific = FALSE), "\n")

# ---- Block E: the 16 settings at a glance ----
settings <- unique(design[, c("prev", "n", "sep.level", "beta0")])
settings$expected.positives <- settings$n * settings$prev
print(as.data.frame(settings), row.names = FALSE)

# ---- Block F: save ----
saveRDS(design, file.path(RESULTS_DIR, "design.rds"))
cat("\nSaved results/design.rds\n")

# 02_design_grid.R
# Builds the design grid, checks it and saves results/design.rds.
source("R/00_setup.R")
set.seed(SEED)

beta0.tab <- make_beta0_table()   # same seed as 01, so the same intercepts

design <- make_design(beta0.tab)
cat("Rows in the design grid:", nrow(design), "\n\n")
print(head(design, 3))

# every setting should have N_REPS replicates
reps.per.setting <- table(design$setting)
cat("\nSettings:", length(reps.per.setting),
    "| replicates in each:", unique(as.vector(reps.per.setting)), "\n")
stopifnot(length(reps.per.setting) == length(PREVALENCES) * length(SAMPLE_SIZES) * length(SEPARATION),
          all(reps.per.setting == N_REPS),
          !any(is.na(design$beta0)))

n.fits <- nrow(design) * length(APPROACHES) * length(CLASSIFIERS)
cat("Model fits in the full study:", n.fits, "\n")
cat("Rows of test data scored:", format(nrow(design) * N_TEST, big.mark = ",", scientific = FALSE), "\n")

settings <- unique(design[, c("prev", "n", "sep.level", "beta0")])
settings$expected.positives <- settings$n * settings$prev
print(as.data.frame(settings), row.names = FALSE)

saveRDS(design, file.path(RESULTS_DIR, "design.rds"))
cat("\nSaved results/design.rds\n")

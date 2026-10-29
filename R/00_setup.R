# =============================================================
# 00_setup.R  --  source() this at the top of EVERY script and .qmd
#   source("R/00_setup.R")
# =============================================================

# ---- Packages (every session) ----
suppressPackageStartupMessages({
  library(tidyverse)     # class notes: design grid, summaries, ggplot figures
  library(glmnet)        # ISLR 6.5.2 lab: ridge / lasso / penalized logistic
  library(randomForest)  # ISLR 8.3.3 lab: random forests
  library(ROCR)          # ISLR 9.6.3 lab: ROC curves and AUC
  library(FNN)           # nearest neighbours for our own SMOTE helper
})

# ---- Reproducibility (simulation deck, Mistake 1) ----
# ONE seed for the whole project. Record it in the report.
SEED <- 447
set.seed(SEED)

# ---- Paths (relative to the project root, so they work on every laptop) ----
DATA_PATH    <- "data/creditcard.csv"
RESULTS_DIR  <- "results"
FIGURES_DIR  <- "figures"

# ---- Study settings: set ONCE here, never scattered in the code ----
PREVALENCES <- c(0.50, 0.20, 0.05, 0.01)   # minority-class rate (handout)
SAMPLE_SIZES <- c(1000, 20000)             # small count vs. small proportion
SEPARATION  <- c(low = 0.5, high = 1.5)    # multiplies the true slopes
N_REPS      <- 100                         # replicates per setting
N_TEST      <- 20000                       # fresh, large test set per replicate
APPROACHES  <- c("baseline", "oversample", "undersample", "smote",
                 "weights", "threshold")
CLASSIFIERS <- c("logistic", "forest")
N_TREES     <- 200                         # trees per random forest (see Step 3 timing)
CUTOFF_CRITERION <- "youden"                # Approach C: "youden" or "f1"
FOREST_SAMPSIZE  <- NULL                    # rows per tree; NULL = all (real data sets its own)
LASSO_MIN_RATIO  <- NULL                    # lasso lambda range; NULL = glmnet default (real data sets its own)
LASSO_NLAMBDA    <- NULL                    # number of lambdas; NULL = glmnet default (100)
N_SPLITS         <- 10                      # real data: repeated train/test splits

# ---- Plot theme ----
theme_set(theme_minimal(base_size = 11) +
            theme(plot.background = element_rect(fill = "white", colour = NA)))

# ---- Load the project's own functions (each person's file) ----
for (f in c("R/gen_data.R", "R/design.R", "R/approaches.R", "R/evaluate.R",
            "R/run_one.R", "R/real_data.R",
            "R/ensemble.R")) {
  if (file.exists(f)) source(f)
}

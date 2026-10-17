# =============================================================
# 00_setup.R  --  source() this at the top of EVERY script and .qmd
#   source("R/00_setup.R")
# =============================================================

# ---- Packages (every session) ----
suppressPackageStartupMessages({
  library(tidyverse)
  library(MASS)      # loads select(); we restore dplyr::select below
  library(glmnet)
  library(ranger)
  library(pROC)
  library(FNN)
})
select <- dplyr::select   # MASS masks dplyr::select -- undo that

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

# ---- Plot theme ----
theme_set(theme_minimal(base_size = 11))

# ---- Load the project's own functions (each person's file) ----
for (f in c("R/gen_data.R", "R/approaches.R", "R/evaluate.R", "R/ensemble.R")) {
  if (file.exists(f)) source(f)
}

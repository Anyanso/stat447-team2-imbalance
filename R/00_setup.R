# 00_setup.R
# Packages, seed, paths and study settings. Every script starts with
# source("R/00_setup.R").

suppressPackageStartupMessages({
  library(tidyverse)     # design grid, summaries, plots
  library(glmnet)        # lasso logistic regression (ISLR lab 6.5.2)
  library(randomForest)  # random forests (ISLR lab 8.3.3)
  library(ROCR)          # ROC and PR curves (ISLR lab 9.6.3)
  library(FNN)           # nearest neighbours for SMOTE
})

# One seed for the whole project, reported in the write-up
SEED <- 447
set.seed(SEED)

# Paths are relative to the project folder, so open the .Rproj first
DATA_PATH    <- "data/creditcard.csv"
RESULTS_DIR  <- if (exists("RESULTS_DIR_OVERRIDE")) RESULTS_DIR_OVERRIDE else "results"
FIGURES_DIR  <- if (exists("FIGURES_DIR_OVERRIDE")) FIGURES_DIR_OVERRIDE else "figures"

# Study settings -- change them here and nowhere else
PREVALENCES <- c(0.50, 0.20, 0.05, 0.01)   # share of positives
SAMPLE_SIZES <- c(1000, 20000)             # training set sizes
SEPARATION  <- c(low = 0.5, high = 1.5)    # multiplies the true slopes
N_REPS      <- 100                         # replicates per setting
N_TEST      <- 20000                       # size of each fresh test set
APPROACHES  <- c("baseline", "oversample", "undersample", "smote",
                 "weights", "threshold")
CLASSIFIERS <- c("logistic", "forest")
N_TREES     <- 200                         # trees per forest
CUTOFF_CRITERION <- "youden"                # how the threshold approach picks its cutoff ("youden" or "f1")
FOREST_SAMPSIZE  <- NULL                    # rows drawn per tree; NULL = usual bootstrap of all rows
BIG_N_SAMPSIZE   <- 20000                   # rows per tree for simulated forests at n = 100,000
LASSO_MIN_RATIO  <- NULL                    # NULL = glmnet's default lambda path
LASSO_NLAMBDA    <- NULL                    # NULL = glmnet's default (100 lambdas)
N_SPLITS         <- 10                      # train/test splits of the real data
ENSEMBLE_B       <- 10                      # members in EasyEnsemble

# Plot theme. 10_slide_figures.R sets SLIDE_MODE for bigger text and no
# titles inside the figures (the slide title is enough).
PLOT_BASE_SIZE <- if (exists("SLIDE_MODE") && SLIDE_MODE) 17 else 11
theme_set(theme_minimal(base_size = PLOT_BASE_SIZE) +
            theme(plot.background = element_rect(fill = "white", colour = NA)))
if (exists("SLIDE_MODE") && SLIDE_MODE) {
  theme_update(plot.title = element_blank(), plot.subtitle = element_blank())
}

# Our own functions
for (f in c("R/gen_data.R", "R/design.R", "R/approaches.R", "R/evaluate.R",
            "R/run_one.R", "R/real_data.R",
            "R/ensemble.R")) {
  if (file.exists(f)) source(f)
}

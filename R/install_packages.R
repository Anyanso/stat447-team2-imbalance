# =============================================================
# install_packages.R  --  run ONCE per computer (Lecture 2: install vs load)
# =============================================================
pkgs <- c(
  "tidyverse",  # data wrangling + ggplot2 (course standard)
  "MASS",       # mvrnorm(): correlated normal predictors for the simulation
  "glmnet",     # penalized logistic regression (Lecture 11)
  "ranger",     # fast random forest
  "pROC",       # ROC curve and ROC-AUC (Lecture 6)
  "FNN"         # nearest neighbours, used by our own SMOTE helper
)

missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing) > 0) install.packages(missing)

cat("All packages installed:", all(pkgs %in% rownames(installed.packages())), "\n")

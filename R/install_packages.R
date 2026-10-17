# =============================================================
# install_packages.R  --  run ONCE per computer (Lecture 2: install vs load)
# =============================================================
pkgs <- c(
  "tidyverse",     # class notes: design grid, summaries, ggplot figures
  "glmnet",        # ISLR 6.5.2 lab: penalized logistic regression
  "randomForest",  # ISLR 8.3.3 lab: random forests
  "ROCR",          # ISLR 9.6.3 lab: ROC curves and AUC
  "FNN"            # nearest neighbours, used by our own SMOTE helper
)

missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing) > 0) install.packages(missing)

cat("All packages installed:", all(pkgs %in% rownames(installed.packages())), "\n")

# install_packages.R -- run once on each computer

pkgs <- c(
  "tidyverse",
  "glmnet",
  "randomForest",
  "ROCR",
  "FNN"            # used by our SMOTE function
)

missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing) > 0) install.packages(missing)

cat("All packages installed:", all(pkgs %in% rownames(installed.packages())), "\n")

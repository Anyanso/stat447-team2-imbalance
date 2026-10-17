# =============================================================
# 00_check_setup.R  --  every teammate runs this once after cloning.
# Open the .Rproj first so the working directory is the project root.
# =============================================================
source("R/00_setup.R")

cat("R version:  ", R.version.string, "\n")
cat("Seed:       ", SEED, "\n\n")

# ---- 1. Is the data where it should be? ----
if (!file.exists(DATA_PATH)) {
  stop("Download creditcard.csv from Kaggle and put it in data/")
}

cc <- read_csv(DATA_PATH, show_col_types = FALSE)

cat("Rows:", nrow(cc), " Columns:", ncol(cc), "\n")
stopifnot(nrow(cc) == 284807, ncol(cc) == 31)

# ---- 2. How imbalanced is it? ----
cc |>
  count(Class) |>
  mutate(percent = round(100 * n / sum(n), 3)) |>
  print()

# ---- 3. Any missing values? ----
cat("Missing values:", sum(is.na(cc)), "\n")

# ---- 4. The 'useless 99.8%' classifier from the handout ----
null_accuracy <- mean(cc$Class == 0)
cat("Accuracy of always predicting 'not fraud':",
    round(100 * null_accuracy, 2), "%\n")

cat("\nSetup OK.\n")

# 00_check_setup.R
# Run once after cloning (with the .Rproj open) to check packages and data.
source("R/00_setup.R")

cat("R version:  ", R.version.string, "\n")
cat("Seed:       ", SEED, "\n\n")

if (!file.exists(DATA_PATH)) {
  stop("Download creditcard.csv from Kaggle and put it in data/")
}

cc <- read_csv(DATA_PATH, show_col_types = FALSE)

cat("Rows:", nrow(cc), " Columns:", ncol(cc), "\n")
stopifnot(nrow(cc) == 284807, ncol(cc) == 31)

# class balance
cc |>
  count(Class) |>
  mutate(percent = round(100 * n / sum(n), 3)) |>
  print()

cat("Missing values:", sum(is.na(cc)), "\n")

# accuracy of predicting "not fraud" for everything
null_accuracy <- mean(cc$Class == 0)
cat("Accuracy of always predicting 'not fraud':",
    round(100 * null_accuracy, 2), "%\n")

cat("\nSetup OK.\n")

# 07_run_real_data.R
# Runs every approach on the credit card data. Needs data/creditcard.csv.
#   1. PILOT <- TRUE runs one split and reports how long it took.
#   2. PILOT <- FALSE runs all N_SPLITS splits, saving each as it
#      finishes (run again to resume).
#   3. Then push results/real_results.rds.
PILOT   <- TRUE
N_CORES <- max(1, min(3, parallel::detectCores() - 1))  # at most 3 at once, to save memory

source("R/00_setup.R")
library(parallel)
FOREST_SAMPSIZE <- 20000      # rows per tree
LASSO_MIN_RATIO <- 1e-3       # glmnet's default path doesn't converge here,
LASSO_NLAMBDA   <- 200        # so use a shorter, finer one

# which splits still need running?
splits    <- if (PILOT) 1 else 1:N_SPLITS
chunk.dir <- file.path(RESULTS_DIR, if (PILOT) "real_pilot_chunks" else "real_chunks")
dir.create(chunk.dir, showWarnings = FALSE)
done <- as.integer(sub("split_(\\d+)\\.rds", "\\1", list.files(chunk.dir)))
todo <- setdiff(splits, done)
cat(length(splits), "splits in total;", length(done), "already done;",
    length(todo), "to run on", min(N_CORES, max(1, length(todo))), "cores.\n")

run_and_save <- function(s) {
  res <- run_real_split(s, cc)
  saveRDS(res, file.path(chunk.dir, sprintf("split_%02d.rds", s)))
  s
}

start <- Sys.time()
if (length(todo) == 1) {
  cc <- load_creditcard()
  run_and_save(todo)
} else if (length(todo) > 1) {
  cl <- makeCluster(min(N_CORES, length(todo)))
  clusterCall(cl, setwd, getwd())
  clusterEvalQ(cl, {
    suppressMessages(source("R/00_setup.R"))
    FOREST_SAMPSIZE <- 20000
    LASSO_MIN_RATIO <- 1e-3
    LASSO_NLAMBDA   <- 200
    cc <- load_creditcard()
    NULL
  })
  clusterExport(cl, c("chunk.dir", "run_and_save"))
  invisible(parLapplyLB(cl, todo, run_and_save))
  stopCluster(cl)
}
mins <- as.numeric(difftime(Sys.time(), start, units = "mins"))
cat("Finished", length(todo), "split(s) in", round(mins, 1), "minutes.\n")

# combine and save
files <- list.files(chunk.dir, full.names = TRUE)
real <- do.call(rbind, lapply(files, readRDS))
out.file <- file.path(RESULTS_DIR, if (PILOT) "real_pilot_results.rds" else "real_results.rds")
saveRDS(real, out.file)
cat("Splits in results:", length(unique(real$split.id)), "| saved", out.file, "\n")

# quick look
show <- aggregate(cbind(roc.auc, pr.auc, sensitivity, precision, bal.accuracy, accuracy) ~
                    classifier + approach, data = real, FUN = mean)
show[, -(1:2)] <- round(show[, -(1:2)], 3)
print(show[order(show$classifier), ], row.names = FALSE)
if (PILOT) {
  cat("\nSeconds per fit in this split:\n")
  print(real[, c("classifier", "approach", "seconds")], row.names = FALSE)
  cat("\nEstimated full run on this laptop:",
      round(mins * N_SPLITS / min(N_CORES, N_SPLITS) / 60, 1), "hours with",
      min(N_CORES, N_SPLITS), "cores\n")
}

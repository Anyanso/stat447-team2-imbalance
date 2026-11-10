# 09_run_ensemble.R
# Runs EasyEnsemble on the same simulated data sets and real splits as
# 05 and 07 (the other methods' results are reused, not rerun).
#   PART <- "real" : the 10 real splits (needs data/creditcard.csv)
#   PART <- "sim"  : the simulated replicates (settings already in
#                    results/ens_sim_results.rds are skipped and the new
#                    ones are added to it)
# Saved in chunks; run again to resume.
PART    <- "real"                                   # "real" or "sim"
if (length(commandArgs(TRUE)) > 0) PART <- commandArgs(TRUE)[1]   # e.g. Rscript ... sim
N_CORES <- if (PART == "real") max(1, min(3, parallel::detectCores() - 1)) else  # real data: save memory
             max(1, parallel::detectCores() - 1)

source("R/00_setup.R")
library(parallel)
if (PART == "real") {              # same settings as 07_run_real_data.R
  FOREST_SAMPSIZE <- 20000
  LASSO_MIN_RATIO <- 1e-3
  LASSO_NLAMBDA   <- 200
}

out.file <- file.path(RESULTS_DIR, paste0("ens_", PART, "_results.rds"))

# jobs: one per real split, or chunks of 25 replicates (10 at n = 100,000)
if (PART == "sim") {
  old <- if (file.exists(out.file)) readRDS(out.file) else NULL
  design <- readRDS(file.path(RESULTS_DIR, "design.rds"))
  design <- design[!design$setting %in% old$setting, ]
  design$chunk <- paste0(gsub("[^0-9a-z]+", "_", design$setting), "_",
                         sprintf("%03d", ceiling(design$rep.id / ifelse(design$n > 20000, 10, 25))))
  jobs <- sort(unique(design$chunk))
} else {
  jobs <- sprintf("split_%02d", 1:N_SPLITS)
}
chunk.dir <- file.path(RESULTS_DIR, paste0("ens_", PART, "_chunks"))
dir.create(chunk.dir, showWarnings = FALSE)
todo <- setdiff(jobs, sub("\\.rds$", "", list.files(chunk.dir)))
cat(length(jobs), "jobs;", length(jobs) - length(todo), "already done;",
    length(todo), "to run.\n")

run_job <- function(job) {
  if (PART == "sim") {
    rows <- design[design$chunk == job, ]
    res <- NULL
    for (i in 1:nrow(rows)) res <- rbind(res, run_one_ensemble(rows[i, ]))
  } else {
    res <- run_real_split_ensemble(as.integer(sub("split_", "", job)), cc)
  }
  saveRDS(res, file.path(chunk.dir, paste0(job, ".rds")))
  job
}

start <- Sys.time()
if (length(todo) > 0) {
  cl <- makeCluster(min(N_CORES, length(todo)))
  clusterCall(cl, setwd, getwd())
  clusterExport(cl, c("PART", "chunk.dir", "run_job"))
  if (PART == "sim") clusterExport(cl, "design")
  clusterEvalQ(cl, {
    suppressMessages(source("R/00_setup.R"))
    if (PART == "real") {
      FOREST_SAMPSIZE <- 20000; LASSO_MIN_RATIO <- 1e-3; LASSO_NLAMBDA <- 200
      cc <- load_creditcard()
    }
    NULL
  })
  invisible(parLapplyLB(cl, todo, run_job))
  stopCluster(cl)
}
cat("Finished in", round(as.numeric(difftime(Sys.time(), start, units = "mins")), 1), "minutes.\n")

# combine and save
res <- do.call(rbind, lapply(file.path(chunk.dir, paste0(jobs, ".rds")), readRDS))
if (PART == "sim") res <- rbind(old, res)
saveRDS(res, out.file)
cat("Rows:", nrow(res), "| saved", out.file, "\n")

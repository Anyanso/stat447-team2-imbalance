# =============================================================
# 09_run_ensemble.R  --  Step 9 (STAT 847): run EasyEnsemble on the
# SAME simulated data sets (Step 5) and real splits (Step 7).
# Only the ensemble is run; the other 12 methods' results are reused.
#
# HOW TO USE (open the .Rproj first)
#   PART <- "real" : 10 real splits (minutes; needs data/creditcard.csv)
#   PART <- "sim"  : all 1,600 simulated replicates (about 2 hours here)
#   Saved in chunks as it goes; rerun to resume.
# Output: results/ens_real_results.rds or results/ens_sim_results.rds
# =============================================================
PART    <- "real"                                   # "real" or "sim"
if (length(commandArgs(TRUE)) > 0) PART <- commandArgs(TRUE)[1]   # e.g. Rscript ... sim
N_CORES <- max(1, min(3, parallel::detectCores() - 1))

source("R/00_setup.R")
library(parallel)
if (PART == "real") {              # same real-data settings as Step 7
  FOREST_SAMPSIZE <- 20000
  LASSO_MIN_RATIO <- 1e-3
  LASSO_NLAMBDA   <- 200
}

# ---- Block A: the jobs (one per real split, or 25 replicates per chunk) ----
if (PART == "sim") {
  design <- readRDS(file.path(RESULTS_DIR, "design.rds"))
  design$chunk <- paste0(gsub("[^0-9a-z]+", "_", design$setting), "_",
                         sprintf("%03d", ceiling(design$rep.id / 25)))
  jobs <- sort(unique(design$chunk))
} else {
  jobs <- sprintf("split_%02d", 1:N_SPLITS)
}
chunk.dir <- file.path(RESULTS_DIR, paste0("ens_", PART, "_chunks"))
dir.create(chunk.dir, showWarnings = FALSE)
todo <- setdiff(jobs, sub("\\.rds$", "", list.files(chunk.dir)))
cat(length(jobs), "jobs;", length(jobs) - length(todo), "already done;",
    length(todo), "to run.\n")

# ---- Block B: run one job and save it straight away ----
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

# ---- Block C: in parallel (same pattern as Steps 5 and 7) ----
start <- Sys.time()
if (length(todo) > 0) {
  cl <- makeCluster(min(N_CORES, length(todo)))
  clusterCall(cl, setwd, getwd())
  clusterExport(cl, c("PART", "chunk.dir", "run_job"))
  clusterEvalQ(cl, {
    suppressMessages(source("R/00_setup.R"))
    if (PART == "real") {
      FOREST_SAMPSIZE <- 20000; LASSO_MIN_RATIO <- 1e-3; LASSO_NLAMBDA <- 200
      cc <- load_creditcard()
    } else {
      design <- readRDS(file.path(RESULTS_DIR, "design.rds"))
      design$chunk <- paste0(gsub("[^0-9a-z]+", "_", design$setting), "_",
                             sprintf("%03d", ceiling(design$rep.id / 25)))
    }
    NULL
  })
  invisible(parLapplyLB(cl, todo, run_job))
  stopCluster(cl)
}
cat("Finished in", round(as.numeric(difftime(Sys.time(), start, units = "mins")), 1), "minutes.\n")

# ---- Block D: combine and save ----
res <- do.call(rbind, lapply(list.files(chunk.dir, full.names = TRUE), readRDS))
out.file <- file.path(RESULTS_DIR, paste0("ens_", PART, "_results.rds"))
saveRDS(res, out.file)
cat("Rows:", nrow(res), "| saved", out.file, "\n")

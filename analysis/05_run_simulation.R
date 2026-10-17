# =============================================================
# 05_run_simulation.R  --  Step 5: run the full simulation
#
# HOW TO USE (open the .Rproj first so you are in the project folder)
#   1. Leave PILOT <- TRUE and run the whole script once (~10 min).
#      It runs 2 replicates per setting and prints an estimate of the
#      full run time on YOUR laptop.
#   2. Set PILOT <- FALSE and run it again (overnight is fine).
#      Results are saved in small chunks as it goes. If it stops for any
#      reason, just run it again: finished chunks are skipped.
#   3. When it finishes, upload results/sim_results.rds.
# =============================================================
PILOT   <- TRUE
N_CORES <- max(1, parallel::detectCores() - 1)   # leave one core free for you

source("R/00_setup.R")
library(parallel)                                # comes with R; nothing to install
design <- readRDS(file.path(RESULTS_DIR, "design.rds"))

# ---- Block A: which rows, in which chunks? ----
# Chunk = 25 replicates of one setting. Fast settings (n = 1000) first,
# so you get usable results early.
if (PILOT) design <- design[design$rep.id <= 2, ]
design$chunk <- paste0(ifelse(design$n == 1000, "a", "b"), "_",
                       gsub("[^0-9a-z]+", "_", design$setting), "_",
                       sprintf("%03d", ceiling(design$rep.id / 25)))
chunks <- sort(unique(design$chunk))

chunk.dir <- file.path(RESULTS_DIR, if (PILOT) "pilot_chunks" else "sim_chunks")
dir.create(chunk.dir, showWarnings = FALSE)
done <- sub("\\.rds$", "", list.files(chunk.dir))
todo <- setdiff(chunks, done)
cat(length(chunks), "chunks in total;", length(done), "already done;",
    length(todo), "to run on", N_CORES, "cores.\n")

# ---- Block B: run one chunk and save it straight away ----
run_chunk <- function(ch) {
  rows <- design[design$chunk == ch, ]
  res  <- NULL
  for (i in 1:nrow(rows)) res <- rbind(res, run_one(rows[i, ]))
  saveRDS(res, file.path(chunk.dir, paste0(ch, ".rds")))
  ch
}

# ---- Block C: run the chunks in parallel ----
# makeCluster() starts N_CORES separate R sessions (works on Windows,
# Mac and Linux). Each one loads our code, then takes chunks in turn.
start <- Sys.time()
if (length(todo) > 0) {
  cl <- makeCluster(N_CORES)
  clusterCall(cl, setwd, getwd())                    # same project folder
  clusterEvalQ(cl, suppressMessages(source("R/00_setup.R")))
  clusterExport(cl, c("design", "chunk.dir", "run_chunk"))
  finished <- parLapplyLB(cl, todo, function(ch) {
    out <- run_chunk(ch)
    out
  })
  stopCluster(cl)
}
mins <- as.numeric(difftime(Sys.time(), start, units = "mins"))
cat("Finished", length(todo), "chunks in", round(mins, 1), "minutes.\n")

# ---- Block D: combine all chunks into one results file ----
files <- list.files(chunk.dir, full.names = TRUE)
results <- do.call(rbind, lapply(files, readRDS))
cat("Result rows:", nrow(results), "| replicates:",
    nrow(unique(results[, c("setting", "rep.id")])), "\n")

out.file <- file.path(RESULTS_DIR, if (PILOT) "pilot_results.rds" else "sim_results.rds")
saveRDS(results, out.file)
cat("Saved", out.file, "\n")

# ---- Block E (pilot only): estimate the full run time on this laptop ----
if (PILOT && length(todo) == length(chunks)) {
  full.mins <- mins * (N_REPS / 2)
  cat("\nEstimated full run on this laptop with", N_CORES, "cores:",
      round(full.mins / 60, 1), "hours\n")
  cat("Fit time per approach (seconds, average over the pilot):\n")
  print(round(tapply(results$seconds, list(results$approach, results$n), mean), 2))
}

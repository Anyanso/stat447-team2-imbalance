# 05_run_simulation.R
# Runs the simulation. Open the .Rproj first.
#   1. With PILOT <- TRUE it runs 2 replicates per setting and estimates
#      how long the full run will take on your laptop.
#   2. Set PILOT <- FALSE for the full run (leave it overnight).
#      Results are saved in chunks, so if it stops just run it again
#      and finished chunks are skipped.
#   3. Then push results/sim_results.rds.
# Settings already in results/sim_results.rds are not run again; the new
# ones (n = 100,000) are added to that file at the end.
PILOT   <- TRUE
N_CORES <- max(1, parallel::detectCores() - 1)   # leave one core free

source("R/00_setup.R")
library(parallel)
design <- readRDS(file.path(RESULTS_DIR, "design.rds"))

sim.file <- file.path(RESULTS_DIR, "sim_results.rds")
old <- if (file.exists(sim.file)) readRDS(sim.file) else NULL
design <- design[!design$setting %in% old$setting, ]
cat("Settings already done:", length(unique(old$setting)),
    "| settings to run:", length(unique(design$setting)), "\n")

# chunks of 25 replicates (10 at n = 100,000, which is slow); small n first
if (PILOT) design <- design[design$rep.id <= 2, ]
per.chunk <- ifelse(design$n > 20000, 10, 25)
design$chunk <- paste0(c("a", "b", "c")[match(design$n, SAMPLE_SIZES)], "_",
                       gsub("[^0-9a-z]+", "_", design$setting), "_",
                       sprintf("%03d", ceiling(design$rep.id / per.chunk)))
chunks <- sort(unique(design$chunk))

chunk.dir <- file.path(RESULTS_DIR, if (PILOT) "pilot_chunks" else "sim_chunks")
dir.create(chunk.dir, showWarnings = FALSE)
done <- sub("\\.rds$", "", list.files(chunk.dir))
todo <- setdiff(chunks, done)
cat(length(chunks), "chunks in total;", length(chunks) - length(todo), "already done;",
    length(todo), "to run on", N_CORES, "cores.\n")

# run one chunk and save it straight away
run_chunk <- function(ch) {
  rows <- design[design$chunk == ch, ]
  res  <- NULL
  for (i in 1:nrow(rows)) res <- rbind(res, run_one(rows[i, ]))
  saveRDS(res, file.path(chunk.dir, paste0(ch, ".rds")))
  ch
}

# run the chunks in parallel; makeCluster() works on Windows, Mac and Linux
start <- Sys.time()
if (length(todo) > 0) {
  cl <- makeCluster(N_CORES)
  clusterCall(cl, setwd, getwd())
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

# combine this run's chunks
files <- file.path(chunk.dir, paste0(chunks, ".rds"))
new <- do.call(rbind, lapply(files, readRDS))
cat("New result rows:", nrow(new), "| replicates:",
    nrow(unique(new[, c("setting", "rep.id")])), "\n")

if (PILOT) {
  saveRDS(new, file.path(RESULTS_DIR, "pilot_results.rds"))
  if (length(todo) == length(chunks)) {
    full.mins <- mins * (N_REPS / 2)
    cat("\nEstimated full run on this laptop with", N_CORES, "cores:",
        round(full.mins / 60, 1), "hours\n")
    cat("Fit time per approach (seconds, average over the pilot):\n")
    print(round(tapply(new$seconds, list(new$approach, new$n), mean), 2))
  }
} else {
  results <- rbind(old, new)
  saveRDS(results, sim.file)
  cat("Saved", sim.file, "with", nrow(unique(results[, c("setting", "rep.id")])),
      "replicates in total\n")
}

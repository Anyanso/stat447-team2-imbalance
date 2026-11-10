# design.R -- the simulation design (John Anyanso)
# One row per simulated data set: prevalence x n x separation x replicate.

make_design <- function(beta0.tab, n.reps = N_REPS) {

  grid <- function(sizes) {
    expand_grid(prev      = PREVALENCES,
                n         = sizes,
                sep.level = names(SEPARATION),
                rep.id    = 1:n.reps)
  }
  # n = 100,000 was added after the first run, so its rows go at the end.
  # That way the first 1,600 rows keep the seeds (and data) they had then.
  first <- c(1000, 20000)
  design <- bind_rows(grid(first), grid(setdiff(SAMPLE_SIZES, first)))

  design <- left_join(design,
                      beta0.tab[, c("prev", "sep.level", "sep", "beta0")],
                      by = c("prev", "sep.level"))

  design$setting <- paste0("prev = ", 100 * design$prev, "%, n = ",
                           format(design$n, scientific = FALSE, trim = TRUE),
                           ", sep = ", design$sep.level)

  # Each row gets its own seed, so any replicate can be rerun on its own
  # (for debugging, or to share the work between laptops).
  design$seed <- SEED + seq_len(nrow(design))

  design
}

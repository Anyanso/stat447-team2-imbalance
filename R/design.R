# =============================================================
# design.R  --  OWNER: Person 1   (Step 2)
# The design grid: one row = one simulated data set to create.
# Class simulation deck, S.7 Step 3 ("Factors = the things you vary;
# levels = the values they take. Set them in ONE place.")
# =============================================================

make_design <- function(beta0.tab, n.reps = N_REPS) {

  # ---- Block 1: every combination of the three factors x replicates ----
  design <- expand_grid(prev      = PREVALENCES,         # 4 levels
                        n         = SAMPLE_SIZES,        # 2 levels
                        sep.level = names(SEPARATION),   # 2 levels
                        rep.id    = 1:n.reps)            # 100 replicates

  # ---- Block 2: attach the intercept (and sep value) for each setting ----
  design <- left_join(design,
                      beta0.tab[, c("prev", "sep.level", "sep", "beta0")],
                      by = c("prev", "sep.level"))

  # ---- Block 3: a readable label for each of the 16 settings ----
  design$setting <- paste0("prev = ", 100 * design$prev, "%, n = ", design$n,
                           ", sep = ", design$sep.level)

  # ---- Block 4: one seed per replicate ----
  # Any single row can be re-run on its own and give the same data,
  # e.g. to debug replicate 37, or to split the work across laptops.
  design$seed <- SEED + seq_len(nrow(design))

  design
}

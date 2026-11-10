# 10_slide_figures.R
# Redraws the figures for the slides (bigger text, no titles) into
# figures/slides/. Tables get written to a temporary copy of results/
# so the real one isn't touched.
SLIDE_MODE <- TRUE
FIGURES_DIR_OVERRIDE <- "figures/slides"
RESULTS_DIR_OVERRIDE <- file.path(tempdir(), "results_copy")
dir.create(FIGURES_DIR_OVERRIDE, showWarnings = FALSE, recursive = TRUE)
dir.create(RESULTS_DIR_OVERRIDE, showWarnings = FALSE, recursive = TRUE)
invisible(file.copy(list.files("results", full.names = TRUE, pattern = "\\.(rds|csv)$"),
                    RESULTS_DIR_OVERRIDE, overwrite = TRUE))

for (script in c("analysis/01_check_gen_data.R",
                 "analysis/06_summarise_simulation.R", "analysis/08_reconcile.R",
                 "analysis/08b_compare_wadud.R", "analysis/09b_summarise_ensemble.R")) {
  cat("\n=== ", script, " ===\n")
  source(script)
}
cat("\nSlide figures:", list.files(FIGURES_DIR_OVERRIDE), sep = "\n  ")

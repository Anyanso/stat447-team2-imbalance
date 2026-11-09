# =============================================================
# 10_slide_figures.R  --  Redraw the report figures at SLIDE size:
# bigger text, no in-figure titles (the slide title does that job).
# Output: figures/slides/*.png   (the report keeps figures/*.png)
# Results tables are written to a temporary copy, so the committed
# results/ folder is never touched.
# =============================================================
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

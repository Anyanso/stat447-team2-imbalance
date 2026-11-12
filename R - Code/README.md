# STAT 847 Team 2 — Class Imbalance: Can Prediction Be Improved?
University of Saskatchewan · Fall 2026 · Dr. Li Xing

## Project contents

| File | Purpose |
|---|---|
| `00_shared_functions.R` | Reusable metrics, train-only preprocessing, sampling, nested tuning, model comparison and ensemble calculations. |
| `01_simulation_analysis.R` | Generate and analyze logistic simulation scenarios (50%, 20%, 5%, 1% positive prevalence). |
| `02_fraud_real_data_analysis.R` | Analyze the original ULB credit-card fraud data, using your local CSV and two holdout designs. |
| `03_run_all.R` | Run simulation and fraud analyses and write a combined results comparison. |

The original student working script was reorganized into these files. The research design is unchanged in its main features, and an extra fraud hourly descriptive figure and a combined AP figure were added.

## Setup in RStudio (Windows)

1. Unzip the project into your **RStudio project folder** (the directory containing the project's `.Rproj` file), or create a new RStudio project in the extracted folder.
2. Install required CRAN packages once:

   ```r
   install.packages(c("data.table", "ggplot2", "glmnet", "ranger", "themis"))
   ```

3. Download the **original** ULB dataset as `creditcard.csv` and place it at:

   ```text
   C:/Users/HP/Downloads/creditcard.csv
   ```

   To use a different location, edit `csv_file` near the top of `02_fraud_real_data_analysis.R`, or set:

   ```r
   Sys.setenv(STAT847_FRAUD_CSV = "D:/my_data/creditcard.csv")
   ```

   **No Kaggle CLI, authentication, or internet download is used by the analysis scripts.**

4. Open the RStudio project and ensure `getwd()` is the project folder. Run the complete project:

   ```r
   source("03_run_all.R")
   ```

   Alternatively run the two analyses separately:

   ```r
   source("01_simulation_analysis.R")
   source("02_fraud_real_data_analysis.R")
   ```

   The combined-results tables are produced when `03_run_all.R` is run.

## Fast structural check (optional)

The full study is computationally intensive because it repeats nested cross-validation across many scenarios and fits forests to the full fraud dataset twice. To run a **reduced smoke test** first:

```r
Sys.setenv(STAT847_FAST_CHECK = "1")
source("03_run_all.R")
```

This uses one simulation replicate, N=2,000, 30 trees and 25 bootstrap samples. Both fraud holdouts still run; the reduced run may remain slow. **Fast-check outputs are not the full research results.** To run the intended full settings afterward:

```r
Sys.unsetenv("STAT847_FAST_CHECK")
source("03_run_all.R")
```

Each run replaces files with the same output filenames, so copy or rename any results you want to preserve before switching modes.

## Statistical design

- **Simulation.** Independent standard-normal predictors `X1`–`X4`; slopes `(1.25, −1.0, 0.8, 0)` fixed. Target positive prevalence 50%, 20%, 5%, or 1% is controlled through the intercept. Full settings: N=2,000 and 20,000 for all rates, plus N=100,000 at 1%; three independent replicates of each scenario. Realized sample prevalence may fluctuate.
- **Models.** Ridge-penalized logistic regression (`glmnet`, `alpha=0`) and a probability random forest (`ranger`, 150 trees at full settings).
- **Training remedies.** Unmodified baseline, minority oversampling, majority undersampling, SMOTE, and class weighting. F1/F2 threshold tuning is a distinct decision rule applied to predicted probabilities.
- **Cross-validation.** A stratified 80%/20% split. The training portion uses stratified 5-fold out-of-fold (OOF) predictions. For logistic regression, ridge lambda is selected by an inner 3-fold CV performed *inside the current training fold*. Scaling and resampling are learned only on the corresponding training rows; validation and test rows are never resampled.
- **Thresholds and ensembles.** Cutoffs and the mixing weight (logistic vs random forest) are chosen from training OOF predictions; the best single model is chosen by training OOF AP. Final reporting uses held-out test labels only *after* these decisions are fixed.
- **Metrics.** Accuracy, sensitivity, specificity, precision, F1, balanced accuracy, ROC-AUC, **average precision** (`PR_AUC_AP`, non-interpolated AP), Brier score, log loss, confusion-matrix counts, and prevalence. Note that AP is not identical to trapezoidally interpolated PR-AUC.
- **Uncertainty.** Paired, class-stratified bootstrap intervals for the test AP difference between the ensemble and the best training-OOF-selected individual learner. These reflect uncertainty in the paired held-out comparison conditional on the fitted/selected models; they do not account for retraining and reselection across bootstrap samples.
- **Real data.** The original CSV is checked for exactly 284,807 transactions, 492 frauds, the expected 31 columns, and valid numeric values. `Amount` is transformed using `log1p`; exact duplicate rows are removed before splitting. Both a stratified random holdout and a chronological 20% test holdout are evaluated. **The chronological TEST is later in time, but the 5-fold training CV remains random stratified, not forward-chaining.** Do not describe it as rolling-origin CV.

## Output layout

Every file created by the analysis appears under `results/simulation/`:

```text
results/
└── simulation/
    ├── tables/                       # per-scenario scores, summaries and comparisons
    ├── data/
    │   ├── generated/                # all simulated raw datasets, gzip CSV
    │   ├── design/                   # simulation-generating information
    │   └── splits/                   # original-row IDs of train/test partitions
    ├── plots/
    │   ├── eda/                      # class imbalance and predictor density plots
    │   ├── final/                    # AP, ROC, F1, remedies, ensembles etc.
    │   └── curves/
    ├── metadata/                     # settings and session information
    ├── fraud_real_data/
    │   ├── tables/                   # EDA, test metrics, ensembles, predictions
    │   ├── data/
    │   │   ├── raw/                  # copy of local creditcard.csv
    │   │   ├── processed/            # deduplicated, log1p-amount model data
    │   │   └── splits/               # processed-row IDs for both holdouts
    │   ├── plots/
    │   │   ├── eda/
    │   │   ├── curves/               # held-out PR and ROC curves, both splits
    │   │   └── final/                # model comparisons and selected confusion matrices
    │   └── metadata/
    └── combined_results/
        ├── tables/                   # cross-study comparisons; no pooled fitting
        ├── plots/                    # faceted comparison of held-out AP
        └── metadata/
```

### Main files to inspect after completion

- `results/simulation/tables/simulation__all_metrics.csv`
- `results/simulation/tables/simulation__scenario_summary.csv`
- `results/simulation/tables/simulation__remedy_deltas.csv`
- `results/simulation/tables/simulation__ensemble_comparisons.csv`
- `results/simulation/fraud_real_data/tables/fraud__all_metrics.csv`
- `results/simulation/fraud_real_data/tables/fraud__CV_selected_models.csv`
- `results/simulation/fraud_real_data/tables/fraud__ensemble_comparisons.csv`
- `results/simulation/combined_results/tables/both_studies_default_threshold.csv`

## Interpretation and limitations

- **Do not compare raw AP values across the simulation and fraud datasets without mentioning their very different prevalence levels.** The combined plot uses separate panels; it is descriptive, not evidence of equal data-generating conditions.
- OOF scores are used for model, cutoff and ensemble selection; selected OOF scores can be optimistic because the same OOF sample supports selection. The **untouched test set** is the final performance evaluation.
- A simulation "oracle" row uses known true test probabilities for reference only; it is not a model available for real fraud prediction and is never eligible for selection.
- Three Monte Carlo replicates are useful for an exploratory project but cannot establish strong simulation precision. Increase `repeats` in `00_shared_functions.R` if greater Monte Carlo certainty is needed.
- The complete script may take considerable CPU time and memory, particularly for the N=100,000 scenarios and the fraud dataset. Save intermediate files frequently (the scripts already do this after each scenario/split).
- This set has been **statically inspected** and reorganized from the attached working script. The full R analysis has **not been executed in this environment**, because an R interpreter and the local Windows dataset are not available here. Run and validate it in your RStudio environment before reporting numerical findings.

## Code and academic-integrity notes

The analysis remains a statistical learning exercise aligned with STAT 847, including its graduate-only ensemble requirement. Verify the output and interpretations yourself. Follow Dr. Li Xing's project AI-use disclosure requirements, including a disclosure statement and relevant citations in any submitted report.

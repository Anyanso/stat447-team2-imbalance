# Robustness check (Ambali Wadud)

An independent second implementation of the study, written separately from
the main code in `R/` and `analysis/`, with different packages and choices.
It is used as a robustness check: both studies reach the same conclusions
(see Slide 13 and the robustness section of the report).

| | Main study | This check |
| --- | --- | --- |
| Logistic model | lasso (`glmnet`, 10-fold CV) | ridge (`glmnet`, nested CV) |
| Random forest | `randomForest` | `ranger` |
| SMOTE | our own function | `themis::smote` |
| Rebalancing | 50:50 | partial (25% fraud, 4:1, square-root weights) |
| Cutoff for the threshold approach | Youden | F1 / F2 |
| Simulation replicates | 100 | 3 |
| Real-data test sets | 10 random splits | 1 random + 1 time-ordered split |
| Extra measures | | Brier score, log-loss |

## Files

- `Simulation_Experiment.R`: the simulation. Writes to `results/simulation/`.
- `Fraud_Real_Data.R`: the credit card data. Reads `data/creditcard.csv`
  and writes to `results/simulation/fraud_real_data/`.

Run both from the project folder (open the `.Rproj` first), for example
`source("robustness/Fraud_Real_Data.R")`. They need `data.table`, `ggplot2`,
`glmnet`, `ranger` and `themis`. The real-data script takes about 1.5-3 hours.

## Results on GitHub

- `results/simulation/tables/`: simulation results.
- `results/simulation/fraud_real_data/tables/`: real-data results
  (`fraud__all_metrics.csv` has every method on both splits).
- `results/simulation/fraud_real_data/plots/`: real-data figures.

The script also writes copies of the Kaggle data and the held-out
predictions; these are not on GitHub (they are re-created on each run).

The real-data results come from Wadud's run on Nov 11, 2026. That run used a
slightly newer version of `Fraud_Real_Data.R` than the one here (its settings
file lists a few extra options), so a rerun may differ slightly.

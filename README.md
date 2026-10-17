# STAT 447/847 Team 2: Class Imbalance

**Question:** when positives are rare, do the standard approaches (resampling,
class weights, threshold adjustment) improve *prediction*, or only the *metric*?

## Getting started (each teammate, once)

1. Clone the repo and open `stat447-team2-imbalance.Rproj` in RStudio.
2. Run `source("R/install_packages.R")`.
3. Download `creditcard.csv` from Kaggle (Credit Card Fraud Detection) and put it
   in `data/`. It is git-ignored; **never commit it** (about 150 MB).
4. Run `source("analysis/00_check_setup.R")`. It should end with `Setup OK.`

## Folder layout

| Folder / file | What goes there |
|---|---|
| `R/00_setup.R` | packages, seed (447), paths, study settings. Source it everywhere. |
| `R/gen_data.R` | simulation generator (Person 1) |
| `R/approaches.R` | the three approaches + two classifiers (all three people) |
| `R/evaluate.R` | the metric function and Bayes benchmark (Person 3) |
| `R/ensemble.R` | EasyEnsemble, STAT 847 extension (Person 3) |
| `analysis/` | scripts that run things and save results |
| `results/` | saved `.rds` results, so the report never reruns the simulation |
| `figures/` | saved plots |
| `report.qmd`, `slides.qmd` | Quarto report and presentation |

## Who owns what

| Person | Approach | Also owns |
|---|---|---|
| 1 | A. Resampling (over, under, SMOTE) | simulation generator |
| 2 | B. Class weights | real-data pipeline |
| 3 (STAT 847) | C. Threshold | evaluation + benchmark, EasyEnsemble |

## Function contract (do not change without telling the team)

```r
gen_data(n, prev, sep, beta0)                    # -> data frame: x1..x5, y, p.true
fit_and_predict(train, test, approach, classifier) # -> list(prob, cutoff)
evaluate(y, prob, cutoff = 0.5)                  # -> one-row tibble of metrics
easy_ensemble(train, test, classifier, B = 10)   # -> list(prob, cutoff)
```

`y` is always a factor with levels `c("0", "1")`, where `"1"` is the rare class.

## Rules

- **Syntax:** methods are written in the base-R style of the ISLR (R edition) labs.
  The simulation bookkeeping (design grid, summaries, figures) follows the class
  simulation deck.

- Seed is **447**, set once in `R/00_setup.R`.
- Resample or reweight the **training** data only. Never touch the test set.
- Tune λ and the cutoff using **training** data only.
- Work on your own branch; merge into `main` only when the code runs.
- Commit small and often, with a clear message.

## Git workflow

```bash
git checkout -b person1-resampling   # your own branch
# ... edit, then:
git add R/approaches.R
git commit -m "Add over-sampling approach"
git push -u origin person1-resampling
# open a pull request into main on GitHub
```

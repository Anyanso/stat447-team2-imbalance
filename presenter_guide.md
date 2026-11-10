# Team 2 Presenter's Guide: Class Imbalance

Nov 9, 2026 · Team 2

## How to use this guide

The whole talk answers one question: **the standard remedies for class imbalance make the model catch more of the rare class, but for logistic regression they do not make it rank cases any better; only flexible models like random forests sometimes gain real accuracy.**

Every slide below has the same parts: the idea in plain words, the formula if there is one, a worked example with our real numbers, what to say, and the questions you may get. Each part also names the lecture in our study notes ("Notes, Lecture 6 §6.8") so you can reread it the night before.

### Who presents what

| Slides | Speaker | Topic | Time |
| --- | --- | --- | --- |
| 1–6 | John Anyanso | Problem, question, simulation design, first two results | 5 min |
| 7–8 | Ambali Wadud | Gap to the best possible, threshold | 3 min |
| 9–11 | Nidhi Prajapati | Real fraud data, precision trade-off, reconciling | 4 min |
| 12–13 | Ambali Wadud | STAT 847 EasyEnsemble, robustness check | 3 min |
| 14–15 | Everyone | Conclusions, questions | 2 min + Q&A |


### Five words everyone must be able to define

- **Prevalence**: the share of cases that are the rare class. Fraud is 0.17% of transactions (Notes, Project Support §S.1).
- **Sensitivity**: of the real frauds, the share we catch (Notes, Lecture 6 §6.7).
- **Precision**: of the cases we flag, the share that really are fraud (Notes, Lecture 6 deep dive, "positive predictive value").
- **ROC-AUC**: the chance a random fraud gets a higher score than a random normal transaction; 0.5 is a coin flip, 1 is perfect (Notes, Lecture 6 §6.9).
- **PR-AUC**: like ROC-AUC but built from precision and sensitivity, so it focuses on the rare class. A random guess scores the prevalence, not 0.5.

### The one distinction that carries the talk

**Ranking versus cutoff.** A classifier gives every case a probability. *Ranking* is the order of those probabilities: are frauds near the top? ROC-AUC and PR-AUC measure ranking. The *cutoff* is the line we draw ("flag anything above 0.5"); sensitivity, specificity and precision depend on where it sits. Our finding is that the remedies mostly move the cutoff, not the ranking.

## Slides 1–3: the problem and our question

### Slide 1 — Title (John Anyanso, 15 seconds)

Say the title, the team, and the one-sentence answer from "How to use this guide". Telling the audience the answer up front makes every later slide easier to follow.

### Slide 2 — "99.8% accurate and useless" (John Anyanso)

**The idea.** When one class is very rare, a model can score high accuracy by never predicting it. Accuracy counts every correct answer equally, so the 99.83% of normal transactions swamp the 0.17% of frauds.

**Worked example.** The Kaggle data has 284,807 transactions and 492 frauds. A rule that says "not fraud" every time is right on 284,315 of them: 284,315 / 284,807 = 99.83%. It catches 0 of the 492 frauds, so its sensitivity is 0.

**Link to the notes.** This is the same trap as the Default data in Lecture 6 §6.6: LDA's error rate was 2.75%, but "always No" already got 3.33%, and LDA missed 252 of 333 defaulters (75.7%). The notes call this the null classifier.

**What to say.** "Accuracy rewards ignoring the rare class. So the question is not how to raise accuracy, but how to find the frauds."

### Slide 3 — Our question (John Anyanso)

**The idea.** People usually reach for one of three fixes. We test whether each one makes the model genuinely better, or only makes a number look better.

| Remedy | What it does | Plain-language picture |
| --- | --- | --- |
| A. Resampling | Change the training data to 50:50. Over-sampling copies frauds; under-sampling throws away normal rows; SMOTE makes new synthetic frauds between real ones | Re-printing the rare flashcards so you study them more |
| B. Class weights | Keep the data, but each fraud counts more in the fit | Marking the rare questions worth more points |
| C. Threshold | Keep the model; flag a case at a lower probability than 0.5 | Same exam, lower pass mark |

**SMOTE in one line.** Pick a fraud, pick one of its 5 nearest fraud neighbours, and create a new point somewhere on the line between them.

**"Improve prediction" versus "improve the metric".** Prediction improves if frauds move higher in the ranking: ROC-AUC or PR-AUC goes up. The metric improves if sensitivity at the chosen cutoff goes up. You can raise sensitivity without changing the ranking at all, just by flagging more cases.

**The two classifiers.**

- **Lasso logistic regression** (Notes, Lecture 5 and Lecture 11): logistic regression with a penalty that shrinks weak coefficients to zero. We chose λ by 10-fold cross-validation with `cv.glmnet` and used `lambda.min` (Notes, Lecture 11 §11.10–11.11).
- **Random forest** (Notes, Lecture 15 §15.6 and the bootstrap extra): 200 trees, each grown on a bootstrap sample, votes averaged. Averaging fixes the instability of single trees.

We picked one rigid, correctly shaped model and one flexible model on purpose, because the remedies behave differently for each.

**Likely question.** *"Why these two classifiers?"* Logistic regression is the model the handout and Lecture 5 use for a yes/no outcome. The forest is the most common flexible alternative. Between them they cover "simple and right" and "flexible and noisy".

## Slide 4: the simulation design (John Anyanso)

**The idea.** With real data we can never know how good the best possible model is. In a simulation we write the true rule ourselves, so we have an answer key. The notes put it as "you play nature" (Notes, Project Support §S.3).

**The true model.** Each simulated person has 5 predictors x₁ to x₅, drawn from a standard normal. The chance of being positive is:

```latex
P(Y = 1 \mid x) = \frac{e^{\eta}}{1 + e^{\eta}}, \qquad \eta = \beta_0 + \text{sep} \times (x_1 - x_2 + 0.5\,x_3 - 0.5\,x_4 + 0 \cdot x_5)
```

This is the logistic function from Notes, Lecture 5 §5.5. On the log-odds scale it is a straight line (§5.6). x₅ has slope 0: it is pure noise, which tests whether the lasso drops it.

**Why only β₀ changes.** β₀ is the intercept. Lowering it shifts everyone's log-odds down by the same amount, so positives become rarer, but the slopes, the shape of the problem, stay the same. That way "rarer" is not mixed up with "harder". We found the β₀ for each prevalence by bisection on a million-row sample.

**Worked example.** At high separation our code found β₀ = 0.00 for 50% prevalence, −2.48 for 20%, −4.90 for 5% and −7.04 for 1%. Take an average person (all x = 0): at 50% their chance is e⁰/(1+e⁰) = 50%; at 1% it is e⁻⁷·⁰⁴/(1+e⁻⁷·⁰⁴) ≈ 0.09%. Only people with large x₁ or small x₂ reach a real chance of being positive.

**Separation (sep).** It multiplies all the slopes. sep = 0.5 means the predictors barely tell the classes apart (low); sep = 1.5 makes them much more informative (high).

**The design grid.**

| Setting | Values |
| --- | --- |
| Prevalence | 50%, 20%, 5%, 1% |
| Training size n | 1,000 and 20,000 |
| Separation | low 0.5, high 1.5 |
| Replicates per setting | 100 |
| Test set | fresh 20,000 rows each replicate |

4 × 2 × 2 = 16 settings, 100 replicates each = 1,600 data sets. Each is fit by 2 classifiers × 6 approaches = 12 methods, so 19,200 model fits.

**Why two sample sizes.** At 1% prevalence, n = 1,000 gives only about 10 positives: a small *count*. n = 20,000 gives about 200: still a small *proportion*, but enough to learn from. This separates "too few examples" from "too rare".

**The Bayes benchmark.** Because we know the true probabilities, we can score them as if they were a model. No method can beat them. This is the Bayes classifier from Notes, Lecture 5 §5.4: "a target, not a method". The distance between a method and Bayes is the gap a better method could close (Notes, §S.3, "report the gap").

**Following the four mistakes (Notes, §S.8).**

1. One seed, 447, set once and recorded.
2. 100 replicates, and every claim checked against its Monte Carlo standard error.
3. λ and the threshold chosen on training data only; the test set is used once.
4. Every mean shown with error bars.

**Fair comparison.** All 12 methods see the *same* data sets in each replicate, so we can compare each remedy with the baseline on identical data (a paired comparison).

**Likely question.** *"Isn't a logistic truth too easy for logistic regression?"* Yes, and that's deliberate. It is the best case for logistic regression: if the remedies can't help a correctly specified model, that tells us something. Slide 11 shows what changes when the truth is not logistic.

## Slides 5–7: simulation results

### How to read all three figures

Every point is a **paired change**: in each replicate, the remedy's score minus the baseline's score *on the same data set*, averaged over replicates. Zero means "no different from doing nothing". The bars are ±2 MCSE.

**MCSE** (Monte Carlo standard error) = sd / √B, where B is the number of replicates (Notes, Project Support §S.6). It says how precisely the simulation pinned down the average. Rule from the notes (§S.8, Mistake 2): if a difference is smaller than about 2 MCSE, we have found nothing and must say so.

**Worked example of MCSE.** Forest under-sampling at 1% prevalence raised ROC-AUC by 0.048 with MCSE 0.002. 0.048 is 24 MCSEs from zero: a real effect, not luck.

### Slide 5 — Result 1: sensitivity jumps (John Anyanso)

**What the figure shows.** For logistic regression, every remedy raises sensitivity, by +0.44 to +0.72 at the rarer prevalences. For the forest, everything except class weights raises it too.

**Why.** A model trained on 1% positives learns that positives are rare, so almost no one gets a probability above 0.5. Rebalancing to 50:50 inflates every probability, so many more cases cross 0.5. Lowering the threshold does the same thing from the other side.

**What to say.** "Judged by sensitivity, every remedy works. Most papers stop here. We asked whether the model got better, or whether it just flags more people."

### Slide 6 — Result 2: the ranking barely moves (John Anyanso)

**What the figure shows.** Change in ROC-AUC. For logistic regression every remedy sits on zero. For the forest, under-sampling gains +0.048 at 1% (MCSE 0.002); class weights lose about 0.018.

**Why logistic regression can't improve: the key argument.**

1. Rebalancing the training data to 50:50 changes how common positives look, and nothing else.
2. In a logistic model, how common positives are is controlled by β₀ alone. That's exactly how we created the imbalance on slide 4.
3. So rebalancing changes only the fitted intercept; the fitted slopes stay (nearly) the same.
4. Adding a constant to everyone's log-odds doesn't change who is above whom. The ranking, and so ROC-AUC, is unchanged.

This is a known result for case-control sampling (Prentice & Pyke, 1979): sampling on the outcome only shifts the intercept of a logistic model.

**Worked example.** On slide 8, the baseline and over-sampling both have ROC-AUC 0.924, with sensitivity 0.23 versus 0.88. Same ranking, different cutoff.

**Why the forest is different.** A forest has no intercept to shift. Its trees split on the data they see, and at 1% with n = 1,000 each tree sees about 10 positives. Under-sampling makes every tree look closely at the positives, which changes the splits and so the ranking. Forests are unstable, high-variance learners (Notes, Lecture 15 §15.6), so changing what each tree sees matters.

**Why class weights hurt the forest.** `randomForest`'s `classwt` only changes the split criterion slightly; it does not change what each tree samples. In our runs it lowered both sensitivity and ROC-AUC. We report that honestly.

**One small exception for logistic.** At 1% with n = 1,000, over-sampling and SMOTE gained +0.015 ROC-AUC. The reason: with about 10 positives, the baseline lasso shrank every slope to zero in 12.5% of replicates, a flat model with AUC 0.5. Rebalancing stopped that collapse. It's a small-count rescue, not a better model.

### Slide 7 — Result 3: nothing closes the gap (Ambali Wadud)

**What the figure shows.** For each method, Bayes PR-AUC minus the method's PR-AUC. Lower is better; zero would mean as good as the true probabilities.

**Why PR-AUC.** ROC-AUC can look great when positives are rare, because specificity is easy when almost everything is negative. PR-AUC uses precision, which collapses if the flagged cases are mostly false alarms. A random guess gets PR-AUC equal to the prevalence (0.01 at 1%), not 0.5.

**The message.** No remedy brings any method closer to Bayes than the baseline does. Resampling makes the forest's PR-AUC *worse*. The threshold line sits on top of the baseline, because the threshold uses the same model; only the cutoff differs, and PR-AUC doesn't depend on the cutoff.

**Link to the notes.** This is the "report the gap" idea (§S.3): raw scores fall as positives get rarer for every method; the gap tells you how much a better method could still gain.

**Likely question.** *"If PR-AUC got worse, why does sensitivity look better?"* Because sensitivity only counts frauds caught. Precision counts the false alarms you paid to catch them, and PR-AUC tracks both.

## Slide 8: the threshold gets the same gains, for free (Ambali Wadud)

**The idea.** If rebalancing only moves the cutoff, we can move the cutoff directly and skip the rebalancing. That keeps the model's probabilities honest.

**The table, one replicate (logistic, 5%, n = 20,000).**

| Method | Sensitivity | Specificity | ROC-AUC | Mean predicted P(fraud) |
| --- | --- | --- | --- | --- |
| Baseline, cutoff 0.5 | 0.23 | 0.995 | 0.924 | 0.05 |
| Over-sampling, cutoff 0.5 | 0.88 | 0.82 | 0.924 | 0.27 |
| Baseline model, cutoff 0.046 | 0.88 | 0.81 | 0.924 | 0.05 |

**How to read it, row by row.**

- Baseline: ranks well (0.924) but almost no one passes 0.5, so it catches only 23%.
- Over-sampling: catches 88%, but it now says the average person has a 27% chance of fraud when the truth is 5%. Its probabilities are wrong.
- Threshold: the *same model* as the baseline with a lower cutoff. It catches 88%, with the same specificity as over-sampling, and its average probability is still the true 5%.

All three have ROC-AUC 0.924: one ranking, three different cutoffs.

**Calibration in plain words.** A model is calibrated if, among people it gives 5%, about 5% really are positive. The mean predicted probability should match the true prevalence. Resampling and weights break this; the threshold doesn't. That matters whenever someone uses the probability itself, for example to set an insurance price or to rank how urgently to investigate.

**How we chose the cutoff (Youden index).** For each candidate cutoff we compute sensitivity + specificity − 1 and keep the cutoff where it is largest. That's the point on the ROC curve farthest above the diagonal (Notes, Lecture 6 §6.9).

**No peeking.** The cutoff is chosen on out-of-fold training predictions: `cv.glmnet(..., keep = TRUE)` keeps each row's prediction from the fold it was left out of. For the forest we use out-of-bag votes, from the trees that did not see that row (Notes, bootstrap extra: about 37% of rows are left out of each bootstrap sample). The test set is never used to tune (Notes, §S.8, Mistake 3; Lecture 9 §9.7).

**The cost link (Notes, Lecture 6 §6.8).** A cutoff t is the best rule when

```latex
\frac{\text{cost of a missed fraud}}{\text{cost of a false alarm}} = \frac{1 - t}{t}
```

Worked example: t = 0.046 means (1 − 0.046) / 0.046 ≈ 21. Choosing that cutoff says "missing a fraud is about 21 times worse than a false alarm". The notes' Default example used t = 0.2, a 4-to-1 ratio. So the threshold is the honest way to put real costs into the decision.

**What to say.** "Resampling and the threshold reach the same place. The threshold gets there without retraining and without breaking the probabilities."

**Likely question.** *"This is one replicate. Does it hold in general?"* Yes: across all 100 replicates the threshold's ROC-AUC change is exactly zero (same model), and its sensitivity gain is in the same range as resampling's (slide 5).

## Slides 9–11: real data, the trade-off, and reconciling (Nidhi Prajapati)

### Slide 9 — Real data: 284,807 transactions, 10 splits

**The data.** Kaggle credit card transactions from two days in September 2013: 284,807 rows, 492 frauds (0.17%). 28 of the predictors (V1–V28) are anonymised principal components, so we can't interpret them; the other two are Time and Amount.

**What we did.**

1. Split 70% training / 30% test, **stratified**, so both parts keep 0.17% fraud (about 344 frauds to train on, 148 to test on).
2. Standardised Time and Amount using the *training* mean and sd only, so no test information leaks in (Notes, Lecture 9 §9.7: preprocessing is part of fitting).
3. Ran all 12 methods, then repeated with 10 different random splits. Bars are ±2 SE across splits.

**Two practical problems we solved.** glmnet's default λ path did not converge, because some predictors nearly separate fraud from normal; we used a finer path (200 values down to 0.001 of the maximum). And each forest tree draws 20,000 rows instead of all 199,000, to keep run time reasonable.

**Results.**

| Method | ROC-AUC | PR-AUC | Sensitivity | Precision |
| --- | --- | --- | --- | --- |
| Always "not fraud" | 0.500 | 0.005 | 0.00 | — |
| Logistic baseline | 0.977 | 0.764 | 0.61 | 0.88 |
| Logistic + over-sampling | 0.979 | 0.717 | 0.90 | 0.07 |
| Logistic + under-sampling | 0.977 | 0.574 | 0.90 | 0.07 |
| Logistic + SMOTE | 0.977 | 0.713 | 0.91 | 0.06 |
| Logistic + class weights | 0.979 | 0.694 | 0.90 | 0.08 |
| Logistic + threshold | 0.977 | 0.765 | 0.89 | 0.06 |
| Forest baseline | 0.949 | 0.797 | 0.72 | 0.91 |
| Forest + over-sampling | 0.977 | 0.819 | 0.83 | 0.78 |
| Forest + under-sampling | 0.977 | 0.737 | 0.90 | 0.05 |
| Forest + SMOTE | 0.983 | 0.828 | 0.85 | 0.60 |
| Forest + class weights | 0.945 | 0.775 | 0.45 | 0.90 |
| Forest + threshold | 0.944 | 0.795 | 0.87 | 0.14 |

Means over the 10 splits. Every logistic row has ROC-AUC 0.977–0.979: the ranking doesn't move. "Always not fraud" gets PR-AUC 0.005, close to the 0.0017 prevalence, which is what a useless ranking scores.

**What to say.** "For logistic regression, the real data agrees with the simulation: the ranking doesn't move. Resampling actually lowers PR-AUC, by 0.05 to 0.19. For the forest, the real data disagrees: SMOTE and over-sampling improve both ROC-AUC and PR-AUC."

### Slide 10 — Catching more fraud costs precision

**The idea.** There's no free lunch at a fixed ranking: moving the cutoff to catch more frauds means flagging more innocent transactions too.

**Worked example.** Every logistic remedy that pushes sensitivity to about 90% drops precision to about 6%. Precision 6% means 6 in 100 flagged transactions are fraud, so about 94 / 6 ≈ 15 false alarms for every real fraud. Compare the logistic baseline: 61% caught at 88% precision.

**The best balance.** Forest + SMOTE catches 85% at 60% precision: fewer than one false alarm per fraud found. That's the only method that moves *up and to the right* on this plot, which is what a better ranking looks like.

**Link to the notes.** Lecture 6 §6.8: lowering the Default threshold from 0.5 to 0.2 cut missed defaulters from 252 to 138 but raised false alarms from 23 to 235. Same trade-off. Which point is right depends on costs, not statistics.

### Slide 11 — Reconciling simulation and real data

**The puzzle.** At the closest simulated setting (1%, n = 20,000, high separation), forest + SMOTE *lost* 0.084 PR-AUC. On the real data it *gained* 0.031. Why the opposite sign?

**Three explanations; say all three.**

1. **The real model isn't logistic.** In our simulation the truth was logistic, so rebalancing only moved the intercept. When the model is wrong, re-weighting the classes also changes which errors the fit cares about, so it can bend the slopes (Prentice & Pyke, 1979 show the intercept-only result needs a correct logistic model).
2. **Frauds are probably clustered.** Real frauds likely come in small groups of similar transactions. A forest can carve out those pockets once it sees enough examples; SMOTE fills them in. Our simulated positives were spread smoothly.
3. **Our own set-up.** With 20,000 rows per tree and 0.17% fraud, a baseline tree sees only about 20,000 × 0.0017 ≈ 34 frauds. After SMOTE balances the data, a tree sees about 10,000. Part of the gain is fixing a "too few frauds per tree" problem our simulated forests didn't have.

**What to say.** "Point 3 is about our design, and we want to be upfront about it. The simulation tells us what happens when the model is right; the real data shows what can happen when it isn't."

**Likely question.** *"So should we trust the simulation or the real data?"* Both answer different questions (Notes, §S.2): real data says which method wins here; the simulation says why, and how far from perfect each method is.

## Slides 12–13: STAT 847 ensemble and robustness (Ambali Wadud)

### Slide 12 — STAT 847: EasyEnsemble

**The problem it solves.** Under-sampling throws away most of the normal rows. At 1% with n = 20,000 that's about 19,600 rows discarded, plus a lot of noise from which 200 happened to be kept.

**The idea.** Under-sample 10 times, each time keeping a *different* random set of normal rows. Fit a model to each balanced set. Average the 10 predicted probabilities.

```latex
\hat p_{\text{ens}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat p_b(x), \qquad B = 10
```

Each member sees balanced data; together they see most of the normal rows.

**Why averaging helps (Notes, Lecture 15 §15.6).** Averaging B independent estimates divides the variance by B. It's the same reason a random forest beats a single tree. It helps when the members are noisy; it can't fix bias.

**Fair comparison.** Forest members get 200 / 10 = 20 trees each, so the ensemble has the same 200 trees as one forest. In each replicate we compare the ensemble against the *best* single approach for that setting, not just the baseline. A difference counts only if it is bigger than 2 MCSE and at least 0.005.

**Results.**

- **Forest:** beats the best single approach on ROC-AUC and balanced accuracy in 11 of 12 imbalanced settings. The gain grows as imbalance worsens, up to +0.028 ROC-AUC and +0.032 balanced accuracy at 1%, n = 1,000.
- **Logistic:** never better. At 1%, n = 1,000 it's worse (−0.012 ROC-AUC, −0.036 PR-AUC), because each member sees only about 10 positives and 10 negatives.
- **PR-AUC:** mixed for both.
- **Real data:** ties the best single method on ROC-AUC and balanced accuracy; loses about 0.08 PR-AUC to SMOTE and the threshold.

**Balanced accuracy** = (sensitivity + specificity) / 2. Unlike accuracy, "always not fraud" scores 0.5 on it, not 99.8%.

**What to say.** "Averaging helps an unstable learner, the forest. It doesn't help a stable, correct one, logistic regression, which has little variance to remove."

**Likely question.** *"Why not boosting?"* Boosting is the other ensemble from the Oct 30 lecture (ISLR 8.2), but it fits members one after another, so it doesn't fix the information thrown away by under-sampling. EasyEnsemble is the ensemble designed specifically for imbalance (Liu, Wu & Zhou, 2009).

### Slide 13 — Robustness: an independent second implementation

**The idea.** One team member independently wrote a second simulation with different tools and choices. If both studies reach the same conclusions, those conclusions don't depend on our particular code.

**What was different in the second study.** Different R packages (`ranger` for forests, `themis` for SMOTE); ridge instead of lasso, with nested cross-validation; *partial* rebalancing (over-sample to 25%, under-sample to 4:1, square-root weights) instead of 50:50; F1/F2 thresholds instead of Youden; a cross-validated blend of logistic and forest as the ensemble; 3 replicates.

**How to read the plot.** Each dot is one remedy × prevalence × classifier at n = 20,000. x = change vs baseline in our study; y = change in the second study. Dots on the dashed line mean the two studies agree exactly.

- Every logistic dot sits at (0, 0): both say "no change in ranking".
- Forest dots mostly share our quadrant (both lose PR-AUC), but the second study's drops are smaller because it rebalanced only partway.

**Numbers.** We reran the second study from its source code: results at 20% and 50% reproduce to every digit. Same direction as ours in 91% of ROC-AUC comparisons and 78% of PR-AUC comparisons (a change under 0.005 counts as "no change"). Its blended ensemble never beat its best single model.

**What to say.** "With only 3 replicates, this is a check on our conclusions, not a separate set of findings. And it agrees."

## Slides 14–15: conclusions and closing (everyone)

**Slide 14: each person states the conclusion from their own part.** Suggested split:

1. **John Anyanso:** "For logistic regression, the remedies change the operating point, not the ranking. Sensitivity rises to about 90%, but no remedy improves ROC-AUC or PR-AUC."
2. **Nidhi Prajapati:** "Forests are different. Under-sampling, and on the real data SMOTE, can improve the ranking when the rare class is very rare."
3. **Ambali Wadud:** "Accuracy is the wrong target. Judge the ranking by PR-AUC, then pick the cutoff from the real costs of a missed fraud versus a false alarm."
4. **Ambali Wadud:** "For logistic models, prefer the threshold: the same sensitivity, honest probabilities, no retraining."
5. **Ambali Wadud (STAT 847):** "EasyEnsemble helps forests, up to +0.03 ROC-AUC at 1%, but not logistic regression, and not PR-AUC."

**The one-line answer to the handout.** "Do the standard remedies improve prediction, or only the metric? For a correctly specified logistic model, only the metric. For flexible models on very rare classes, sometimes prediction too."

**Slide 15: thank you and questions.** Point to the GitHub repository: everything reruns from seed 447. Then take questions, using the next section.

**Tip for Q&A.** Answer in one sentence first, then give one number. If you don't know, say which slide or part of the report covers it and offer to follow up.

## Likely questions and answers

Short answer first, then the reason. The note reference tells you where to review.

**Q1. Why did you use the logistic model to generate the data?** The outcome is yes/no, and logistic regression is the standard model for that (Notes, Lecture 5). It also lets us control prevalence with one number, β₀, while the slopes stay fixed.

**Q2. Isn't it unfair that the truth matches one of your classifiers?** It's the best case for logistic regression, on purpose. If the remedies can't help even when the model is right, that's informative. The real data is the test of what happens when the model is wrong (slide 11).

**Q3. Why lasso and not plain `glm`?** The lasso picks λ by cross-validation and can drop the noise predictor x₅ (Notes, Lecture 11). It also stays stable with only about 10 positives, where plain `glm` can fail to converge.

**Q4. How did you choose λ, and did the test set influence it?** 10-fold cross-validation on the training data, with folds stratified so every fold has positives; we used `lambda.min` (Notes, Lecture 11 §11.11). The test set was used once, for scoring (Notes, §S.8, Mistake 3).

**Q5. Why 100 replicates? Is that enough?** Our key differences are many MCSEs from zero, for example +0.048 with MCSE 0.002. The rule from the notes is that MCSE must be clearly smaller than the differences you claim (§S.8, Mistake 2). Where a difference is under 2 MCSE, we say "no difference".

**Q6. Why ROC-AUC and PR-AUC instead of accuracy?** Accuracy rewards ignoring the rare class (slide 2). Both AUCs measure the ranking over all cutoffs, so they show whether the model itself improved (Notes, Lecture 6 §6.9). PR-AUC is more demanding when positives are rare because it counts false alarms through precision.

**Q7. What's the difference between sensitivity and precision?** Sensitivity: of the real frauds, how many we caught. Precision: of the cases we flagged, how many were fraud. On the real data, logistic + threshold has sensitivity 0.89 but precision 0.06.

**Q8. Why does rebalancing not change logistic regression's ranking?** Rebalancing only changes how common positives look, which in a logistic model is the intercept. Shifting the intercept adds the same amount to everyone's log-odds, so the order doesn't change (Prentice & Pyke, 1979).

**Q9. If the threshold is so good, why does anyone resample?** Resampling can help models that have no intercept to shift, like forests, and it rescues models that collapse with very few positives. For a logistic model, the threshold does the same job and keeps the probabilities calibrated.

**Q10. How do you choose the threshold in practice?** From costs: a cutoff t means a missed fraud costs (1 − t)/t times a false alarm (Notes, Lecture 6 §6.8). We used the Youden index on training predictions because we had no real costs.

**Q11. Why did class weights hurt the random forest?** `randomForest`'s `classwt` only changes the split criterion a little and doesn't change what each tree sees. The real-data sensitivity fell to 0.45. Other packages implement weights differently, so this is a finding about this implementation.

**Q12. Why did the simulation and real data disagree for the forest?** Three reasons (slide 11): the real model isn't logistic, real frauds are probably clustered, and our 20,000-row trees saw only about 34 frauds each at baseline.

**Q13. Isn't SMOTE leaking information?** No. SMOTE was applied only to the training data, after the split. Synthetic points never enter the test set.

**Q14. Why only 10 splits for the real data?** Each split fits 12 methods on about 199,000 rows; 10 splits already make the standard errors small enough to separate the main differences. The bars on slide 9 show that.

**Q15. What would you do with more time?** A non-logistic simulated truth (clusters or interactions) to test the explanation on slide 11, real fraud costs to pick the threshold, and boosting as a third classifier.

**Q16. Is the result reproducible?** Yes. One seed, 447, set once; per-replicate seeds derived from it; all code and results on GitHub. Rerunning gives identical numbers (Notes, §S.8, Mistake 1).

## Cheat sheet

Print this page and keep it at the podium.

### Numbers to know by heart

| What | Number |
| --- | --- |
| Real data size | 284,807 transactions, 492 frauds (0.17%) |
| "Always not fraud" | 99.83% accuracy, 0 frauds caught |
| Simulation size | 16 settings × 100 replicates = 1,600 data sets; 19,200 fits |
| True intercepts, high separation | β₀ = 0.00 (50%), −2.48 (20%), −4.90 (5%), −7.04 (1%) |
| Logistic sensitivity gain (sim) | +0.44 to +0.72; ROC-AUC change ≈ 0 |
| Forest under-sampling, 1% (sim) | +0.048 ROC-AUC, MCSE 0.002 |
| Forest class weights (sim) | −0.018 ROC-AUC |
| Threshold example | sensitivity 0.23 → 0.88 at cutoff 0.046, ROC-AUC 0.924 throughout |
| Real logistic baseline | ROC 0.977, PR 0.764, sensitivity 0.61, precision 0.88 |
| Real logistic + threshold | sensitivity 0.89, precision 0.06 (about 15 false alarms per fraud) |
| Real forest + SMOTE | ROC 0.983, PR 0.828, sensitivity 0.85, precision 0.60 |
| Sim vs real, forest + SMOTE PR-AUC | −0.084 vs +0.031 |
| Frauds per tree, real data | about 34 at baseline, about 10,000 after SMOTE |
| EasyEnsemble, forest | wins 11 of 12 imbalanced settings; up to +0.028 ROC-AUC |
| Robustness agreement | 91% ROC-AUC, 78% PR-AUC same direction |

### Definitions and where they are in the notes

| Term | One-line meaning | Notes |
| --- | --- | --- |
| Logistic function | p = e^η / (1 + e^η); always between 0 and 1 | Lecture 5 §5.5 |
| Log-odds | log(p / (1 − p)) = η, a straight line in x | Lecture 5 §5.6 |
| Bayes classifier | uses the true probabilities; no method can beat it | Lecture 5 §5.4 |
| Confusion matrix | table of predicted vs true classes | Lecture 6 §6.6 |
| Sensitivity | caught frauds / all frauds | Lecture 6 §6.7 |
| Specificity | cleared normals / all normals | Lecture 6 §6.7 |
| Precision | real frauds / all flagged | Lecture 6 deep dive |
| Threshold and cost | cutoff t ⇔ missed-fraud cost = (1 − t)/t × false-alarm cost | Lecture 6 §6.8 |
| ROC-AUC | chance a fraud outranks a normal; 0.5 = coin flip | Lecture 6 §6.9 |
| PR-AUC | area under precision vs sensitivity; random = prevalence | — |
| Balanced accuracy | (sensitivity + specificity) / 2 | — |
| Youden index | sensitivity + specificity − 1; we pick the cutoff that maximises it | Lecture 6 §6.9 |
| Cross-validation | fit on k − 1 folds, score the left-out fold, repeat | Lecture 9 §9.1, §9.5 |
| Lasso, λ | penalty that shrinks coefficients, some to exactly 0; λ chosen by CV | Lecture 11 |
| Random forest | many trees on bootstrap samples, averaged; fixes tree instability | Lecture 15 §15.6 |
| Out-of-bag | the \~37% of rows a tree didn't see; a free test set | Bootstrap extra §B.6 |
| MCSE | sd / √B; how precise a simulated average is | Project Support §S.6 |
| The gap | Bayes score minus method score; what a better method could gain | Project Support §S.3 |
| SMOTE | new synthetic positives on lines between a positive and its nearest positive neighbours | — |
| EasyEnsemble | average of B models, each on a different balanced under-sample | — |

Terms marked — aren't in the lecture notes; they come from the project handout and our report.

### References to cite if asked

- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning with Applications in R*, 2nd ed.: §4.3 (logistic regression), §4.4.2 (sensitivity, ROC), §6.2 (lasso), §8.2 (random forests).
- Prentice & Pyke (1979), "Logistic disease incidence models and case-control studies", *Biometrika*: case-control sampling shifts only the intercept.
- Chawla et al. (2002), "SMOTE: Synthetic Minority Over-sampling Technique", *JAIR*.
- Liu, Wu & Zhou (2009), "Exploratory undersampling for class-imbalance learning", *IEEE Trans. SMC-B*: EasyEnsemble.

---

*Living version (editable, with comments): https://claude.ai/artifact/G9ELDrGEqyEw8zg4u47cLL*

# Forward Goal Involvement — Analytic Task 4 of 4

**Author:** Shreejan Shrestha (Shriz)
**Unit:** HIT140 Foundations of Data Science (S226) — Assessment 2, Group Project Presentation

Analytic question: among forward players registered in FIFA World Cup 2026 squads, is
there a statistically significant difference in average goal involvement (goals +
assists) between forwards whose teams advanced to the knockout stage versus forwards
whose teams were eliminated in the group stage?

## Folder structure

```
forward_analysis/
├── README.md
├── data/
│   ├── raw/            source CSVs (auto-downloaded on first run — see below)
│   │   ├── squads_and_players.csv
│   │   ├── match_events.csv
│   │   ├── matches.csv
│   │   └── tournament_stages.csv
│   └── processed/
│       └── sample_forward_goal_involvement.csv   the drawn stratified sample (n=188)
├── figures/
│   ├── descriptive_overview.png    population/sample, distribution shape, group comparison
│   └── results_comparison.png      group means with 95% CI (raw count + per-match rate)
├── notebooks/
│   └── task_forward_goal_involvement.ipynb   same analysis, cell-by-cell with output baked in
└── src/
    └── task_forward_goal_involvement.py      the analysis script
```

## Running it

Either run the script or the notebook — both do the same thing and write to the same
`data/processed/` and `figures/` folders.

```bash
cd src
python3 task_forward_goal_involvement.py
```

or open `notebooks/task_forward_goal_involvement.ipynb` and run all cells.

**No manual data setup needed.** If `data/raw/` is empty, the script downloads the four
required CSVs itself, pinned to a fixed commit of the source dataset
(github.com/mominullptr/FIFA-World-Cup-2026-Dataset, CC0) so the results stay exactly
reproducible.

Requires: `pandas`, `numpy`, `scipy`, `matplotlib` (`pip install pandas numpy scipy matplotlib`).

## Method summary

1. **Data wrangling** — merge player records, match events, and match/stage tables to
   build two variables: whether each forward's team reached the knockout stage, and
   their goal involvements (goals + assists) for the tournament.
2. **Sampling** — population = all 313 registered forwards; stratified random sample,
   60% of each group (n = 188).
3. **Descriptive statistics** — mean/median/std/range per group, visualised in
   `descriptive_overview.png`.
4. **Confidence interval** — 95% CI for the population mean goal involvement.
5. **Two-sample t-test** — Welch's t-test (unequal variances) comparing the two groups.
6. **Robustness check** — the same comparison re-run on a per-match rate
   (goal involvements ÷ team matches played), to confirm the effect isn't just
   "knockout teams play more games."

Result: forwards on knockout-stage teams recorded significantly higher goal
involvement than forwards eliminated in the group stage (Welch's t = 5.48, p < 0.0001),
and the effect holds up on a per-match basis too (t = 4.71, p < 0.0001).

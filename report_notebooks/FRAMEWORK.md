# Report notebook framework (all tasks)

Every task notebook (`T1`–`T5`) uses the same 8 sections, so a reader can jump to the same place in each one.
The structure follows the CRISP-DM order (understand → data → model → evaluate). Inside each EDA finding it uses the
**Claim → Evidence → Implication** pattern: one sentence, then one cell that proves it, then what it changes in the method.

| # | Section | Contents | Length |
|---|---|---|---|
| 0 | **At a glance** | summary card: task, metric, final method, score, the 1–3 insights that mattered | 1 table |
| 1 | **Problem & metric** | what is predicted, the metric formula, what the metric rewards | ≤ 5 lines |
| 2 | **Data** | rows, columns, column groups, missing values | 1 code cell |
| 3 | **EDA findings** | 3.1, 3.2, …: each **Finding** (bold claim) → evidence cell (number or plot) → **→ Implication** | 1 claim + 1 cell each |
| 4 | **Method** | pipeline table (step, technique, why) and the reasons for the chosen family of methods | 1 table + bullets |
| 5 | **Implementation** | the final pipeline run step by step from raw data, ending with an assert that it reproduces the submitted predictions | short cells |
| 6 | **Validation & results** | CV / leaderboard evidence for each design decision | 1 table |
| 7 | **What did not work** | method, result, and the reason it fails | 1 table |
| 8 | **Output & reproducibility** | files, scripts, how to rerun | ≤ 5 lines |

Writing rules
- **Text:** English, concise. Every number shown is computed in the notebook or cited with its source (CV log, LB submission).
- **Execution:** code cells run top to bottom from the repo root (`python report_notebooks/build_tN.py` rebuilds and executes).
- **Plots:** one idea per plot, with axes labelled.
- **Final check:** the notebook ends by matching its predictions to the submitted file, so the report and the submission cannot drift apart.

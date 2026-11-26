# Handoff: CPE342 technical report (team ML003)

You are writing the technical report and tidying the source code for a finished Kaggle competition.
The modelling is done. Do not train models, do not submit to Kaggle, and do not change any file in `submissions/`.

## 1. Deliverables and deadlines

The course brief is in `docs/CPE342_canva_instruction_text.txt` (page 6 has the checklist).

| part | weight | what |
|---|---|---|
| Kaggle leaderboard | 70% | done; nothing to do |
| Technical report | 20% | one document, posted on LEB2 |
| Source code | 10% | GitHub repo with one clearly labelled folder per task (Task1 ... Task5) |

**Deadlines conflict.** The brief says Kaggle closes 26 Nov and the report is due 28 Nov. Our notes say 24 Nov 13:30 and 30 Nov.
Ask the user which is right before planning; assume 28 Nov until told otherwise.

**Still open (ask the user, do not guess):**
- report language: Thai or English (the brief allows either)
- format: Word/PDF/Google Doc, or one notebook per task (see `report_notebooks/FRAMEWORK.md`)
- each team member's role, for the Team Collaboration section

## 2. Required report structure (from the brief)

1. **Methodology**, one subsection per task (Task 1-5). Each subsection covers:
   EDA findings · data preprocessing · model design · evaluation and results · insights from model behaviour ·
   failed experiments and mistakes · domain interpretation (game data and business impact).
2. **Team Collaboration**: role of each member, key takeaways from teamwork.
3. **Conclusion**: lessons from model building and evaluation, insights on ML in the game industry, future improvements.

Suggested order of work: T4 (largest, no draft yet) → T1 → T3 → T2 → T5 → sections 2-3 → GitHub folders.

## 3. Sources of truth

- `CLAUDE.md`: full experiment log and leaderboard history (section 4). Every number in the report should come from here or
  from re-running a script. Read the entries from "Data re-release (12 Nov)" onward; earlier rows use the old data.
- `work/insight/INSIGHTS.md`: data-generator analysis for T1, T2, T3, T5 (good material for EDA and domain sections).
- `report_notebooks/`: 12-13 Nov drafts. `FRAMEWORK.md` (8-section layout, reusable), `T3_spending_regression.ipynb`,
  `T5_anomaly_detection.ipynb` + `T5_methodology.md`, `build_t1.py`, `build_t2.py`. **Stale:** T2 and T5 were written on the
  old data (before the 12 Nov re-release); T4 has no draft. Check every number against `CLAUDE.md` before reusing it.
- `t*/research/REPORT.md` and `t*/research/*_techniques_report.md`: cloud-agent research notes (ideas, not results).

## 4. Final submissions

| file | public LB | role |
|---|---|---|
| `submissions/FINAL/sub_v85a_soup_trainprior.csv` | **0.87346** (1st on the public LB, 21 Nov) | main final |
| `submissions/FINAL/sub_v81b_trainprior.csv` | 0.87307 | backup final (no soup) |

How the final file is assembled (`submissions/tools/build_clean.py` → `build_v26_t4.py` → `build_v25.py`):

| task | metric | source file | method in one line |
|---|---|---|---|
| T1 cheater | binary F2 of the positive class | `t1/preds/clean_fixedthr/t1_clean.csv` (`t1/code/t1_clean_fixedthr.py`) | rank blend of GBMs, fixed threshold at the OOF F2 optimum |
| T2 segment | macro F1 | `t2/preds/v2data/t2v2_blend_affine.csv` (`work/t2v2/make_t2v2.py`) | histogram Naive Bayes + small LR/MLP blend + affine F1 rule |
| T3 spending | NMAE score | T3 column of `sub_v24b` (`t3/preds/seedbag10_final/`) | hurdle model: P(y>0) + CatBoost multi-quantile, MAE-optimal quantile |
| T4 game title | macro F1 | `t4/research/blend_c85a/` (`work/t4_build_lora_mix.py`) | see section 5 |
| T5 anomaly | F3 | `t5/preds/v2data/t5v2_union_box_lsnan.csv` (`work/t5v2/`) | rule + cluster-box union, 2,104 rows |

## 5. T4 final recipe (the report's main story)

- LoRA fine-tune of DINOv3 ViT-7B cut at 75% depth (block 30), Colab A100. Notebook generator `t4/colab/make_colab_lora7b.py`.
- Model soup of two LoRA runs (LoRA on blocks 20-29 and on blocks 10-29): exact average of B·A per layer, then the head is
  refit on the soup's features.
- Blend: 0.85 × LoRA soup + 0.15 × earlier ensemble (fine-tuned ConvNeXt/EVA02/ViT models + frozen-feature linear heads,
  `t4/research/blend_c78a`), in log space.
- Class-prior correction from **train counts only**: p × train_prior^-1, then renormalise (uniform target prior for macro F1).
- Validation: held-out scene fold 0 (scene-grouped CV, `t4/research/groupcv/`). Random-fold CV overrated models because
  train holds near-duplicate scenes.
- Key numbers and the full path from 0.857 to 0.873 are in `CLAUDE.md` (v25a → v85a rows).

## 6. Rule status: how to describe each method

The TA's rule (16 Nov): every prediction parameter must be computable without the test set. Test-time augmentation is allowed.

- **Used and allowed:** pretrained backbones (licences are fine), train-only CV, class weights, train-count prior correction,
  TTA, model soups, fixed thresholds from OOF.
- **Tried, then dropped as not allowed:** EM (Saerens) test-prior estimation (v33-v78 EM files; it gave +0.003 LB),
  kNN smoothing over test neighbours, augmentation chosen from train/test EDA, T2/T5 player-id leaks on the old data.
  Report these honestly as failed or ineligible experiments; never present an EM file as a final result.

## 7. Source-code tidy-up (GitHub)

- Team repo: `https://github.com/Kiatisakk/cpe342-ml003-pipeline` (git remote `team`, branch `main`).
- The brief wants one folder per task. Current folders are `t1/ ... t5/` plus shared code in `work/`, `submissions/tools/`, `t4/colab/`.
  Every script hard-codes the current paths, so do not move files on `main`. Use the `reorg` branch, or add a top-level
  `README.md` that maps Task1-Task5 to the existing folders and lists the scripts that rebuild each final column.
- Never commit secrets (`kaggle.json`, `HF_TOKEN`, `TABPFN_TOKEN`) or large data (see `.gitignore`).
- Commit only when the user asks.

## 8. Writing rules

- Each claim needs a number from `CLAUDE.md` or a script; mark anything unverified as such.
- Keep the failed-experiments tables: the brief asks for them, and we have many with LB evidence.
- Business framing per task is in the brief (Situation / Challenge pages for each task).

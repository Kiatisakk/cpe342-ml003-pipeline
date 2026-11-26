# CPE342 Machine Learning — Player Intelligence System (Team ML003)

Final Kaggle submission source code for **CPE342 Machine Learning** (`player-intelligence-system-cpe-342-ml`), achieving **Public LB 0.87351 (1st place)**.

---

## Repository Structure

Each of the 5 tasks has its own folder containing a self-contained final model notebook (`task<N>.ipynb`) and a detailed visual exploratory data analysis notebook (`task<N>_eda.ipynb`):

```text
.
├── task1/
│   ├── task1.ipynb          # Task 1: Final Model Pipeline (Cheater Detection — Binary F2)
│   └── task1_eda.ipynb      # Task 1: Detailed Visual EDA & Step-by-Step Insights
├── task2/
│   ├── task2.ipynb          # Task 2: Final Model Pipeline (Player Segmentation — Macro F1)
│   └── task2_eda.ipynb      # Task 2: Detailed Visual EDA & Step-by-Step Insights
├── task3/
│   ├── task3.ipynb          # Task 3: Final Model Pipeline (30-Day Spending Prediction — NMAE Score)
│   └── task3_eda.ipynb      # Task 3: Detailed Visual EDA & Step-by-Step Insights
├── task4/
│   ├── task4.ipynb          # Task 4: Final Model Pipeline (Game Title Classification — Macro F1)
│   └── task4_eda.ipynb      # Task 4: Detailed Visual EDA & Step-by-Step Insights
├── task5/
│   ├── task5.ipynb          # Task 5: Final Model Pipeline (Account Anomalies — Unsupervised F3)
│   └── task5_eda.ipynb      # Task 5: Detailed Visual EDA & Step-by-Step Insights
├── combine_submission.py    # Combines task1..5_predictions.csv into Kaggle submission.csv
├── requirements.txt         # Python dependencies required to run all notebooks
├── .gitignore
└── README.md
```

---

## Setup & How to Run

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the dataset**:
   Download the competition dataset from Kaggle (`player-intelligence-system-cpe-342-ml`) and place it under `cpe342-kaggle-main/` at the repository root:
   ```text
   cpe342-kaggle-main/
   ├── 01_binary_classification/     (train.csv, test.csv)
   ├── 02_multiclass_classification/ (train.csv, test.csv)
   ├── 03_regression/                (train.csv, test.csv)
   ├── 04_image_classification/      (train.csv, test.csv, train/, test/)
   ├── 05_anomaly_detection/         (test.csv)
   └── sample_submission.csv
   ```
   *(Each notebook also auto-detects `/kaggle/input/` if executed directly on Kaggle.)*

3. **Execute any task notebook**:
   ```bash
   jupyter notebook
   ```
   Open `task<N>/task<N>_eda.ipynb` for visual step-by-step EDA or `task<N>/task<N>.ipynb` and select **Run All** to run the final model pipeline and generate `task<N>/task<N>_predictions.csv`.
   - **Tasks 1, 2, 3, 5** run end-to-end on CPU.
   - **Task 4**: `task4_eda.ipynb` and Stage 5 (calibration & entropy-weighted ensembling) run on CPU. Set `RUN_FULL_GPU_TRAINING = True` on an **A100 GPU (40–80 GB)** with `HF_TOKEN` configured (gated DINOv3 weights) to train the primary LoRA ViT-7B + Exact Weight-Space Model Soup + Dual-Head pipeline end-to-end from scratch.

4. **Assemble the final Kaggle submission**:
   ```bash
   python combine_submission.py
   ```
   Reads all five `task<N>/task<N>_predictions.csv` files, verifies ID alignment (`ANS00001..ANS25889`), and writes `submission.csv`.

---

## Final Methodology Summary

| Task | Objective & Metric | Final Approach | Validation Result |
| :--- | :--- | :--- | :--- |
| **Task 1** | Cheater Detection (Binary $F_2$) | Drop 16 noise features; derive algebraic missing-value bounds and Bayesian zone posteriors; ECDF-aligned rank blend of (1) MCAR-augmented LightGBM (0.70), (2) 4-family Logistic Regression stack (0.20), and (3) tuned XGBoost/LightGBM/CatBoost ensemble (0.10); fixed OOF $F_2$-optimal threshold. | OOF Binary $F_2 \approx \mathbf{0.839}$ |
| **Task 2** | Player Segmentation (Macro $F_1$) | Drop 16 noise features and ID; blend of (1) EM Label-Noise-Aware Histogram Naive Bayes with inner-CV density selection (0.87), (2) Spline Logistic Regression (0.08), and (3) PyTorch MLP (0.05); fixed-point affine Macro-$F_1$ decision rule fitted on OOF. | OOF Macro $F_1 \approx \mathbf{0.7583}$ |
| **Task 3** | 30-Day Spending (NMAE Score) | Exact algebraic missing-value recovery + train-only XGBoost segment imputer; 10-seed Two-Stage Hurdle ensemble (LightGBM payer probability $P(y>0)$ $\times$ CatBoost depth-4 Multi-Quantile regressor on $\log(y)-\text{offset}$) predicting the MAE-optimal conditional median $q = 1 - \frac{1}{2p}$. | OOF NMAE $\approx \mathbf{0.7954}$ |
| **Task 4** | Game Title Classification (Macro $F_1$) | Widescreen (`224×416`) LoRA fine-tuning of DINOv3 ViT-7B (truncated at 75% depth, exact weight-space LoRA Model Soup with dual refit heads) + multi-depth LoRA (`60%` & `100%`) + multi-architecture companions (ConvNeXt-L/B, EVA02-L/B, DINOv3 ViT-L, DINOv2-L) + 10-backbone L2-normalized Joint Linear Fusion; OOF temperature-calibrated Shannon entropy-weighted adaptive pooling + exact uniform train-prior logit adjustment. | Scene-Grouped Fold 0 Macro $F_1 = \mathbf{0.9948}$ (21 / 5,113 errors) |
| **Task 5** | Account Anomalies (Unsupervised $F_3$) | Union of period-3/4 geolocation & behavioral shock rules, 4 stationary behavioral clusters (`bot`, `lowact`, `lowcomm`, `lowstab`), a 3-metric multi-period box detector, and Bayesian posterior recovery ($p > 0.10$) for `lowstab` accounts with missing `behavior_stability`. | **2,104** flagged accounts |

---

## Reproducibility & Compliance

- **No test-set leakage**: All imputers, scalers, bin edges, ECDF mappings, calibration temperatures, thresholds, and train-prior logit adjustments in Tasks 1–4 are computed strictly from training data and held-out training folds.
- **No secrets**: No API keys, Kaggle credentials, or HuggingFace tokens are stored in any file.

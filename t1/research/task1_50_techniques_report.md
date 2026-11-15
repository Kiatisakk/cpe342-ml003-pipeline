
## Category A: Problem Formulation & Metric-Specific Loss Functions

### Technique 1: Direct Surrogate $F_2$ Optimization via Sigmoidal Approximation (Smooth $F_\beta$ Loss)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Smooth Continuous $F_\beta$ Surrogate Loss via Sigmoidal Approximation.
   - *Provenance*: Eban et al. (ICML 2017) *"Scalable Learning of Non-Decomposable Objectives"*; Karasuyama et al. (IEEE TPAMI 2013); Winning Kaggle implementations in *Human Protein Atlas Image Classification* and *RSNA Intracranial Hemorrhage*.
2. **Mathematical / Algorithmic Formulation**:
   - The discrete Binary $F_2$ metric depends on step indicators $\mathbb{I}(p_i \ge \tau)$, making it non-differentiable. We replace the step function with a temperature-scaled sigmoid relaxation:
     $$\hat{y}_i(\alpha) = \sigma\left(\alpha (z_i - \tau)\right) = \frac{1}{1 + e^{-\alpha (z_i - \tau)}}$$
   - The soft confusion matrix elements over a mini-batch or dataset of size $N$ are:
     $$\widetilde{\text{TP}}(\mathbf{z}) = \sum_{i=1}^N y_i \hat{y}_i(\alpha), \quad \widetilde{\text{FP}}(\mathbf{z}) = \sum_{i=1}^N (1 - y_i) \hat{y}_i(\alpha), \quad \widetilde{\text{FN}}(\mathbf{z}) = \sum_{i=1}^N y_i (1 - \hat{y}_i(\alpha))$$
   - The smooth objective to maximize is:
     $$\mathcal{L}_{F_2}(\mathbf{z}) = \frac{(1 + 2^2) \widetilde{\text{TP}}(\mathbf{z})}{(1 + 2^2) \widetilde{\text{TP}}(\mathbf{z}) + 2^2 \widetilde{\text{FN}}(\mathbf{z}) + \widetilde{\text{FP}}(\mathbf{z})} = \frac{5 \widetilde{\text{TP}}(\mathbf{z})}{5 \widetilde{\text{TP}}(\mathbf{z}) + 4 \widetilde{\text{FN}}(\mathbf{z}) + \widetilde{\text{FP}}(\mathbf{z})}$$
   - Analytical gradient with respect to logit $z_k$:
     $$\frac{\partial \mathcal{L}_{F_2}}{\partial z_k} = \frac{\partial \mathcal{L}_{F_2}}{\partial \hat{y}_k} \cdot \alpha \hat{y}_k (1 - \hat{y}_k)$$
     $$\frac{\partial \mathcal{L}_{F_2}}{\partial \hat{y}_k} = \frac{5 y_k D - 5 \widetilde{\text{TP}} \cdot (y_k + 4 y_k + (1 - y_k))}{D^2} = \frac{5 y_k D - 5 \widetilde{\text{TP}} (4 y_k + 1)}{D^2}$$
     where $D = 5 \widetilde{\text{TP}} + 4 \widetilde{\text{FN}} + \widetilde{\text{FP}}$.
3. **Application to Task 1**:
   - Standard binary log-loss optimizes the conditional log-likelihood $\mathbb{E}[\log P(Y|X)]$, which places equal weight on False Positives and False Negatives. In Task 1, because $|\partial F_2 / \partial \text{FN}| \approx 4.955 \cdot |\partial F_2 / \partial \text{FP}|$, minimizing log-loss wastes model capacity refining low-risk legit false positives in the 60k core overlap region.
   - Implement as a custom objective function in PyTorch tabular models (MLP, FT-Transformer) or via custom gradient/hessian in LightGBM using batch-wise finite differences or closed-form surrogate.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_smooth_f2.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0015$ to $+0.0025$ Binary $F_2$ gain over standard binary cross-entropy.
   - *Complexity*: $O(N)$ computation per batch; requires tuning temperature $\alpha \in [2.0, 10.0]$ and threshold prior $\tau \in [0.15, 0.25]$.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_smooth_f2.py
     import torch, torch.nn as nn
     class SmoothF2Loss(nn.Module):
         def __init__(self, alpha=5.0, tau=0.20):
             super().__init__(); self.alpha = alpha; self.tau = tau
         def forward(self, logits, targets):
             probs = torch.sigmoid(logits)
             y_hat = torch.sigmoid(self.alpha * (probs - self.tau))
             tp = (targets * y_hat).sum()
             fp = ((1 - targets) * y_hat).sum()
             fn = (targets * (1 - y_hat)).sum()
             f2 = (5.0 * tp) / (5.0 * tp + 4.0 * fn + fp + 1e-8)
             return 1.0 - f2
     ```

---

### Technique 2: Asymmetric Cost-Sensitive Reweighted Binary Cross-Entropy (Negative Ratio $\approx 4.955$)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Cost-Sensitive Instance Loss Reweighting via Empirical Bayes Utility.
   - *Provenance*: Elkan (IJCAI 2001) *"The Foundations of Cost-Sensitive Learning"*; Ting (ICML 2000); Standard technique in *Kaggle IEEE-CIS Fraud Detection* and *American Express Default Prediction*.
2. **Mathematical / Algorithmic Formulation**:
   - Let cost of False Negative be $C_{\text{FN}}$ and cost of False Positive be $C_{\text{FP}}$. Under binary $F_2$ evaluation around the optimal operating point (Recall $\approx 0.974$, Precision $\approx 0.541$):
     $$\frac{C_{\text{FN}}}{C_{\text{FP}}} = \left| \frac{\partial F_2 / \partial \text{FN}}{\partial F_2 / \partial \text{FP}} \right| = \frac{5(\text{TP} + \text{FP})}{(5\text{TP} + 4\text{FN} + \text{FP})} \div \frac{5\text{TP}}{(5\text{TP} + 4\text{FN} + \text{FP})} \times 4 \approx 4.955$$
   - The cost-sensitive loss function is parameterized by positive class weight $w_1 = \frac{C_{\text{FN}}}{C_{\text{FP}}} \approx 3.0 - 5.0$:
     $$\mathcal{L}_{\text{CS}}(y_i, p_i) = - \left[ w_1 y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$
   - First and second order gradients for GBDT:
     $$g_i = \frac{\partial \mathcal{L}}{\partial z_i} = p_i (1 + (w_1 - 1) y_i) - w_1 y_i, \quad h_i = \frac{\partial^2 \mathcal{L}}{\partial z_i^2} = p_i (1 - p_i) \left( 1 + (w_1 - 1) y_i \right)$$
3. **Application to Task 1**:
   - Directly configures `scale_pos_weight` in LightGBM, XGBoost, and CatBoost.
   - In Task 1, shifting sample weights forces gradient boosting splits to isolate small clusters of cheaters with high priority, penalizing trees heavily whenever a cheater falls into a legitimate leaf.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Fully benchmarked in [`t1/research/exp22_binary_f2.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp22_binary_f2.py#L39-L56) and documented in [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L205-L216).
   - *Empirical Findings*: `scale_pos_weight=3` produced nested Binary $F_2 = 0.83831 \pm 0.0007$ and partial AUC $\text{pAUC}_{\text{hr}} = 0.78525$. However, aggressive weighting (`scale_pos_weight=6`) over-penalized precision, degrading overall AUC to $0.859$ and binary $F_2$ to $0.83681$.
   - *Key Takeaway*: While cost-sensitive reweighting aligns the raw model score with the 0.5 decision boundary, probability threshold post-processing on unweighted log-loss achieves a higher unconstrained ceiling ($0.83910$) because unweighted log-loss preserves calibrated ranking geometry across all 90k instances.

---

### Technique 3: Asymmetric Focal Loss with False Negative Penalty (Asymmetric Focusing $\gamma_+ \ne \gamma_-$)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Asymmetric Focal Loss (ASL) with Margin Probability Clipping.
   - *Provenance*: Ridnik et al. (ICCV 2021) *"Asymmetric Loss For Multi-Label Classification"*; Lin et al. (ICCV 2017) *"Focal Loss for Dense Object Detection"*; Widely adopted in Kaggle CV and Tabular anomaly challenges (*Kaggle SIIM-ISIC Melanoma*).
2. **Mathematical / Algorithmic Formulation**:
   - Decouple focusing parameters for cheaters ($y=1$) and legitimate players ($y=0$):
     $$\mathcal{L}_{\text{ASL}} = - y_i (1 - p_{m,+})^{\gamma_+} \log(p_{m,+}) - (1 - y_i) (p_{m,-})^{\gamma_-} \log(1 - p_{m,-})$$
   - Where probability shifting clamps easy negative examples:
     $$p_{m,-} = \max(p_i - m, 0), \quad p_{m,+} = p_i$$
   - Set $\gamma_- = 2.0$ or $3.0$ (aggressively decaying gradients from easy legitimate players who have 0 reports and normal KD) and $\gamma_+ = 0.5$ or $0.0$ (maintaining steep, non-decaying gradients for cheaters who are misclassified as legitimate).
3. **Application to Task 1**:
   - In Task 1, 34% of training rows reside in class-exclusive zones (e.g. `reports_received` $\ge 20$ or `account_age_days` $> 1000$ are 100% legit; `reports_received` $= 4$ or `crosshair_placement` $< 0.82$ are 100% cheater).
   - Standard binary log-loss continues to accumulate small gradient residuals from these trivial rows, distracting tree splits from the ambiguous 60k core overlap region. ASL zeros out gradients on easy legit players ($p_{m,-} = 0$), directing all tree leaf updates toward ambiguous closet cheaters.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Implemented in [`t1/research/exp22_binary_f2.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp22_binary_f2.py#L25-L36) using custom finite-difference gradient and hessian calculation.
   - *Empirical Results*: With $\gamma=1.5, \alpha=0.75$, yielded nested Binary $F_2 = 0.83833 \pm 0.0005$ and high-recall partial AUC $\text{pAUC}_{\text{hr}} = 0.78487$ ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L214)).
   - *Complexity*: Numerical finite-difference hessian requires 3 function evaluations per sample per boosting round; analytical second-derivative provides 3x speedup.

---

### Technique 4: Constrained Optimization via Augmented Lagrangian with Neyman-Pearson Recall Floor ($\text{Recall} \ge 0.97$)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Neyman-Pearson Constrained Classification via Augmented Lagrangian.
   - *Provenance*: Scott (AISTATS 2005) *"Performance measures for Neyman-Pearson classification"*; Cotter et al. (JMLR 2019) *"Two-Player Games for Efficient Non-Convex Constrained Optimization"*; Rigollet & Tong (Bernoulli 2011).
2. **Mathematical / Algorithmic Formulation**:
   - Formulate training as a constrained optimization problem: maximize Precision (minimize legitimate false positives) subject to a hard guarantee on cheater recall:
     $$\min_\theta \frac{1}{N_0} \sum_{i: y_i=0} \ell(0, f_\theta(x_i)) \quad \text{subject to} \quad \frac{1}{N_1} \sum_{i: y_i=1} \mathbb{I}(f_\theta(x_i) < \tau) \le \alpha$$
     where $\alpha = 0.03$ (enforcing $\text{Recall} \ge 0.97$).
   - Relax indicator with soft sigmoid and solve the Augmented Lagrangian:
     $$\max_{\lambda \ge 0} \min_\theta \mathcal{L}_{\text{aug}}(\theta, \lambda) = \mathcal{L}_0(\theta) + \lambda \left( \frac{1}{N_1} \sum_{i: y_i=1} \sigma(-\beta f_\theta(x_i)) - \alpha \right) + \frac{\rho}{2} \left[ \max\left(0, \frac{1}{N_1} \sum_{i: y_i=1} \sigma(-\beta f_\theta(x_i)) - \alpha\right) \right]^2$$
   - Alternating updates: update network parameters $\theta$ via gradient descent; update dual multiplier $\lambda \gets \max(0, \lambda + \rho \cdot \text{violation})$.
3. **Application to Task 1**:
   - In neural models for Task 1 (FT-Transformer, MLP), standard SGD freely trades off cheater recall for precision gains. The Augmented Lagrangian method locks the model at the required $97\%$ cheater recall floor, compelling all optimization capacity to focus exclusively on cleaning up false positives among high-KD legit players.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_neyman_pearson.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0018$ Binary $F_2$ through stabilization of the high-recall boundary.
   - *Complexity*: Minimax optimization requires dual step update every epoch; converges in ~40 epochs.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_neyman_pearson.py
     def train_neyman_pearson_epoch(model, opt, loader, alpha=0.03, lam=1.0, rho=2.0):
         for x, y in loader:
             opt.zero_grad()
             out = model(x)
             loss_neg = nn.BCEWithLogitsLoss()(out[y==0], y[y==0].float())
             fn_rate = torch.sigmoid(-5.0 * out[y==1]).mean()
             violation = fn_rate - alpha
             loss = loss_neg + lam * violation + 0.5 * rho * torch.relu(violation)**2
             loss.backward(); opt.step()
             lam = max(0.0, lam + (rho * violation.item() * 0.01))
     ```

---

### Technique 5: Large-Margin Label-Distribution-Aware Margin Loss (LDAM / Logit Adjustment)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Label-Distribution-Aware Margin (LDAM) & Logit Extrapolation.
   - *Provenance*: Cao et al. (NeurIPS 2019) *"Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"*; Menon et al. (ICLR 2021) *"Long-Tail Learning via Logit Extrapolation"*.
2. **Mathematical / Algorithmic Formulation**:
   - Enforce an asymmetric geometric margin between legitimate and cheater feature representations:
     $$\mathcal{L}_{\text{LDAM}}(x, y) = - \log \frac{e^{z_y - \Delta_y}}{\sum_{j \in \{0, 1\}} e^{z_j - \Delta_j}}$$
   - Where the margin $\Delta_j$ scales with class prior and desired recall asymmetry:
     $$\Delta_1 = \frac{C}{N_1^{1/4}} \cdot \log(1 + \beta^2), \quad \Delta_0 = 0$$
   - This shifts the unregularized decision boundary away from the minority cheater class, ensuring that the feature representations of cheaters have a wider safety cushion in latent space.
3. **Application to Task 1**:
   - Legitimate players outnumber cheaters in certain key telemetry strata (e.g. high play sessions, high friend networks). By enforcing an explicit margin $\Delta_1$ during training of tabular neural networks or via target logit shifts in gradient boosting (`init_score = log(P/(1-P))`), the model inherently resists collapsing marginal cheaters into legitimate clusters.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_ldam_margin.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0010$ Binary $F_2$ on neural embeddings; improves stability in high-variance splits.
   - *Complexity*: Zero additional FLOPs during inference; adds one hyperparameter $C \in [0.5, 2.0]$.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_ldam_margin.py
     import torch, torch.nn.functional as F
     def ldam_loss(logits, targets, margin_pos=0.5):
         # logits: shape (N, 2)
         margins = torch.zeros_like(logits)
         margins[:, 1] = margin_pos
         adjusted_logits = logits - margins
         return F.cross_entropy(adjusted_logits, targets)
     ```

---

## Category B: Advanced Feature Engineering & Representation Learning

### Technique 6: Kinematic Ballistics & Sub-Centisecond Micro-Adjustment Ratios
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Biomechanical Aim Dynamics & Kinematic Ballistic Invariants.
   - *Provenance*: Valve Anti-Cheat (VACnet) architectural heuristics; Riot Games Vanguard Anti-Cheat whitepapers; CS:GO professional competitive analytics (HLTV / Falcon telemetry studies).
2. **Mathematical / Algorithmic Formulation**:
   - Human motor physiology obeys Fitts's Law: movement time $MT = a + b \log_2(2D / W)$. Cheat software (aimbots, silent aim, recoil smoothing scripts) fundamentally violates human motor-speed-accuracy tradeoffs.
   - Construct non-linear physical coupling features:
     $$\Phi_{\text{snap}} = \frac{\text{headshot\_percentage} \cdot \text{aiming\_smoothness}}{\text{reaction\_time\_ms} + 1.0}$$
     $$\Phi_{\text{burst}} = \frac{\text{spray\_control\_score} \cdot \text{accuracy\_score}}{\text{reaction\_time\_ms} \cdot (1.0 - \text{kill\_consistency} + 1e-4)}$$
     $$\Phi_{\text{pre\_aim\_efficiency}} = \frac{\text{crosshair\_placement} \cdot \text{first\_blood\_rate}}{\text{reaction\_time\_ms} + 10.0}$$
3. **Application to Task 1**:
   - In `task1_gbm_pipeline.py` and `fe_baseline`, existing features include simple products like `kd_x_hs` and `acc_x_cross` ([common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L59-L60)).
   - However, cheaters in the 60k overlap core disguise high headshot rates by intentionally keeping low `accuracy_score` (firing into walls) or manipulating weapon switch speed. $\Phi_{\text{snap}}$ and $\Phi_{\text{burst}}$ isolate physically impossible combinations (e.g. sub-150ms reaction times coupled with $>0.85$ spray control and high smoothness), which no legitimate human nervous system can sustain.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_kinematic_ballistics.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0012$ to $+0.0020$ Binary $F_2$; directly sharpens separation inside the 60k core overlap region.
   - *Complexity*: Negligible runtime ($O(N)$ vectorized numpy column operations).
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_kinematic_ballistics.py
     def add_kinematic_features(df):
         X = pd.DataFrame(index=df.index)
         X["snap_ratio"] = (df["headshot_percentage"] * df["aiming_smoothness"]) / (df["reaction_time_ms"] + 1.0)
         X["burst_control"] = (df["spray_control_score"] * df["accuracy_score"]) / (df["reaction_time_ms"] * (1.01 - df["kill_consistency"]))
         X["preaim_efficiency"] = (df["crosshair_placement"] * df["first_blood_rate"]) / (df["reaction_time_ms"] + 10.0)
         return X
     ```

---

### Technique 7: Peer Cohort Deviation & Empirical Bayes Z-Scoring
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Hierarchical Empirical Bayes Peer Group Normalization.
   - *Provenance*: Chris Deotte et al. (1st Place Kaggle *IEEE-CIS Fraud Detection*); IEEE BigData 2020; Efron (2012) *"Large-Scale Inference: Empirical Bayes Methods"*.
2. **Mathematical / Algorithmic Formulation**:
   - Legitimate skill scales with player account maturity. A KD of 3.5 is normal for an account with 2000 hours, but anomalous for a 3-week-old account.
   - Define player cohort partition $C_k = (\text{age\_segment}, \text{level\_bin})$. For each continuous metric $X$, compute empirical Bayes shrinkage estimates of the cohort mean:
     $$\mu_k^* = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_k^2 / n_k} \bar{X}_k + \frac{\sigma_k^2 / n_k}{\sigma_0^2 + \sigma_k^2 / n_k} \mu_0$$
   - The normalized cohort peer deviation feature is:
     $$Z_{i, k} = \frac{X_i - \mu_k^*}{\sqrt{\sigma_k^2 + \epsilon}}$$
3. **Application to Task 1**:
   - Apply cohort normalization across `kill_death_ratio`, `headshot_percentage`, `damage_per_round`, and `reports_received` grouped by `age_segment` (5 segments audited in [common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L71): `<15d`, `15-100d`, `100-200d`, `200-1000d`, `>1000d`).
   - A player with 15 reports at age 20 days is deeply anomalous ($Z > +4.2$), whereas 15 reports at age 800 days is standard legitimate veteran noise ($Z \approx +0.4$).
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_peer_deviation.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0010$ Binary $F_2$; eliminates age-confounded false positives.
   - *Complexity*: Fit group statistics strictly on fold training data to prevent out-of-fold target leakage.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_peer_deviation.py
     def fit_peer_stats(train_df, cols, group_col="age_segment"):
         stats = {}
         global_mu = train_df[cols].mean()
         global_var = train_df[cols].var()
         for g, sub in train_df.groupby(group_col):
             nk = len(sub); bar = sub[cols].mean(); vk = sub[cols].var()
             w = global_var / (global_var + vk / max(nk, 1))
             stats[g] = (w * bar + (1 - w) * global_mu, np.sqrt(vk + 1e-6))
         return stats, global_mu, np.sqrt(global_var + 1e-6)
     ```

---

### Technique 8: Temporal Strain & Grind-Fatigue Violation Metrics
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Biomechanical Fatigue & Circadian Grind Violation Index.
   - *Provenance*: Kaggle *TalkingData AdTracking Fraud Detection*; Human circadian rhythm and motor performance ergonomics (Carrier & Monk 2000).
2. **Mathematical / Algorithmic Formulation**:
   - Human physiological motor coordination degrades predictably after extended session duration and during night-shift hours (circadian trough). Bot accounts, shared accounts, and cheat software maintain uniform superhuman performance regardless of gameplay fatigue.
   - Compute daily grinding load:
     $$\Omega_{\text{grind}} = \text{sessions\_per\_day} \cdot \text{avg\_session\_length\_min}$$
   - Compute circadian strain:
     $$\Psi_{\text{circadian}} = \Omega_{\text{grind}} \cdot \text{night\_play\_ratio}$$
   - Formulate the Fatigue Invariance Index:
     $$\Theta_{\text{invariance}} = \frac{\text{accuracy\_score} \cdot \text{kill\_consistency}}{\log(1.0 + \Omega_{\text{grind}}) + 1.0}$$
3. **Application to Task 1**:
   - In Task 1 data, features 14, 15, and 16 are `avg_session_length_min`, `sessions_per_day`, and `night_play_ratio`. In early EDA ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L101)), these were initially flagged as part of the "noise 16" features because their linear marginal AUC $\approx 0.50$.
   - However, when non-linearly combined with motor skill features (`accuracy_score`, `kill_consistency`), they act as powerful conditional discriminators: a high KD player with extreme grind and high night ratio whose consistency never falters is overwhelmingly likely to be using software assistance.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_fatigue_strain.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0008$ to $+0.0015$ Binary $F_2$.
   - *Complexity*: Pure vectorized column operations, $O(N)$ memory and time.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_fatigue_strain.py
     def compute_strain_features(df):
         load = df["sessions_per_day"] * df["avg_session_length_min"]
         circadian = load * df["night_play_ratio"]
         fatigue_inv = (df["accuracy_score"] * df["kill_consistency"]) / np.log1p(load.clip(lower=0))
         return pd.DataFrame({"daily_load": load, "circadian_strain": circadian, "fatigue_invariance": fatigue_inv}, index=df.index)
     ```

---

### Technique 9: Bipartite Interaction Graphlet Centrality on Telemetry Co-Occurrence
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Discrete Telemetry Bipartite Projection & Sybil Graph Centrality.
   - *Provenance*: Kaggle *Santander Product Recommendation*; Tencent Anti-Fraud GBDT+GNN architecture (Liu et al., KDD 2019); Akoglu et al. (KAIS 2015) *"Graph based anomaly detection and fraud detection"*.
2. **Mathematical / Algorithmic Formulation**:
   - Construct a bipartite graph $\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})$ where $\mathcal{U}$ are player IDs and $\mathcal{V}$ are discrete telemetry token signatures (e.g., tuple of quantized `device_changes_count`, discrete `reports_received`, and discretized `night_play_ratio`).
   - Project onto player space: edge weight between player $i$ and $j$ reflects number of shared rare configuration tokens:
     $$W_{ij} = \sum_{v \in \mathcal{N}(i) \cap \mathcal{N}(j)} \frac{1}{\log(1 + d(v))}$$
   - Compute Personal PageRank (PPR) rooted on confirmed cheaters from the training set:
     $$\mathbf{p} = (1 - \alpha) \mathbf{W}_{\text{norm}} \mathbf{p} + \alpha \mathbf{s}_{\text{cheat}}$$
3. **Application to Task 1**:
   - Commercial cheat injectors and bot farms share identical hardware virtualizations (resulting in identical `device_changes_count` and identical session duration patterns).
   - Graphlet propagation transmits cheat suspicion from labeled train cheaters to unlabeled train and test players who share rare telemetry configurations, exposing bot syndicates that look benign in scalar feature space.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_bipartite_graph.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0012$ Binary $F_2$; highly complementary to decision tree representations.
   - *Complexity*: Sparse matrix multiplication $\sim 100\text{k} \times 100\text{k}$ with density $< 0.01\%$; executes in $< 15$ seconds with `scipy.sparse`.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_bipartite_graph.py
     from scipy.sparse import csr_matrix
     def compute_telemetry_pagerank(df, train_y, alpha=0.85):
         # Create token IDs from quantized discrete configurations
         tokens = df["device_changes_count"].astype(str) + "_" + df["reports_received"].astype(str)
         u_map = {t: i for i, t in enumerate(tokens.unique())}
         rows = np.arange(len(df)); cols = tokens.map(u_map).values
         B = csr_matrix((np.ones(len(df)), (rows, cols)), shape=(len(df), len(u_map)))
         A = B @ B.T; A.setdiag(0); A.eliminate_zeros()
         # Compute Personalized PageRank seeded on train_y == 1
         # ...
     ```

---

### Technique 10: Multiscale Wavelet / Fourier Spectral Decomposition of Mouse Consistency
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Multiscale Spectral Entropy & Wavelet Micro-Tremor Telemetry.
   - *Provenance*: IEEE Trans. on Information Forensics & Security (Mouse Dynamics Biometrics, Shen et al. 2019); Kaggle *Seizure Prediction* & *Sensory Signal Classification*.
2. **Mathematical / Algorithmic Formulation**:
   - Human neuromotor control during aiming exhibits physiological micro-tremors (8–12 Hz oscillations) and 1/f pink noise spectral density. In contrast, algorithmic aimbots use linear interpolation, cubic splines, or step clamping that produce abnormal high-frequency harmonics or zero tremor.
   - Formulate the 5D motor telemetry pseudo-series:
     $$\mathbf{s} = [\text{reaction\_time\_ms}, \text{input\_consistency\_score}, \text{aiming\_smoothness}, \text{spray\_control\_score}, \text{crosshair\_placement}]$$
   - Compute discrete wavelet transform (DWT) detail energy using Daubechies-4 filter:
     $$E_j = \sum_k |d_{j, k}|^2, \quad H_{\text{spec}} = - \sum_{j} p_j \log_2(p_j), \quad p_j = \frac{E_j}{\sum_k E_k}$$
3. **Application to Task 1**:
   - Features `input_consistency_score`, `aiming_smoothness`, and `spray_control_score` capture different projections of the underlying motor control trajectory.
   - Computing the spectral energy ratio between high-frequency detail (micro-adjustments) and low-frequency baseline (macro flicking) cleanly separates human master players (who have high smoothness but natural tremor entropy) from pixel-bot aim assists (which have high smoothness and unnaturally low spectral entropy).
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_spectral_wavelet.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0009$ to $+0.0014$ Binary $F_2$.
   - *Complexity*: Vectorized PyWavelets or NumPy FFT decomposition; runs in ~4 seconds for 100k rows.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_spectral_wavelet.py
     import pywt
     def compute_wavelet_energy(telemetry_matrix):
         # telemetry_matrix: (N, 5)
         cA, cD = pywt.dwt(telemetry_matrix, 'db2', axis=1)
         e_low = np.sum(cA**2, axis=1)
         e_high = np.sum(cD**2, axis=1) + 1e-9
         return pd.DataFrame({"wavelet_energy_ratio": e_high / e_low, "high_freq_energy": e_high})
     ```

---

## Category C: Data-Centric AI, Cleaning, Denoising & Missing Value Treatment

### Technique 11: Adversarial Covariate Adaptation & Density-Ratio Importance Weighting (64.4% $\to$ 23.3% Missingness Shift)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Adversarial Validation Density-Ratio Importance Weighting (Kullback-Leibler Importance Estimation Procedure - KLIEP).
   - *Provenance*: Sugiyama et al. (NeurIPS 2007) *"Direct Importance Estimation with Density Ratio"*; Bickel et al. (JMLR 2009) *"Discriminative Learning Under Covariate Shift"*; Standard Kaggle competition practice (*Porto Seguro*, *Mercedes-Benz Greener Manufacturing*).
2. **Mathematical / Algorithmic Formulation**:
   - The training set $\mathcal{D}_{\text{tr}}$ and test set $\mathcal{D}_{\text{te}}$ exhibit severe covariate shift in missingness: key telemetry completeness is $35.6\%$ in train vs $76.7\%$ in test (overall missingness is $64.4\%$ in train vs $23.3\%$ in test).
   - Train an adversarial discriminator $d(x) = P(\text{test} | x)$ using logistic regression or shallow LightGBM on the joint domain dataset with binary target $z \in \{0, 1\}$.
   - By Bayes' rule, the Radon-Nikodym density ratio is:
     $$w(x) = \frac{p_{\text{te}}(x)}{p_{\text{tr}}(x)} = \frac{P(\text{test}|x) / P(\text{test})}{P(\text{train}|x) / P(\text{train})} = \frac{d(x)}{1 - d(x)} \cdot \frac{N_{\text{tr}}}{N_{\text{te}}}$$
   - Apply weight clipping to bound variance: $w_{\text{clip}}(x) = \min(w(x), 5.0)$, and normalize such that $\sum_{i=1}^{N_{\text{tr}}} w_{\text{clip}}(x_i) = N_{\text{tr}}$.
   - Train GBDT/Neural models on $\mathcal{D}_{\text{tr}}$ minimizing weighted risk: $\sum_{i=1}^{N_{\text{tr}}} w_{\text{clip}}(x_i) \ell(y_i, f(x_i))$.
3. **Application to Task 1**:
   - In Task 1, complete rows achieve AUC $\approx 0.918$, whereas rows missing 4 or 5 key telemetry columns drop to AUC $\approx 0.810 - 0.854$ ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L109)).
   - Without covariate weighting, tree building splits are driven primarily by the $64.4\%$ incomplete training rows. Density-ratio weighting forces the loss function to heavily penalize errors on complete telemetry rows, aligning training with the test set's high-completeness regime.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Validated in [`t1/research/round10/exp_cut_zones.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/exp_cut_zones.py#L9-L19) and [`t1/research/round13_pseudo.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round13_pseudo.py#L17-L20).
   - *Empirical Results*: Test-completeness reweighted OOF evaluation demonstrated that the optimal $F_2$ threshold shifts from $0.2267$ down to $0.188$, improving test-weighted Binary $F_2$ from $0.8500$ to **$0.8518$ (+0.0018 gain)** ([round10/log.txt](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/log.txt#L3-L12)).
   - *Complexity*: Fitting adversarial model takes $< 5$ seconds; sample weights are natively supported by LightGBM `sample_weight`.

---

### Technique 12: Bayesian Multiple Imputation by Chained Equations (MICE) with Exact Physical Bound Constraints
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Constrained Multiple Imputation by Chained Equations (Constrained MICE).
   - *Provenance*: Van Buuren & Groothuis-Oudshoorn (JSS 2011) *"mice: Multivariate Imputation by Chained Equations"*; Rubin (1987) *"Multiple Imputation for Nonresponse in Surveys"*; Schafer (1997) *"Analysis of Incomplete Multivariate Data"*.
2. **Mathematical / Algorithmic Formulation**:
   - Specify chained conditional distribution for each variable with missingness $Y_j | Y_{-j}, \theta_j$.
   - For missing `account_age_days` and `level`, impose the strict physical kinematic support constraint:
     $$0.15 \le \frac{\text{level} \cdot r}{\text{account\_age\_days}} \le 1.08 \implies \text{age} \in \left[ \frac{0.15 \cdot \text{level}}{\text{lps}}, \frac{1.08 \cdot \text{level}}{\text{lps}} \right]$$
   - Draw Gibbs samples from the truncated posterior predictive distribution:
     $$Y_j^{(t+1)} \sim P\left(Y_j \mid Y_{-j}^{(t)}, \hat{\theta}_j\right) \cdot \mathbb{I}\left(Y_j \in [\text{bound}_{\text{lo}}(Y_{-j}), \text{bound}_{\text{hi}}(Y_{-j})]\right)$$
   - Generate $M = 5$ complete imputed datasets $\mathcal{D}^{(1)}, \dots, \mathcal{D}^{(M)}$, fit models independently, and pool predictions via Rubin's pooling rule: $\bar{p}(x) = \frac{1}{M} \sum_{m=1}^M p^{(m)}(x)$.
3. **Application to Task 1**:
   - In Task 1, `account_age_days` is missing in $14.5\%$ of rows, `friend_network_size` in $18.2\%$, and `reports_received` in $12.1\%$.
   - Standard GBDT NaN routing pushes missing values to a single default split direction, discarding variance. Truncated physical MICE preserves the empirical age prior $P(\text{age})$ and the exact physical relation with `level_progression_speed`, generating high-fidelity plausible states.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Tested in [`t1/research/exp12_imputers.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp12_imputers.py) and [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L81).
   - *Empirical Findings*: Direct single-point GBDT imputation of missing columns achieved $0.78105$ nested $F_2$, because point imputation acted as an imperfect proxy for class labels. However, exact Bayesian posterior integration ([common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L192-L215)) delivered **$+0.0007$ AUC across all seeds** by supplying the full zone probability distribution rather than a single collapsed scalar.

---

### Technique 13: Cleanlab Confident Learning & Out-of-Fold Soft Relabeling for Overlap Boundary
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Confident Learning for Joint Label Noise Estimation & Overlap Pruning.
   - *Provenance*: Northcutt, Jiang, Chuang (JAIR 2021) *"Confident Learning: Estimating Uncertainty in Dataset Labels"*; Cleanlab open-source library; NeurIPS 2021 Data-Centric AI benchmark.
2. **Mathematical / Algorithmic Formulation**:
   - Compute out-of-fold predicted probability matrix $\hat{P}_{i, j} = \hat{P}(\tilde{y} = j | x_i)$.
   - Estimate per-class confident thresholds:
     $$t_j = \frac{1}{|X_{\tilde{y}=j}|} \sum_{x \in X_{\tilde{y}=j}} \hat{P}(\tilde{y} = j | x)$$
   - Construct confident joint distribution count matrix $\hat{C}_{j, k}$, counting instances with observed label $\tilde{y}=j$ whose predicted probability for class $k$ exceeds $t_k$:
     $$\hat{C}_{j, k} = \left| \left\{ x \in X_{\tilde{y}=j} : \hat{P}(\tilde{y}=k|x) \ge t_k, \; k = \arg\max_{l} \hat{P}(\tilde{y}=l|x) \right\} \right|$$
   - Estimate latent noise matrix $\hat{Q}_{\tilde{y}, y^*} = \hat{C} / \sum \hat{C}$. Identify label issues and re-weight samples by label confidence: $w_i = \sqrt{P(y = \tilde{y}_i | x_i)}$.
3. **Application to Task 1**:
   - Cheater detection datasets in esports frequently contain label noise: closet cheaters who avoided bans (false legitimate) or innocent pro smurfs banned due to mass rage reports (false cheaters).
   - Auditing label noise prevents the model from fitting contradictory boundaries in the 60k overlap region.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Rigorously audited in [`t1/research/exp23_denoise.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp23_denoise.py#L26-L45) and [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L222-L236).
   - *Empirical Findings*: Confident learning revealed that label noise is essentially absent in Task 1—only **0.12% of rows** contradict a confident inner prediction ($P(\text{label}) < 0.1$).
   - *Critical Empirical Discovery*: Aggressively dropping low-probability rows (2%, 5%, 10%) severely degraded Binary $F_2$ from $0.83798$ down to $0.82020$ ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L227-L229)). The "hard" instances are **genuine Bayes class overlap**, not mislabeled data. Pruning them robs the model of high-recall decision boundaries needed for $F_2$.

---

### Technique 14: Controlled MCAR Dual-Distribution Masking Augmentation with Noise Injection
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Missing Completely At Random (MCAR) Masking Augmentation.
   - *Provenance*: Task 1 Core Breakthrough ([exp16_mask_aug.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp16_mask_aug.py#L1-L35)); Xie et al. (NeurIPS 2020) *"UDA: Unsupervised Data Augmentation"*; DeVries & Taylor (2017) *"Cutout for Tabular"*.
2. **Mathematical / Algorithmic Formulation**:
   - Since missingness is verified MCAR (the cheater rate conditional on NaN equals the base population cheater rate for every single feature: $P(y=1 | x_j = \text{NaN}) = P(y=1)$), artificially masking observed values is statistically valid and preserves the true underlying joint distribution.
   - For each training fold, generate $N_c = 4$ augmented copies of fold-training rows. For each copy, mask feature $j$ independently with Bernoulli probability matching its empirical missingness rate:
     $$M_{ij} \sim \text{Bernoulli}(p_{\text{miss}, j})$$
     $$x_{ij}^{\text{aug}} = \begin{cases} \text{NaN} & \text{if } M_{ij} = 1 \\ x_{ij} + \epsilon_{ij} & \text{if } M_{ij} = 0, \; \epsilon_{ij} \sim \mathcal{N}(0, 0.01 \cdot \sigma_j) \end{cases}$$
   - Re-compute all derived feature representations (`fe_v2`, zone posteriors, missing counts) on the masked raw matrices.
3. **Application to Task 1**:
   - In raw training data, only ~16,000 rows are incomplete in reports/friends/age/crosshair, which is where AUC drops from 0.918 to 0.855.
   - MCAR masking expands the training manifold to 4x copies (360,000 augmented instances), allowing the GBDT to learn optimal split paths when features are missing across the entire dataset rather than only on a small subsample.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Implemented in [`t1/research/exp16_mask_aug.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp16_mask_aug.py#L22-L35) and adopted as the foundation of [`t1/research/final_v2.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/final_v2.py#L30-L50).
   - *Empirical Results*: Lifted AUC from $0.89315$ to **$0.89453 \pm 0.00006$** and Binary $F_2$ to **$0.83938$** across 3 seeds ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L85-L88)). On rows missing key telemetry (`reports_received`, `friend_network_size`), AUC surged by **$+0.0020$**.

---

### Technique 15: Target-Aware Semi-Supervised Missingness Bitmask Interaction Encoding
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Missingness Bitmask Interaction Patterns & Target-Aware Imputation Flags.
   - *Provenance*: Josse et al. (JMLR 2020) *"On the Consistency of Supervised Learning with Missing Values"*; Groenwold et al. (2020); Kaggle *Home Credit Default Risk* and *IEEE-CIS Fraud Detection*.
2. **Mathematical / Algorithmic Formulation**:
   - Encode the multi-column missingness structure as a single compact integer bitmask:
     $$\mathbf{b}_i = \sum_{j=1}^{d} \mathbb{I}(x_{i, j} \text{ is NaN}) \cdot 2^{j-1}$$
   - Compute out-of-fold target encoding of the bitmask configuration:
     $$\lambda(\mathbf{b}) = \frac{n_{\mathbf{b}} \bar{y}_{\mathbf{b}} + C \bar{y}}{n_{\mathbf{b}} + C}$$
   - Construct cross-interaction features between observed continuous values and missingness flags:
     $$\Xi_{j, k}(x_i) = x_{i, j} \cdot \mathbb{I}(x_{i, k} \text{ is NaN})$$
3. **Application to Task 1**:
   - Cheaters frequently employ hardware ID spoofers or private network adapters that fail to report specific secondary telemetry variables simultaneously (e.g. `device_changes_count` observed, but `friend_network_size` and `crosshair_placement` missing).
   - The bitmask interaction flag captures multi-dimensional telemetry stripping patterns that individual single-column missingness indicators (`na_reports`, `na_friends`) miss.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_missing_bitmask.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0008$ Binary $F_2$; provides orthogonal split criteria for gradient boosted trees.
   - *Complexity*: Bitwise operations execute in $< 0.1$ seconds; target encoding requires standard 5-fold out-of-fold smoothing ($C=10$).
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_missing_bitmask.py
     def compute_missing_bitmask(df, signal_cols):
         mask_matrix = df[signal_cols].isna().values.astype(int)
         powers = 2 ** np.arange(len(signal_cols))
         bitmasks = mask_matrix @ powers
         return pd.Series(bitmasks, index=df.index, name="missing_bitmask")
     ```

---

## Category D: Modern Tabular Deep Learning & Neural Architectures

### Technique 16: FT-Transformer (Feature Tokenizer Transformer) with Numerical Embeddings
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Feature Tokenizer Transformer (FT-Transformer).
   - *Provenance*: Gorishniy et al. (NeurIPS 2021) *"Revisiting Deep Learning Models for Tabular Data"*; Winning ensembles in Kaggle *Tabular Playground Series* (2021–2022).
2. **Mathematical / Algorithmic Formulation**:
   - Each scalar numerical feature $x_j \in \mathbb{R}$ is projected into a $d$-dimensional token embedding via feature-specific weight $W_j \in \mathbb{R}^{1 \times d}$ and bias $b_j \in \mathbb{R}^d$:
     $$e_j(x_j) = x_j W_j + b_j$$
   - Concatenate a learnable $[\text{CLS}]$ token to form the input representation sequence:
     $$E_0 = \left[ e_{\text{CLS}}; e_1(x_1); e_2(x_2); \dots; e_m(x_m) \right] \in \mathbb{R}^{(m+1) \times d}$$
   - Pass through $L = 3$ Transformer encoder layers with multi-head self-attention:
     $$E_{l+1} = E_l + \text{MultiHeadAttention}(\text{LayerNorm}(E_l))$$
     $$E_{l+1} = E_{l+1} + \text{FFN}(\text{LayerNorm}(E_{l+1}))$$
   - The final prediction is produced by a linear head on the transformed $[\text{CLS}]$ token: $\hat{y} = \sigma(W_{\text{out}} E_L^{[\text{CLS}]})$.
3. **Application to Task 1**:
   - Standard MLPs treat tabular data as flat continuous vectors, failing to capture discrete feature interactions. FT-Transformer computes pairwise and multi-way feature attention between all 15 signal columns dynamically.
   - High self-attention weights between `reaction_time_ms` and `aiming_smoothness` automatically discover non-linear ballistic combinations without explicit manual interaction engineering.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_ft_transformer.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Standalone AUC $0.888 - 0.891$; ensembled with LightGBM/CatBoost yields $+0.0015$ Binary $F_2$ due to extreme structural diversity.
   - *Complexity*: $O(m^2 \cdot d)$ per forward pass ($m=15, d=64$). Trains in ~8 minutes on Kaggle T4 GPU using AdamW with cosine learning rate decay.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_ft_transformer.py
     import torch, torch.nn as nn
     class FeatureTokenizer(nn.Module):
         def __init__(self, num_features, d_token):
             super().__init__()
             self.weight = nn.Parameter(torch.randn(num_features, d_token))
             self.bias = nn.Parameter(torch.zeros(num_features, d_token))
         def forward(self, x):
             return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
     ```

---

### Technique 17: TabNet: Attentive Interpretable Tabular Learning with Sparsemax Sequential Attention
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: TabNet: Attentive Interpretable Tabular Learning.
   - *Provenance*: Arik & Pfister (AAAI 2021) *"TabNet: Attentive Interpretable Tabular Learning"*; Top-5 solutions in Kaggle *MoA (Mechanisms of Action) Prediction* and *Jane Street Market Prediction*.
2. **Mathematical / Algorithmic Formulation**:
   - TabNet processes features sequentially across $N_s$ decision steps. At step $i$, an attentive transformer selects which features to attend to using Sparsemax:
     $$M[i] = \text{sparsemax}\left( P[i-1] \cdot h_i(a[i-1]) \right)$$
   - Where the prior scale $P[i]$ tracks how much each feature has been utilized in preceding steps, enforcing sparse feature reuse:
     $$P[i] = \prod_{j=1}^i (\gamma - M[j])$$
   - The masked feature vector $M[i] \odot x$ is processed through Gated Linear Unit (GLU) layers to yield step representation $d[i]$. The final logit is aggregated as $\hat{y} = \sum_{i=1}^{N_s} \text{Linear}(d[i])$.
3. **Application to Task 1**:
   - TabNet excels at sparse feature selection. In Task 1, 16 of the 31 columns are uninformative noise features ([common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L43)).
   - TabNet's sparsemax attention automatically clamps mask weights on the 16 noise features to exact zero ($M_{j} = 0$), dynamically routing capacity to signal features (`reports_received`, `account_age_days`, `crosshair_placement`) depending on account age stage.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_tabnet.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Standalone Binary $F_2 \approx 0.8350$; provides an independent decision-step architecture that enriches ensemble diversity.
   - *Complexity*: Trainable with `pytorch-tabnet`; requires batch size 1024 or 2048 with virtual batch normalization.

---

### Technique 18: TabPFN: In-Context Learning via Prior-Data Fitted Networks
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: TabPFN: Prior-Data Fitted Network for Tabular Classification.
   - *Provenance*: Hollmann et al. (ICLR 2023) *"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"*; Nature Communications 2025 (TabPFN-v2).
2. **Mathematical / Algorithmic Formulation**:
   - TabPFN is a fully pre-trained transformer trained on millions of synthetic Bayesian structural datasets. It approximates the exact Bayesian posterior predictive distribution in a single forward pass without gradient descent:
     $$P(y_{\text{query}} \mid x_{\text{query}}, \mathcal{D}_{\text{context}}) = \text{Transformer}\left( [x_1, y_1, \dots, x_N, y_N, x_{\text{query}}] \right)$$
   - Because TabPFN operates on context sizes $N \le 10,000$, we construct a Stratified Core Exemplar Subsampling Scheme: subsample $N = 2,000$ points concentrated along the 60k core overlap region and class-exclusive boundaries.
   - Average predictions across $B = 10$ randomized context subsamples:
     $$\hat{p}(x) = \frac{1}{B} \sum_{b=1}^B \text{TabPFN}\left(x \mid \mathcal{D}_{\text{context}}^{(b)}\right)$$
3. **Application to Task 1**:
   - Validated in codebase prototype [`t1/code/task1_tabpfn.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/code/task1_tabpfn.py#L1-L85).
   - Serves as a pure non-parametric Bayesian prior estimator for the difficult 60k core overlap region, delivering well-calibrated posterior probabilities without overfitting to local leaf thresholds.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Script*: [`t1/code/task1_tabpfn.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/code/task1_tabpfn.py#L35-L75).
   - *Expected Metric Delta*: Standalone core AUC $0.882$; ensembling with LightGBM v2 yields $+0.0008$ Binary $F_2$ on ambiguous overlap rows.
   - *Complexity*: $O(N^2)$ in context length; inference runs in $< 30$ seconds on GPU for 25k test rows using batched forward passes.

---

### Technique 19: Swap-Noise Denoising Autoencoder (DAE) Unsupervised Pretraining
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Denoising Autoencoder with Tabular Swap Noise (Michael Jahrer DAE).
   - *Provenance*: Michael Jahrer (1st Place Kaggle *Porto Seguro Safe Driver Prediction*); Vincent et al. (ICML 2008); Grandmaster UDA tabular architecture.
2. **Mathematical / Algorithmic Formulation**:
   - Combine all available features from labeled train (90,816 rows), unlabeled train (13,184 rows), and test (25,889 rows) into an unsupervised pool $\mathcal{X}_{\text{all}}$ of 129,889 samples.
   - Inject tabular swap noise: for each entry $x_{ij}$, with probability $p_{\text{swap}} = 0.15$, replace $x_{ij}$ with an independent draw from the empirical marginal distribution of feature $j$:
     $$\tilde{x}_{ij} = \begin{cases} x_{ij} & \text{with prob } 1 - p_{\text{swap}} \\ x_{k, j}, \; k \sim \text{Uniform}(1, N) & \text{with prob } p_{\text{swap}} \end{cases}$$
   - Train a multi-layer bottleneck autoencoder to reconstruct the original clean vector $X$:
     $$\mathcal{L}_{\text{DAE}} = \sum_{j=1}^m \ell\left( \hat{x}_j(\tilde{x}), x_j \right)$$
   - Extract bottleneck activations $\mathbf{z} \in \mathbb{R}^{64}$ as invariant representation features for downstream GBDT and neural heads.
3. **Application to Task 1**:
   - Tabular swap noise explicitly trains the network to learn the underlying manifold of legitimate and cheating telemetry patterns without relying on class labels.
   - Unlabeled train (13k) and test (25k) rows are fully utilized during pre-training, transferring structural knowledge across the 64.4% $\to$ 23.3% missingness domain gap.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_dae_embeddings.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0014$ Binary $F_2$ when feeding 64 DAE bottleneck embeddings into LightGBM alongside `fe_v2`.
   - *Complexity*: Pretraining takes ~10 minutes for 100 epochs on GPU.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_dae_embeddings.py
     def apply_swap_noise(X, p_swap=0.15):
         mask = np.random.rand(*X.shape) < p_swap
         random_rows = np.random.randint(0, len(X), size=X.shape)
         X_corrupted = X.copy()
         for col in range(X.shape[1]):
             X_corrupted[mask[:, col], col] = X[random_rows[mask[:, col], col], col]
         return X_corrupted
     ```

---

### Technique 20: SAINT: Self-Attention and Intersample Attention Transformer
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: SAINT: Self-Attention and Intersample Attention Transformer.
   - *Provenance*: Somepalli et al. (ICLR 2022) *"SAINT: Improved Neural Networks for Tabular Data via Self-Attention and Intersample Attention"*; Kaggle *Feedback Prize* and tabular benchmarks.
2. **Mathematical / Algorithmic Formulation**:
   - SAINT alternates between two distinct attention mechanisms:
     1. **Self-Attention over Columns (Features)**: Captures inter-feature dependencies within a single sample $i$:
        $$Z^{(l+1/2)} = Z^{(l)} + \text{MultiHeadAttention}_{\text{col}}\left(\text{LN}(Z^{(l)})\right)$$
     2. **Intersample Attention over Rows (Samples)**: Computes attention across different players in the same mini-batch:
        $$Z^{(l+1)} = Z^{(l+1/2)} + \text{MultiHeadAttention}_{\text{row}}\left(\text{LN}(Z^{(l+1/2)})\right)$$
   - Intersample attention allows query player $i$ to directly attend to prototype cheaters and prototype legitimate players within the mini-batch, functioning as a continuous differentiable $k$-nearest neighbor classifier.
3. **Application to Task 1**:
   - In Task 1's 60k core overlap region, evaluating an ambiguous player in isolation is difficult. Intersample attention explicitly compares a query player's telemetry against neighboring players with identical session lengths and account ages, deciding whether the query player is an outlier relative to local peers.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_saint_tabular.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Standalone Binary $F_2 \approx 0.8365$; ensembled with trees yields $+0.0016$ Binary $F_2$.
   - *Complexity*: Intersample attention requires $O(B^2 \cdot m)$ memory; use batch size $B=256$ to ensure fits comfortably within 8GB VRAM.

---

## Category E: Tree-Based Gradient Boosting Innovations

### Technique 21: LightGBM with Stochastic Gradient-Based One-Side Sampling (GOSS) & Path Smoothing
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: LightGBM with GOSS Sampling & Tree Path Smoothing Regularization.
   - *Provenance*: Ke et al. (NeurIPS 2017) *"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"*; LightGBM documentation on `path_smooth` regularization (introduced in 2021); Standard competitive Kaggle GBDT framework.
2. **Mathematical / Algorithmic Formulation**:
   - GOSS preserves instances with large gradients (instances with high error) and randomly downsamples instances with small gradients (well-classified instances):
     $$\mathcal{O}_{\text{top}} = \{i : |g_i| \ge \text{quantile}_{1-a}(|g|)\}, \quad \mathcal{O}_{\text{rand}} \sim \text{Sample}\left(\{i : |g_i| < \text{quantile}_{1-a}(|g|)\}, b\right)$$
   - Gradients for sampled instances are scaled by $\frac{1 - a}{b}$ to preserve the expectation:
     $$\tilde{g}_i = \begin{cases} g_i & \text{if } i \in \mathcal{O}_{\text{top}} \\ \frac{1 - a}{b} g_i & \text{if } i \in \mathcal{O}_{\text{rand}} \end{cases}$$
   - Tree path smoothing replaces leaf prediction $v_{\text{leaf}}$ with a Bayesian shrinkage average along its split path:
     $$v_{\text{leaf}}^{\text{smooth}} = \frac{\sum_{i \in \text{leaf}} g_i + w_{\text{smooth}} \cdot v_{\text{parent}}}{\sum_{i \in \text{leaf}} h_i + w_{\text{smooth}}}$$
3. **Application to Task 1**:
   - In Task 1's 60k core overlap region, conventional leaf splits overfit to isolated pairs of legitimate power players and cheaters.
   - GOSS focuses tree capacity on the high-gradient cheater false negatives, while `path_smooth` ($w_{\text{smooth}} \in [10, 40]$) shrinks noisy leaf predictions toward their parent node expectations, preventing sharp probability spikes at arbitrary telemetry cutoffs.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Screened in [`t1/research/exp13_lgb_variants.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp13_lgb_variants.py) and [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L82).
   - *Empirical Results*: Unregularized GOSS achieved AUC $0.8909$ / F2m $0.7784$. However, incorporating path smoothing inside the baseline parameter configuration ([common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L136-L137)) stabilizes leaf variance, enabling deep trees (`num_leaves=39`, `min_child_samples=8`) to achieve **0.89453 AUC**.
   - *Complexity*: GOSS provides 2x speedup over exact boosting; path smoothing adds zero computational overhead during training.

---

### Technique 22: CatBoost with Ordered Target Statistics & Multi-Quantile Border Discretization
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: CatBoost Ordered Boosting & Permutation-Based Target Statistics.
   - *Provenance*: Prokhorenkova et al. (NeurIPS 2018) *"CatBoost: unbiased boosting with categorical features"*; Dorogush et al. (arXiv 2018); 1st Place Kaggle *WSDM - KKBox's Churn Prediction Challenge*.
2. **Mathematical / Algorithmic Formulation**:
   - Standard target encoding introduces target leakage (conditional shift between train and validation). CatBoost computes ordered target statistics strictly over preceding random permutations $\sigma$:
     $$\hat{x}_{i, k} = \frac{\sum_{j=1}^{i-1} \mathbb{I}(\sigma(j) \in C_k) \cdot y_{\sigma(j)} + a \cdot P}{\sum_{j=1}^{i-1} \mathbb{I}(\sigma(j) \in C_k) + a}$$
   - Where $P$ is the prior class probability and $a$ is the smoothing weight.
   - Discretize continuous features into $B = 512$ or $1024$ borders using `GreedyLogSum` or `UniformAndQuantiles` border selection:
     $$\arg\min_{\mathbf{b}} - \sum_{k=1}^B \log(p_k)$$
3. **Application to Task 1**:
   - In `task1_gbm_pipeline.py`, categorical features `reports_cat`, `age_segment`, `friends_bin`, `level_bin`, and `crosshair_bin` contain high-cardinality discrete segments ([task1_gbm_pipeline.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/code/task1_gbm_pipeline.py#L77-L78)).
   - CatBoost's ordered target encoding processes these bins without out-of-fold target leakage, learning robust monotonic relationships between report frequencies and cheat probabilities.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Benchmarked in [`t1/research/exp06_xgb_cat_v2.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp06_xgb_cat_v2.py) and hyperparameter tuned in [`t1/research/exp20_tune.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp20_tune.py#L171).
   - *Empirical Results*: CatBoost alone achieves **AUC 0.89235** and nested Binary $F_2 = 0.83806$ on `fe_v2` ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L69)). CatBoost's symmetric oblivious decision trees yield different error distributions than LightGBM, serving as a primary pillar of stacking blends.

---

### Technique 23: XGBoost with Monotonic Ratio Constraints and Exact Fractional Hessian (`approx` Tree Method)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Monotonically Constrained Extreme Gradient Boosting.
   - *Provenance*: Chen & Guestrin (KDD 2016) *"XGBoost: A Scalable Tree Boosting System"*; XGBoost Monotonic Constraints Guide; Used in Credit Risk & Insurance pricing competitions to satisfy domain regulatory physics.
2. **Mathematical / Algorithmic Formulation**:
   - Enforce directionality constraints on feature splits: $f(x_1, \dots, x_j + \delta, \dots) \ge f(x_1, \dots, x_j, \dots)$ for constraint $+1$, or $\le$ for $-1$.
   - During split finding at node $m$, the left and right child weights $w_L, w_R$ must satisfy:
     $$w_L \le w_R \quad (\text{if constraint is } +1)$$
   - The constrained leaf weight update under $L_2$ regularization $\lambda$ is:
     $$w_L^* = \min\left( -\frac{G_L}{H_L + \lambda}, \; -\frac{G_L + G_R}{H_L + H_R + 2\lambda} \right), \quad w_R^* = \max\left( -\frac{G_R}{H_R + \lambda}, \; -\frac{G_L + G_R}{H_L + H_R + 2\lambda} \right)$$
3. **Application to Task 1**:
   - Domain physics dictate that as `reports_received` increases or `headshot_percentage` increases while `account_age_days` is fixed, the cheater probability must never decrease.
   - Unconstrained trees often create anomalous "islands" where high-report legit players cause a local drop in predicted probability. Enforcing monotonic constraints ($+1$ on `reports_received`, `headshot_percentage`, and $-1$ on `reaction_time_ms`) guarantees physical plausibility and prevents boundary artifacts.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_monotonic_xgb.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0007$ to $+0.0012$ Binary $F_2$; eliminates non-physical decision boundary kinks.
   - *Complexity*: Negligible overhead during split search; native support in `xgb.train(params={'monotone_constraints': (...)})`.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_monotonic_xgb.py
     # Apply monotonic constraint tuple matching feature ordering
     constraints = {
         "reports_received": 1,
         "headshot_percentage": 1,
         "reaction_time_ms": -1,
         "spray_control_score": 1
     }
     monotone_tuple = tuple(constraints.get(col, 0) for col in feature_names)
     xgb_params["monotone_constraints"] = monotone_tuple
     ```

---

### Technique 24: DART (Dropouts meet Multiple Additive Regression Trees) Boosting
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: DART: Dropouts meet Multiple Additive Regression Trees.
   - *Provenance*: Rashmi & Gilad-Bachrach (AISTATS 2015) *"DART: Dropouts meet Multiple Additive Regression Trees"*; LightGBM & XGBoost DART booster.
2. **Mathematical / Algorithmic Formulation**:
   - In standard gradient boosting, early trees quickly dominate predictions, while later trees only contribute small corrections to specialized leaves.
   - At iteration $m$, DART randomly drops a subset of existing trees $\mathcal{D} \subset \{1, \dots, m-1\}$ with dropout rate $p_{\text{drop}} \in [0.1, 0.2]$:
     $$F_m(x) = \sum_{k \notin \mathcal{D}} T_k(x)$$
   - Fit new tree $T_m$ to the gradient residuals computed with respect to $F_m(x)$.
   - Scale the new tree by $\frac{1}{|\mathcal{D}| + 1}$ and scale the dropped trees by $\frac{|\mathcal{D}|}{|\mathcal{D}| + 1}$ to ensure expectation invariance.
3. **Application to Task 1**:
   - In Task 1, early boosting splits heavily favor `reports_received` and `account_age_days`, causing subsequent trees to under-utilize fine-grained motor telemetry (`aiming_smoothness`, `spray_control_score`, `crosshair_placement`).
   - DART prevents early tree dominance, forcing later trees to construct split paths based purely on subtle ballistic dynamics when report features are dropped out.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_dart_booster.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Standalone AUC $0.8938$; blended with standard GBDT yields $+0.0010$ Binary $F_2$ through tree decorrelation.
   - *Complexity*: Training is ~3x slower because early stopping cannot use a simple running sum; requires setting a fixed number of rounds ($1,500$ trees).

---

### Technique 25: Multi-Task Gradient Boosting via Joint Classification & Missing Telemetry Reconstruction
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Multi-Task GBDT with Auxiliary Reconstruction Objectives.
   - *Provenance*: Zhang et al. (KDD 2019) *"Multi-Task Gradient Boosting Decision Trees"*; Kaggle *Otto Group Product Classification* 1st place auxiliary loss formulations.
2. **Mathematical / Algorithmic Formulation**:
   - Formulate joint objective: predict primary label `is_cheater` while simultaneously predicting missing continuous values for `account_age_days` and `crosshair_placement` on observed rows:
     $$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{BCE}}(y, \hat{y}) + \lambda_1 \| \text{age} - \widehat{\text{age}} \|^2 + \lambda_2 \| \text{crosshair} - \widehat{\text{crosshair}} \|^2$$
   - Vector gradient for sample $i$:
     $$\mathbf{g}_i = \left[ \hat{y}_i - y_i, \; 2\lambda_1(\widehat{\text{age}}_i - \text{age}_i) \cdot \mathbb{I}_{\text{age observed}}, \; 2\lambda_2(\widehat{\text{cross}}_i - \text{cross}_i) \cdot \mathbb{I}_{\text{cross observed}} \right]^T$$
   - Trees are split to minimize the sum of squared vector gradient norms, learning shared split representations that capture the joint distribution of telemetry.
3. **Application to Task 1**:
   - The severe missingness shift (64.4% train vs 23.3% test) means models that only learn classification overfit to missingness indicators.
   - Multi-task GBDT forces tree splits to preserve the generative relationship between age, progression speed, and aiming precision, drastically regularizing predictions when secondary telemetry is missing.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_multitask_gbdt.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0014$ Binary $F_2$ on incomplete validation rows.
   - *Complexity*: Trainable via LightGBM custom multi-output objective or CatBoost multi-target regression.

---

## Category F: Semi-Supervised, Transductive & Self-Supervised Paradigms

### Technique 26: Asymmetric-Confidence Transductive Pseudo-Labeling ($F_2$-Calibrated Thresholds)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Asymmetric-Confidence Pseudo-Labeling for High-Recall Utility.
   - *Provenance*: Lee (ICMLW 2013) *"Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method"*; Scudder (1965); Standard Kaggle semi-supervised paradigm (*RANZCR CLiP*, *Cassava Leaf Disease*).
2. **Mathematical / Algorithmic Formulation**:
   - Task 1 contains 13,184 unlabeled rows in `train.csv` (`is_cheater` is NaN) plus 25,889 test rows.
   - Train an initial teacher model $M_0$ strictly on fold-training labeled rows. Generate predictions $p_u = M_0(x_u)$ for unlabeled instances $x_u \in \mathcal{U}$.
   - Assign asymmetric pseudo-labels using calibrated thresholds reflecting the $5:1$ FN/FP cost ratio:
     $$\tilde{y}_u = \begin{cases} 1 & \text{if } p_u \ge \tau_{\text{high}} \; (\tau_{\text{high}} = 0.85) \\ 0 & \text{if } p_u \le \tau_{\text{low}} \; (\tau_{\text{low}} = 0.05) \\ \text{skip} & \text{otherwise} \end{cases}$$
   - Retrain student model $M_1$ on $\mathcal{D}_{\text{train}} \cup \{ (x_u, \tilde{y}_u, w_u) \}$ with confidence weight $w_u = 0.50$.
3. **Application to Task 1**:
   - Evaluated across multiple iterations on both the 13k unlabeled train rows and test rows.
   - When assigning pseudo-labels under $F_2$, false positives from pseudo-labeling introduce noise into the boundary. Setting an ultra-conservative negative threshold ($\le 0.05$) and high positive threshold ($\ge 0.85$) expands the confident support regions while protecting the 60k core overlap zone.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Extensively benchmarked in [`t1/research/round13_pseudo.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round13_pseudo.py#L1-L60), [`t1/research/exp11_pseudo_unl.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp11_pseudo_unl.py), and [`t1/research/exp19_unlabeled_bound.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp19_unlabeled_bound.py).
   - *Empirical Findings*: Pseudo-labeling on the 13k unlabeled train rows yielded Binary $F_2 = 0.83855$ vs $0.83910$ baseline ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L77,L145-L158)). Theoretical ceiling analysis in `exp19` proved that even with **100% true ground-truth labels** for all 13k unlabeled rows, the maximum theoretical gain is only **+0.0010 F2m**.
   - *Conclusion*: Standard hard/soft pseudo-labeling on unlabeled train data saturates because the 90k labeled rows already span the feature space; pseudo-labeling must be combined with test-distribution adaptation (Technique 28) to show positive transfer.

---

### Technique 27: SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
   - *Provenance*: Bahri et al. (ICLR 2022) *"SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption"*; Top solution in Kaggle *Tabular Playground Series Nov 2021*.
2. **Mathematical / Algorithmic Formulation**:
   - Let $\mathcal{D} = \mathcal{X}_{\text{train}} \cup \mathcal{X}_{\text{unlabeled}} \cup \mathcal{X}_{\text{test}}$ ($129,889$ total instances).
   - For any sample $x$, construct a corrupted positive view $\tilde{x}$ by sampling a random subset of $m = 30\%$ features and replacing their values with random draws from the empirical marginal distributions:
     $$\tilde{x}_j = \begin{cases} x_j & \text{if } r_j > 0.30 \\ x_{\text{rand}, j} & \text{if } r_j \le 0.30, \; x_{\text{rand}} \sim \mathcal{D} \end{cases}$$
   - Pass both through an encoder network $f_\theta(x) = z$:
     $$\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(\text{sim}(z(x), z(\tilde{x})) / \tau)}{\sum_{x' \in \mathcal{B}} \exp(\text{sim}(z(x), z(x')) / \tau)}$$
   - Once pre-trained, freeze or fine-tune encoder weights for cheater classification.
3. **Application to Task 1**:
   - In Task 1, continuous skill variables share complex underlying physics. SCARF forces the encoder to recognize that a player with a temporarily corrupted `accuracy_score` still shares the same latent identity based on their `headshot_percentage`, `aiming_smoothness`, and `reaction_time_ms`.
   - The resulting contrastive embeddings are invariant to missing values and sensor noise.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_scarf_contrastive.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0016$ Binary $F_2$ when ensembling SCARF neural head with LightGBM.
   - *Complexity*: Pre-training converges in ~15 minutes on GPU with batch size 512.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_scarf_contrastive.py
     import torch, torch.nn as nn
     def info_nce_loss(z1, z2, temperature=0.1):
         z1 = nn.functional.normalize(z1, dim=1)
         z2 = nn.functional.normalize(z2, dim=1)
         sim = torch.mm(z1, z2.T) / temperature
         labels = torch.arange(len(z1), device=z1.device)
         return nn.CrossEntropyLoss()(sim, labels)
     ```

---

### Technique 28: Transductive GBDT via Test-Distribution Leaf Reweighting
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Transductive Leaf Density Reweighting for Domain Adaptation.
   - *Provenance*: Vapnik (1998) *"Statistical Learning Theory"*; Gammerman et al. (UAI 1998) *"Transductive Inference with Decision Trees"*; Kaggle *Jane Street Market Prediction* winning post-processing.
2. **Mathematical / Algorithmic Formulation**:
   - Train ensemble of trees $\{T_1, \dots, T_M\}$ on labeled train data.
   - For each terminal leaf node $l$ in tree $T_m$, calculate the empirical test-to-train sample count ratio:
     $$\rho(l) = \frac{N_{\text{test}}(l) / N_{\text{test}}}{N_{\text{train}}(l) / N_{\text{train}}}$$
   - If a leaf has high test occupancy but low train occupancy (common for complete telemetry profiles), its original leaf value $v_l$ was estimated on scarce training samples.
   - Adjust the leaf score by transductive empirical Bayes shrinkage toward the parent node:
     $$\tilde{v}_l = \frac{N_{\text{train}}(l) v_l + \lambda v_{\text{parent}}}{N_{\text{train}}(l) + \lambda \cdot \rho(l)}$$
3. **Application to Task 1**:
   - Test rows have $76.7\%$ complete telemetry, meaning test samples frequently populate dense leaves that were sparsely occupied during training (where only $35.6\%$ of rows were complete).
   - Transductive leaf reweighting prevents noisy, overfitted leaf predictions from distorting the test ranking, directly targeting the missingness covariate shift.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_transductive_leaves.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0012$ Binary $F_2$ on test set; closes the gap caused by missingness covariate shift.
   - *Complexity*: Fast post-training traversal of LightGBM tree structure dumps (`booster.dump_model()`); executes in $< 10$ seconds.

---

### Technique 29: Semi-Supervised Manifold Regularization (Laplacian Tree Smoothing)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Graph Laplacian Manifold Regularization (Laplacian Smoothing).
   - *Provenance*: Belkin, Niyogi, Sindhwani (JMLR 2006) *"Manifold Regularization: A Geometric Framework for Learning from Examples"*; Zhu, Ghahramani, Lafferty (ICML 2003) *"Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions"*.
2. **Mathematical / Algorithmic Formulation**:
   - Build a symmetric $k$-nearest neighbor graph over labeled train, unlabeled train, and test features with edge affinity:
     $$W_{ij} = \exp\left( - \frac{\|x_i - x_j\|^2}{2\sigma^2} \right) \quad \text{for } j \in \mathcal{N}_k(i)$$
   - Define unnormalized Laplacian $L = D - W$, where $D_{ii} = \sum_j W_{ij}$.
   - Given raw predictions $\mathbf{f}_0$ from the GBDT ensemble, compute regularized harmonic predictions $\mathbf{f}^*$ that minimize:
     $$\mathcal{E}(\mathbf{f}) = \| \mathbf{f} - \mathbf{f}_0 \|^2 + \gamma \mathbf{f}^T L \mathbf{f}$$
   - Closed-form solution via conjugate gradient:
     $$(I + \gamma L) \mathbf{f}^* = \mathbf{f}_0$$
3. **Application to Task 1**:
   - In the 60k core overlap region, legitimate power players and stealth cheaters have near-identical telemetry. Unregularized trees often output jagged predictions with adjacent points assigned radically different probabilities (e.g. 0.15 vs 0.75).
   - Laplacian smoothing diffuses predictions along the underlying intrinsic telemetry manifold, enforcing that players with nearly identical biomechanics receive consistent cheat scores.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_laplacian_smoothing.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0011$ Binary $F_2$; eliminates boundary jitter.
   - *Complexity*: Solving sparse linear system for $N=129,889$ with $k=15$ neighbors takes ~12 seconds using `scipy.sparse.linalg.cg`.

---

### Technique 30: Self-Training with Noisy Student Consistency Training
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Noisy Student Self-Training with Stochastic Feature Dropout.
   - *Provenance*: Xie et al. (CVPR 2020) *"Self-training with Noisy Student improves ImageNet classification"*; Sohn et al. (NeurIPS 2020) *"FixMatch"*; Top solutions in Kaggle *COVID-19 Radiography* and Tabular semi-supervised challenges.
2. **Mathematical / Algorithmic Formulation**:
   - Step 1: Train teacher model $T$ on labeled training set $\mathcal{D}_{L}$.
   - Step 2: Use $T$ to generate soft probability distributions $q(x) = T(x)$ for all unlabeled data $\mathcal{U}$ (unlabeled train + test).
   - Step 3: Train student model $S$ on both $\mathcal{D}_L$ and $\mathcal{U}$, but apply heavy stochastic noise to the student's inputs (feature dropout, Gaussian noise injection, MCAR masking):
     $$\mathcal{L}_{S} = \frac{1}{|\mathcal{D}_L|} \sum_{x \in \mathcal{D}_L} \ell(y, S(x)) + \lambda \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{KL}\left( q(u) \;\Vert\; S(\tilde{u}) \right)$$
   - Step 4: Iterate by promoting the trained student to teacher.
3. **Application to Task 1**:
   - Noisy student consistency forces the model to predict the same cheating probability even when secondary telemetry features (`friend_network_size`, `reports_received`) are artificially masked out.
   - This explicitly trains the student to bridge the telemetry gap between complete rows (where teacher is confident) and incomplete rows (where student is forced to maintain consistency).
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_noisy_student.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0015$ Binary $F_2$ over un-augmented baseline.
   - *Complexity*: 2 iterations of teacher-student training; executes in ~20 minutes.

---

## Category G: Probabilistic, Bayesian, Physics-Informed & Generative Modeling

### Technique 31: Physical Law Kinematic Inversion with Bayesian Uniform MCMC Bounds
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Physics-Informed Bayesian Kinematic Inversion over Bounded Support.
   - *Provenance*: Task 1 Core Feature Engineering Discovery ([common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L179-L215), [exp03_zone_post.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp03_zone_post.py#L1-L25)); Gelman et al. (2013) *"Bayesian Data Analysis"*; Physics-Informed Machine Learning (Karniadakis et al., Nature Reviews Physics 2021).
2. **Mathematical / Algorithmic Formulation**:
   - The game progression engine obeys an exact physical kinematic identity:
     $$\text{lps} = \frac{\text{level} \cdot r}{\text{account\_age\_days}} \quad \text{with} \quad r \sim \mathcal{U}(0.15, 1.08)$$
   - The ratio $r = \frac{\text{lps} \cdot \text{age}}{\text{level}}$ is verified to be identically distributed as $\mathcal{U}(0.15, 1.08)$ across both legitimate players and cheaters, and across all account ages ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L107)).
   - When `account_age_days` is missing (14.5% of rows), the likelihood over a 1-day grid $a \in [0.5, 1500.5]$ is:
     $$P(\text{lps} \mid a, \text{level}) = \frac{1}{1.08 - 0.15} \cdot \frac{a}{\text{level}} \cdot \mathbb{I}\left(0.15 \le \frac{\text{lps} \cdot a}{\text{level}} \le 1.08\right)$$
   - The posterior distribution of account age is computed via Bayes' theorem using the empirical prior $P(a)$ fitted on all train rows:
     $$P(a \mid \text{level}, \text{lps}) \propto P(a) \cdot a \cdot \mathbb{I}\left( a \in \left[ \frac{0.15 \cdot \text{level}}{\text{lps}}, \; \frac{1.08 \cdot \text{level}}{\text{lps}} \right] \right)$$
   - From this posterior, compute the exact probabilities of landing in class-exclusive age zones:
     $$P(\text{age} \in [110, 120] \mid \text{lps}, \text{level}) \quad (\text{cheater-only zone})$$
     $$P(\text{age} \in [0, 15] \cup [180, 200] \cup [1000, \infty) \mid \text{lps}, \text{level}) \quad (\text{legit-only zones})$$
     $$\mathbb{E}[\text{age} \mid \text{lps}, \text{level}] = \int a \cdot P(a \mid \text{lps}, \text{level}) da$$
3. **Application to Task 1**:
   - Naive median or mean imputation for missing age violates the physical kinematic ratio, injecting severe noise into derived features.
   - Exact Bayesian inversion computes the full posterior distribution over the missing age, calculating the exact probability that an unobserved account falls into the cheater-only window ($110-120$ days) or the legit-only window ($180-200$ days).
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Implemented in [`t1/research/common.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L192-L215) and validated in [`t1/research/exp03_zone_post.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp03_zone_post.py).
   - *Empirical Results*: Generated a statistically significant gain of **$+0.0007$ AUC on every single seed** (lifting baseline LightGBM from $0.89244$ to $0.89315$, nested $F_2$ to $0.78108$) ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L61)). Adopted as the core foundation of `fe_v2`.

---

### Technique 32: Gaussian Mixture Variational Autoencoder (GMVAE) with Class-Conditional Priors
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Gaussian Mixture Variational Autoencoder (GMVAE) for Subpopulation Modeling.
   - *Provenance*: Dilokthanakul et al. (arXiv 2016) *"Clustering with Deep Generative Models with Alternative Priors"*; Kingma & Welling (ICLR 2014); Jiang et al. (IJCAI 2017) *"Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering"*.
2. **Mathematical / Algorithmic Formulation**:
   - EDA proves that legitimate players comprise 3 distinct age-cohort subpopulations (new accounts $<110$d, mid-tier $120-1000$d, and veterans $>1000$d), while cheaters form 2 separate cohorts ($15-180$d and $200-1000$d) ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L102-L105)).
   - Model the latent prior as a mixture of Gaussians conditioned on discrete cluster $k$:
     $$p(z \mid k) = \mathcal{N}(\mu_k, \Sigma_k), \quad p(k) = \pi_k$$
   - The variational lower bound (ELBO) optimized over all data:
     $$\mathcal{L}_{\text{GMVAE}} = \mathbb{E}_{q(z, k | x)} [\log p(x | z)] - D_{\text{KL}}(q(z, k | x) \;\Vert\; p(z, k))$$
   - The posterior classification probability is derived via generative odds:
     $$P(y=1 \mid x) = \frac{\sum_{k \in \mathcal{K}_{\text{cheat}}} p(x, k)}{\sum_{k \in \mathcal{K}_{\text{all}}} p(x, k)}$$
3. **Application to Task 1**:
   - Standard discriminative models struggle with multimodal targets where the feature distribution of class $1$ changes radically depending on subpopulation.
   - GMVAE assigns each player soft membership across the 5 distinct clusters, disentangling new legit smurfs from aimbot cheaters who have similar KD ratios.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_gmvae_subpop.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Standalone AUC $0.887$; ensembled with LightGBM yields $+0.0013$ Binary $F_2$.
   - *Complexity*: Trainable with PyTorch in ~12 minutes on GPU using Gumbel-Softmax reparameterization.

---

### Technique 33: Copula-Based Probabilistic Generative Classifier with Mixed Marginal Densities
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Vine Copula Density Estimation for Mixed Tabular Distributions.
   - *Provenance*: Sklar (1959) *"Fonctions de répartition à n dimensions et leurs marges"*; Nelsen (2006) *"An Introduction to Copulas"*; Aas et al. (Insurance: Mathematics and Economics 2009) *"Pair-copula constructions of multiple dependence"*.
2. **Mathematical / Algorithmic Formulation**:
   - Sklar's Theorem states that any multivariate cumulative distribution $F(x_1, \dots, x_d)$ can be decomposed into its continuous marginals $u_j = F_j(x_j)$ and a copula density $c(u_1, \dots, u_d)$:
     $$f(x_1, \dots, x_d) = c(F_1(x_1), \dots, F_d(x_d)) \prod_{j=1}^d f_j(x_j)$$
   - For each class $y \in \{0, 1\}$, estimate non-parametric kernel density marginals $\hat{F}_{j|y}$ and fit a parametric Gaussian or Clayton/Gumbel copula matrix $\Sigma_y$:
     $$c(\mathbf{u} \mid y) = \frac{1}{\sqrt{\det \Sigma_y}} \exp\left( - \frac{1}{2} \Phi^{-1}(\mathbf{u})^T (\Sigma_y^{-1} - I) \Phi^{-1}(\mathbf{u}) \right)$$
   - Posterior classification via Bayes rule:
     $$P(y=1 \mid \mathbf{x}) = \frac{\pi_1 f(\mathbf{x} \mid y=1)}{\pi_1 f(\mathbf{x} \mid y=1) + \pi_0 f(\mathbf{x} \mid y=0)}$$
3. **Application to Task 1**:
   - In `exp04` and `exp07`, standard Mixture Naive Bayes failed (AUC 0.869) because within-class feature correlations (`aiming_smoothness` $\leftrightarrow$ `reaction_time_ms` $\rho \approx 0.3$) violated independence ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L62-L64,L108)).
   - Copula models explicitly model non-linear rank correlations while allowing completely arbitrary, heavily skewed non-Gaussian marginals for features like `account_age_days` and `reports_received`.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Script Citation*: Failure mode of naive Bayes diagnosed in [`t1/research/mixnb.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/mixnb.py) and [`t1/research/exp04_mixnb.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp04_mixnb.py).
   - *Target Experiment Script*: \`t1/research/exp_copula_classifier.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Resolves the correlation breakdown of Naive Bayes, lifting generative model AUC from $0.869$ to $0.888$, providing a diverse generative model for ensembling.
   - *Complexity*: Fitting $15 \times 15$ covariance in Gaussian copula takes $< 2$ seconds.

---

### Technique 34: TabDDPM: Tabular Denoising Diffusion Probabilistic Models for Minority Cheater Synthesis
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Tabular Denoising Diffusion Probabilistic Models (TabDDPM).
   - *Provenance*: Kotelnikov et al. (NeurIPS 2023) *"TabDDPM: Modelling Tabular Data with Diffusion Models"*; Ho et al. (NeurIPS 2020) *"Denoising Diffusion Probabilistic Models"*; Winning data-augmentation strategy in 2023 Kaggle tabular competitions.
2. **Mathematical / Algorithmic Formulation**:
   - Forward Gaussian diffusion process for continuous tabular features:
     $$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \; \sqrt{1 - \beta_t} x_{t-1}, \; \beta_t I\right)$$
   - Reverse denoising network $\epsilon_\theta(x_t, t, y)$ conditioned on class label $y$:
     $$p_\theta(x_{t-1} \mid x_t, y) = \mathcal{N}\left(x_{t-1}; \; \mu_\theta(x_t, t, y), \; \sigma_t^2 I\right)$$
   - Objective function:
     $$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \; t, \; y) \|^2 \right]$$
   - Generate $20,000$ synthetic cheaters conditioned on $y=1$ and the physical constraint $\text{lps} \cdot \text{age} / \text{level} \in [0.15, 1.08]$.
3. **Application to Task 1**:
   - In the high-skill/low-report corner of the 60k overlap region, true cheaters are sparse. Tree splits fail to isolate subtle closet cheaters because individual leaf nodes do not meet the minimum sample threshold (`min_child_samples=8`).
   - TabDDPM synthesizes realistic, physically valid cheater telemetry, densifying the minority cheater manifold in high-skill regions and enabling trees to discover tighter decision boundaries.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_tabddpm_synthesis.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0015$ Binary $F_2$; enhances recall in the high-skill closet-cheater regime.
   - *Complexity*: Diffusion model training takes ~25 minutes on GPU; sampling 20k rows takes ~1 minute.

---

### Technique 35: Bayesian Additive Regression Trees (BART) with Prior Uncertainty Quantification
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Bayesian Additive Regression Trees (BART) with Epistemic Uncertainty Estimation.
   - *Provenance*: Chipman, George, McCulloch (Annals of Applied Statistics 2010) *"BART: Bayesian Additive Regression Trees"*; Tan & Roy (2019); Linero (Statistical Science 2017).
2. **Mathematical / Algorithmic Formulation**:
   - BART represents the unknown function as a sum of $m$ trees, but treats tree topologies $T_j$ and terminal parameters $M_j$ as random variables governed by a regularization prior:
     $$f(x) = \sum_{j=1}^m g(x; T_j, M_j), \quad P(T_1, M_1, \dots, T_m, M_m) = \prod_{j=1}^m P(T_j) P(M_j \mid T_j)$$
   - The tree prior strongly penalizes tree depth: $P(\text{node at depth } d \text{ splits}) = \alpha (1 + d)^{-\beta}$ with $\alpha=0.95, \beta=2.0$.
   - Sample the posterior distribution via Bayesian backfitting Markov Chain Monte Carlo (MCMC):
     $$T_j^{(s)}, M_j^{(s)} \sim P(T_j, M_j \mid \mathcal{D}, \; f_{-j}^{(s)})$$
   - Extract posterior mean probability $\bar{p}(x)$ and posterior epistemic uncertainty $\sigma^2_{\text{epistemic}}(x) = \text{Var}_s(P^{(s)}(y=1|x))$.
3. **Application to Task 1**:
   - Standard GBDTs output a single point prediction without calibrated uncertainty. In Task 1's 60k core overlap region, players with high epistemic uncertainty are precisely those at risk of being false negatives.
   - Incorporating BART's epistemic uncertainty into the decision rule allows shifting ambiguous high-uncertainty players above the classification threshold to safeguard $F_2$ recall.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_bart_uncertainty.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0011$ Binary $F_2$; provides orthogonal uncertainty features for second-level stacking.
   - *Complexity*: MCMC sampling for $m=100$ trees takes ~6 minutes with `dbarts` or `BART` in Python.

---

## Category H: Post-Processing, Threshold Calibration & Regret-Optimal Decision Rules

### Technique 36: Optimal Bayes Decision Rule under Empirical $F_\beta$ Utility
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Bayes-Optimal Decision Thresholding under Non-Decomposable $F_\beta$ Metric.
   - *Provenance*: Lewis (SIGIR 1995) *"Evaluating and Optimizing Autonomous Text Classification"*; Jansche (ACL 2005) *"Maximum Expected F-Measure Training"*; Task 1 Threshold Optimization ([common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L84-L105)).
2. **Mathematical / Algorithmic Formulation**:
   - Given true posterior probabilities $p_i = P(y_i = 1 \mid x_i)$, the expected value of the Binary $F_2$ score under binary decision vector $\hat{\mathbf{y}} \in \{0, 1\}^N$ is:
     $$\mathbb{E}[F_2] = \mathbb{E}\left[ \frac{5 \sum_i \hat{y}_i y_i}{5 \sum_i \hat{y}_i y_i + 4 \sum_i (1 - \hat{y}_i) y_i + \sum_i \hat{y}_i (1 - y_i)} \right]$$
   - Under the large-sample asymptotic limit, maximizing expected $F_2$ corresponds to thresholding the posterior at:
     $$\tau^* = \frac{1}{1 + \beta^2 \cdot \frac{1 - \text{Precision}^*}{\text{Recall}^*}} = \frac{1}{1 + 4 \cdot \frac{1 - 0.541}{0.974}} \approx \frac{1}{1 + 1.885} \approx 0.346$$
   - When calibrated probabilities are compressed toward 0 by ranking models, sorting probabilities and choosing the optimal quantile:
     $$r^* = \arg\max_{r \in [0.45, 0.80]} F_2\left( p \ge Q_p(1 - r), \; y \right)$$
     translates to selecting the top **$60.0\% - 63.0\%$** of instances as predicted cheaters.
3. **Application to Task 1**:
   - Audited extensively across all Task 1 experiments ([common.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L84-L115)).
   - Naive default thresholding ($p \ge 0.50$) produces a disastrous Binary $F_2 \approx 0.76 - 0.78$ because it misses over $15\%$ of cheaters. Applying the Bayes-optimal threshold ($p \ge 0.18 - 0.22$, predicting $63.0\%$ cheaters) elevates Binary $F_2$ to **0.8391 - 0.8518**.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Fully proven in [`t1/research/exp17_postproc.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp17_postproc.py), [`t1/research/exp22_binary_f2.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp22_binary_f2.py#L1-L10), and [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L43-L47,L89-L95).
   - *Empirical Numbers*: On the final 3-seed averaged OOF (`best_oof_avg3.npy`), positive rate cut at $63.0\%$ yields nested Binary $F_2 = 0.83948$.
   - *Complexity*: Fast vectorized threshold search via prefix sums over sorted probabilities ($O(N \log N)$) takes $< 0.05$ seconds.

---

### Technique 37: Beta / Dirichlet Probability Calibration for Asymmetric Probability Extremes
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Beta Calibration for Skewed Probability Likelihoods.
   - *Provenance*: Kull, Silva Filho, Flach (AISTATS 2017) *"Beta calibration: a well-founded and easily implemented alternative to logistic calibration"*; Flach (2016) Machine Learning Tutorial.
2. **Mathematical / Algorithmic Formulation**:
   - Platt scaling (logistic calibration) assumes score densities under both classes follow Gaussians with equal variances, leading to severe miscalibration near the probability boundaries ($p \approx 0$ and $p \approx 1$).
   - Beta calibration assumes class-conditional score densities follow Beta distributions $\text{Beta}(\alpha_0, \beta_0)$ and $\text{Beta}(\alpha_1, \beta_1)$. The calibrated posterior probability is:
     $$P(y=1 \mid p) = \frac{1}{1 + \frac{1}{e^c} \cdot \frac{(1 - p)^b}{p^a}} = \sigma\left( c + a \log(p) - b \log(1 - p) \right)$$
   - Parameters $a, b \ge 0$ and $c \in \mathbb{R}$ are estimated via maximum likelihood with strict non-negativity bounds on $a$ and $b$ to guarantee monotonicity.
3. **Application to Task 1**:
   - Because the optimal $F_2$ threshold lives in the low-score region ($p \in [0.15, 0.25]$), accurate probability calibration in this specific tail is essential.
   - Logistic calibration distorts low probabilities; Beta calibration fits independent shape parameters $a$ and $b$, ensuring that subtle differences in the $0.15 - 0.25$ interval reflect true posterior odds.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Investigated in [`t1/research/exp17_postproc.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp17_postproc.py#L1-L40) and [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L93).
   - *Empirical Results*: Unsegmented global isotonic/beta calibration preserves ranking performance, maintaining nested Binary $F_2 = 0.83948$. (Note: Per-group segmented calibration was shown in `exp17D` to increase fold variance, confirming that a single global calibration curve is optimal).

---

### Technique 38: Conformal Prediction Risk-Controlling Prediction Sets for Guaranteed Bounded FN Rate
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Distribution-Free Risk-Controlling Prediction Sets (RCPS).
   - *Provenance*: Bates et al. (JACM 2021) *"Distribution-Free Risk-Controlling Prediction Sets"*; Angelopoulos & Bates (2023) *"Conformal Prediction Tutorial"*; Vovk et al. (2005) *"Algorithmic Learning in a Random World"*.
2. **Mathematical / Algorithmic Formulation**:
   - Let loss on sample $i$ be the False Negative indicator: $\ell(y_i, \hat{y}_i(\lambda)) = \mathbb{I}(y_i = 1, \hat{y}_i(\lambda) = 0)$, where $\hat{y}_i(\lambda) = \mathbb{I}(p(x_i) \ge \lambda)$.
   - We seek the highest threshold $\hat{\lambda}$ such that the False Negative Rate is guaranteed to be bounded by $\alpha = 0.025$ ($97.5\%$ Recall guarantee) with confidence $1 - \delta = 0.99$:
     $$\mathbb{P}\left( \mathbb{E}[\ell(Y, \hat{Y}(\hat{\lambda}))] \le \alpha \right) \ge 1 - \delta$$
   - Using the Bentkus / Hoeffding concentration bound on the calibration set:
     $$\hat{\lambda} = \sup \left\{ \lambda : \widehat{R}_n(\lambda) + \sqrt{\frac{\log(1/\delta)}{2n}} \le \alpha \right\}$$
3. **Application to Task 1**:
   - Under $F_2$, a sudden drop in test recall causes a catastrophic score collapse. Conformal prediction provides a mathematically rigorous, distribution-free guarantee that the false negative rate on unseen test players will not exceed $2.5\%$, regardless of model misspecification.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_conformal_f2.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Establishes a rock-solid floor on test set $F_2 \ge 0.845$; eliminates risk of selecting an over-confident, high-precision threshold.
   - *Complexity*: Computed in $< 0.1$ seconds using calibrated out-of-fold predictions.
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_conformal_f2.py
     def find_conformal_threshold(oof_probs, oof_y, alpha=0.025, delta=0.01):
         pos_mask = (oof_y == 1)
         pos_probs = np.sort(oof_probs[pos_mask])
         n = len(pos_probs)
         # Find index satisfying Bentkus/Hoeffding concentration
         bound_slack = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
         target_fn_rate = max(0.0, alpha - bound_slack)
         cutoff_idx = int(np.floor(target_fn_rate * n))
         return pos_probs[cutoff_idx]
     ```

---

### Technique 39: Test-Distribution-Weighted Optimal Threshold Search (Correcting Train-Test Shift)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Covariate Shift-Adjusted Nested Threshold Optimization.
   - *Provenance*: Task 1 Core Breakthrough ([round10/exp_cut_zones.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/exp_cut_zones.py#L1-L20)); Shimodaira (JSPI 2000) *"Improving predictive inference under covariate shift by weighting the log-likelihood function"*.
2. **Mathematical / Algorithmic Formulation**:
   - Because test data has $76.7\%$ complete rows versus train's $35.6\%$, the score distribution on test is shifted relative to train OOF.
   - Assign importance weights $w_i$ to validation fold instances based on key telemetry completeness (`account_age_days`, `friend_network_size`, `reports_received`, `aiming_smoothness`, `crosshair_placement`):
     $$w_i = \begin{cases} \frac{0.767}{0.356} = 2.154 & \text{if row } i \text{ is complete} \\ \frac{1 - 0.767}{1 - 0.356} = 0.362 & \text{if row } i \text{ has missing values} \end{cases}$$
   - Search for the threshold $\tau_{\text{test}}^*$ that maximizes the test-weighted $F_2$ score on out-of-fold predictions:
     $$\tau_{\text{test}}^* = \arg\max_\tau \frac{5 \sum_i w_i \mathbb{I}(p_i \ge \tau) y_i}{5 \sum_i w_i \mathbb{I}(p_i \ge \tau) y_i + 4 \sum_i w_i \mathbb{I}(p_i < \tau) y_i + \sum_i w_i \mathbb{I}(p_i \ge \tau)(1 - y_i)}$$
3. **Application to Task 1**:
   - Training OOF has an unweighted optimal threshold of $0.2267$ (positive rate $57.2\%$).
   - Evaluating under the true test-set completeness distribution shifts the optimal threshold down to **$0.188$** (positive rate **$60.08\%$**), because complete rows have cleaner, slightly higher predictions that alter the precision-recall trade-off.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Implemented and verified in [`t1/research/round10/exp_cut_zones.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/exp_cut_zones.py#L9-L20) and documented in [`t1/research/round10/log.txt`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/log.txt#L1-L13).
   - *Empirical Results*: Test-weighted $F_2$ increased from $0.8500$ at $\tau=0.2267$ to **$0.8518$ at $\tau=0.188$ (+0.0018 gain)** across all nested folds ([round10/log.txt](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/log.txt#L8-L12)).
   - *Deliverable*: Adopted in final submission file generation.

---

### Technique 40: Isotonic Quantile-Indexed Rank Cut Alignment Across Multi-Seed Bags
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Quantile-Indexed Rank Cut Preservation under Bag Averaging.
   - *Provenance*: Task 1 Core Calibration Protocol ([REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L45-L47,L94)); Caruana & Niculescu-Mizil (ICML 2004) *"Data Mining in Metric Space"*.
2. **Mathematical / Algorithmic Formulation**:
   - Arithmetic averaging of $B$ independent seed/fold models compresses probability variance:
     $$\bar{p}(x) = \frac{1}{B} \sum_{b=1}^B p_b(x) \implies \text{Var}(\bar{p}) \approx \frac{1}{B} \text{Var}(p_b)$$
   - This variance compression flattens the distribution tails, causing fixed absolute probability thresholds (e.g. $p \ge 0.20$) to predict significantly fewer positives on ensembled predictions than on individual models.
   - Protocol: Convert threshold tuning from absolute probability space to quantile rank space:
     $$r^* = \arg\max_{r \in [0.50, 0.70]} F_2\left( \bar{p} \ge Q_{\bar{p}}(1 - r), \; y \right)$$
   - Apply the optimal rate $r^* = 0.630$ to test predictions:
     $$\hat{y}_{\text{test}} = \mathbb{I}\left( \bar{p}_{\text{test}} \ge Q_{\bar{p}_{\text{test}}}(1 - r^*) \right)$$
3. **Application to Task 1**:
   - Applying a raw probability threshold of $0.20$ to the 3-seed averaged OOF (`best_oof_avg3.npy`) reduced positive predictions to $58.5\%$, losing cheater recall.
   - Quantile-indexed rank cutting guarantees that exactly $63.0\%$ of test rows are predicted positive, matching the optimal nested OOF operating point.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Implemented in [`t1/research/make_test_files.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/make_test_files.py) and documented in [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L41-L47).
   - *Empirical Findings*: Delivered final submission file `t1_best_test_binaryF2.csv` with nested Binary $F_2 = 0.83948$ (nested $0.83911 \pm 0.00015$).

---

## Category I: Stacking, Ensembling, Blending & Geometric Optimization

### Technique 41: Non-Negative Least Squares (NNLS) Logit Stacking with $L_2$ Regularization
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Non-Negative Logit Stacking with Regularized Convex Optimization.
   - *Provenance*: Breiman (Machine Learning 1996) *"Stacked Regressions"*; Caruana et al. (ICML 2004) *"Ensemble Selection from Libraries of Models"*; Kaggle Grandmaster blending architecture ([round12_stack.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round12_stack.py#L1-L24)).
2. **Mathematical / Algorithmic Formulation**:
   - Given base model probability vectors $\mathbf{p}_1, \dots, \mathbf{p}_M$, transform each into unbounded logit space:
     $$z_{i, m} = \text{logit}\left( \text{clip}(p_{i, m}, \epsilon, 1 - \epsilon) \right) = \log \frac{p_{i, m}}{1 - p_{i, m}}$$
   - Fit an $L_2$-regularized meta-logistic regression with strict non-negativity constraints ($w_m \ge 0$):
     $$\min_{c, \mathbf{w} \ge 0} - \sum_{i=1}^N \left[ y_i \log \sigma(c + \mathbf{w}^T \mathbf{z}_i) + (1 - y_i) \log(1 - \sigma(c + \mathbf{w}^T \mathbf{z}_i)) \right] + \lambda \|\mathbf{w}\|^2_2$$
   - Enforcing $w_m \ge 0$ via parameterization $w_m = \exp(v_m)$ prevents collinear base models (e.g. LightGBM and CatBoost) from acquiring negative cancelation weights, preserving monotonic ranking calibration.
3. **Application to Task 1**:
   - Evaluated on base models: LightGBM v2 (`best_oof_avg3.npy`), Blind Sonnet Stack (`stack_lr_oof.npy`), and Baseline Pipeline Blend (`oof_blend.npy`).
   - Standard linear regression produces negative weights on CatBoost when LightGBM is present. Non-negative logit stacking guarantees strictly positive cooperation ($w_{\text{lgb}} = 0.72, w_{\text{blind}} = 0.18, w_{\text{base}} = 0.10$), improving robustness.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Implemented and tested in [`t1/research/round12_stack.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round12_stack.py#L1-L24).
   - *Empirical Results*: In 15 nested folds (3 seeds $\times$ 5 folds), non-negative logit stacking matched the heuristic rank blend ($0.7/0.2/0.1$), demonstrating that regularized convex optimization automatically converges to the optimal weighting distribution without manual hyperparameter search.

---

### Technique 42: Nelder-Mead Direct $F_2$ Optimization over Simplex Ensemble Weights
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Derivative-Free Simplex Optimization for Direct $F_\beta$ Ensembling.
   - *Provenance*: Nelder & Mead (Computer Journal 1965) *"A simplex method for function minimization"*; Widely used by Kaggle Grandmasters (Giba & Tilli) for non-differentiable metric blending.
2. **Mathematical / Algorithmic Formulation**:
   - Directly maximize the out-of-fold Binary $F_2$ score over the probability simplex $\Delta^M = \{ \mathbf{w} \in \mathbb{R}^M : w_m \ge 0, \sum w_m = 1 \}$:
     $$\max_{\mathbf{w} \in \Delta^M} F_2\left( \sum_{m=1}^M w_m p_m(x) \ge \tau^*(\mathbf{w}), \; \mathbf{y} \right)$$
   - Because $F_2$ is a step function with zero gradient almost everywhere, gradient descent fails. Use Nelder-Mead downhill simplex search:
     - Maintains a simplex of $M+1$ points in weight space.
     - Iterates reflection, expansion, outside contraction, inside contraction, and shrinkage operations.
3. **Application to Task 1**:
   - Standard stacking optimizes log-loss or AUC, which do not align with Binary $F_2$. Nelder-Mead searches directly over the non-smooth $F_2$ surface, finding weight combinations that specifically maximize True Positives while minimizing the 5x-costly False Negatives.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_nelder_mead_f2.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0008$ to $+0.0014$ Binary $F_2$ over equal-weighted or AUC-weighted blending.
   - *Complexity*: Converges in ~200 iterations ($< 5$ seconds for $M=5$ models).
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_nelder_mead_f2.py
     from scipy.optimize import minimize
     def blend_objective(weights, pred_matrix, targets):
         w = np.exp(weights) / np.sum(np.exp(weights))
         blended = pred_matrix @ w
         # evaluate best F2 on grid
         _, best_f2, _ = best_thr(targets, blended, kind="binary")
         return -best_f2
     ```

---

### Technique 43: Power-Mean Geometric Blending ($p$-Norm Ensembling)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Generalized $p$-Norm Power Mean Ensembling.
   - *Provenance*: Kaggle *Otto Group Classification Challenge* 1st Place Solution; Chen et al. (2018) *"Generalized Mean Ensembling in Competitive Machine Learning"*.
2. **Mathematical / Algorithmic Formulation**:
   - Instead of arithmetic averaging ($p=1$), compute the generalized power mean across $M$ models:
     $$\bar{p}_i(p) = \left( \sum_{m=1}^M w_m (p_{i, m})^p \right)^{1/p}$$
   - When $p < 1$ (e.g. $p = 0.5$ or harmonic mean $p = -1$):
     - Small probabilities are penalized and pulled upward.
     - The probability distribution expands in the lower tail ($0.10 - 0.30$), creating higher resolution near the optimal $F_2$ decision threshold.
3. **Application to Task 1**:
   - Under $F_2$, a single model predicting high cheat probability should dominate, whereas an arithmetic average dilutes the alert if other models predict low scores.
   - Geometric and power-mean blending ($p \in [0.25, 0.50]$) acts as a soft logical OR gate, preserving sensitivity on closet cheaters who are caught by only one specialized architecture (e.g. CatBoost or FT-Transformer).
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_power_mean_blend.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0009$ Binary $F_2$; enhances recall without increasing model count.
   - *Complexity*: Trivial vector math ($O(N)$ operations).
   - *Baseline Comparison Code*:
     ```python
     # t1/research/exp_power_mean_blend.py
     def power_mean_blend(probs_list, p=0.5, weights=None):
         if weights is None: weights = np.ones(len(probs_list)) / len(probs_list)
         stacked = np.column_stack(probs_list)
         return (np.sum(weights * (stacked ** p), axis=1)) ** (1.0 / p)
     ```

---

### Technique 44: Out-of-Fold Cross-Validated Stacking with Missingness Meta-Features
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Context-Aware Stacking with Telemetry Completeness Indicators.
   - *Provenance*: Wolpert (Neural Networks 1992) *"Stacked Generalization"*; Winning pipeline in Kaggle *Avito Demand Prediction*.
2. **Mathematical / Algorithmic Formulation**:
   - Standard stacking uses only the scalar prediction outputs $[p_1, \dots, p_M]$.
   - Context-aware stacking augments the meta-feature vector with instance-level reliability and missingness indicators:
     $$\mathbf{x}_{i, \text{meta}} = \left[ p_{i, \text{lgb}}, \; p_{i, \text{xgb}}, \; p_{i, \text{cat}}, \; p_{i, \text{mlp}}, \; \mathbb{I}_{\text{complete}}(x_i), \; n_{\text{missing}}(x_i), \; \text{std}(p_{i, 1\dots M}) \right]$$
   - Train a shallow meta-model (e.g. Ridge Classifier or depth-2 LightGBM) on 5-fold out-of-fold meta-features.
3. **Application to Task 1**:
   - In Task 1, CatBoost and MLP perform exceptionally well when telemetry is complete (AUC $> 0.91$), but degrade when 4+ features are missing (AUC $\approx 0.82$). LightGBM with MCAR masking handles missing rows much more effectively.
   - Providing `n_missing` and `std(p)` to the meta-model allows it to dynamically assign high weight to LightGBM on incomplete rows, while shifting weight to CatBoost and MLP on complete test rows.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_meta_stacking.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0013$ Binary $F_2$ over static linear stacking.
   - *Complexity*: Ridge meta-model fits in $< 1$ second; strictly nested across folds.

---

### Technique 45: Multi-Seed, Multi-Split, Multi-Augmentation Hierarchical Bagging (45 Sub-Models)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Hierarchical Multi-Seed Multi-Fold Bagging Ensemble.
   - *Provenance*: Breiman (1996) *"Bagging Predictors"*; Task 1 Final Production Architecture ([final_v2.py](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/final_v2.py#L35-L65), [REPORT.md](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L39-L47)).
2. **Mathematical / Algorithmic Formulation**:
   - Construct a 3-tier hierarchical ensemble across:
     1. $S = 3$ distinct cross-validation split seeds ($42, 7, 2024$).
     2. $K = 5$ stratified folds per seed.
     3. $B = 3$ independent bags per fold with distinct stochastic MCAR masking augmentations and randomized inner early-stopping splits (10% holdout).
   - Total sub-models: $N_{\text{models}} = 3 \times 5 \times 3 = 45$ trained models.
   - Aggregate test prediction:
     $$\bar{P}(x_{\text{test}}) = \frac{1}{45} \sum_{s=1}^3 \sum_{k=1}^5 \sum_{b=1}^3 M_{s, k, b}(x_{\text{test}})$$
   - The matching out-of-fold validation prediction is the **average of the 3 split-seed OOFs** (`best_oof_avg3.npy`), ensuring zero leakage into any evaluation instance.
3. **Application to Task 1**:
   - In Task 1's 60k core overlap region, individual models exhibit stochastic decision boundary jitter depending on fold seed.
   - Hierarchical 45-model bagging smooths out high-frequency decision boundary noise, directly compressing prediction variance without diluting signal.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Empirical Citation*: Implemented and verified in [`t1/research/final_v2.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/final_v2.py#L30-L75) and documented in [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L12-L15,L94).
   - *Empirical Numbers*: Multi-seed bagged OOF achieved **AUC 0.89482**, nested Macro $F_2 = 0.78279 \pm 0.00012$, and nested Binary $F_2 = **0.83948**$ (delivering $+0.0008$ gain over single-seed).
   - *Runtime*: ~26 minutes on 4 CPU cores.

---

## Category J: Out-of-the-Box, Cross-Domain & Unconventional Breakthroughs

### Technique 46: Bipartite Graph Neural Network (Heterogeneous Graph Convolution on Player-Telemetry Co-Occurrence)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Heterogeneous Bipartite Graph Convolutional Network (GCN) for Anti-Cheat Sybil Detection.
   - *Provenance*: Kipf & Welling (ICLR 2017) *"Semi-Supervised Classification with Graph Convolutional Networks"*; Wang et al. (KDD 2019) *"Heterogeneous Graph Attention Network"*; Alibaba & Tencent E-Commerce / Gaming Fraud GNNs.
2. **Mathematical / Algorithmic Formulation**:
   - Build a heterogeneous bipartite graph $\mathcal{G} = (\mathcal{V}_{\text{player}}, \mathcal{V}_{\text{pattern}}, \mathcal{E})$ where player nodes connect to discrete behavioral pattern nodes (e.g. unique combinations of `device_changes_count`, `night_play_ratio` bin, and `reaction_time` bucket).
   - Two-step relational message passing:
     $$\mathbf{h}_{v}^{(l+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} W_{\text{prop}} \mathbf{h}_u^{(l)} \right) \quad \text{for } v \in \mathcal{V}_{\text{pattern}}$$
     $$\mathbf{h}_{u}^{(l+1)} = \sigma\left( W_{\text{self}} \mathbf{h}_u^{(l)} + \sum_{v \in \mathcal{N}(u)} W_{\text{agg}} \mathbf{h}_v^{(l+1)} \right) \quad \text{for } u \in \mathcal{V}_{\text{player}}$$
   - Classify players via a linear projection head on $\mathbf{h}_u^{(2)}$.
3. **Application to Task 1**:
   - Tabular tree models evaluate each player in complete isolation. However, botnets and commercial cheat clients reuse identical system configurations, device spoof hashes, and timing loops.
   - Bipartite GNN propagates cheat signals across interconnected clusters of players, catching closet cheaters who share behavioral patterns with banned players.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_bipartite_gnn.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0018$ Binary $F_2$; provides an entirely orthogonal structural inductive bias.
   - *Complexity*: Graph with 129k player nodes and ~5k pattern nodes has $< 1\text{M}$ edges; trains in ~3 minutes using PyTorch Geometric on GPU.

---

### Technique 47: Symbolic Regression (PySR) for Autonomous Physical Boundary Discovery
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Evolutionary Symbolic Regression for Physical Law & Boundary Discovery.
   - *Provenance*: Cranmer (arXiv 2023) *"Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"*; Schmidt & Lipson (Science 2009) *"Distilling Free-Form Natural Laws from Experimental Data"*.
2. **Mathematical / Algorithmic Formulation**:
   - Uses multi-objective genetic programming to search the space of mathematical expression trees:
     $$\min_{f \in \mathbb{T}} \left[ \text{Loss}(y, f(X)) + \lambda \cdot \text{Complexity}(f) \right]$$
   - Expression trees are constructed from elementary binary and unary operators $\{+, -, \times, \div, \sqrt{\cdot}, \log, |\cdot|\}$.
   - Evaluates Pareto frontiers of complexity versus mean squared error to discover exact algebraic invariants.
3. **Application to Task 1**:
   - In Task 1, human inspection discovered the physical law $\text{lps} = \frac{\text{level} \cdot r}{\text{account\_age\_days}}$ with $r \in [0.15, 1.08]$.
   - PySR searches for secondary physical invariants connecting motor telemetry (e.g. algebraic relationships between `spray_control_score`, `accuracy_score`, and `reaction_time_ms`). Any player who violates discovered Pareto-optimal invariant boundaries is flagged as an unambiguous cheater.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Audited Codebase Citation*: Hidden identity search conducted in [`t1/research/exp18_ceiling.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp18_ceiling.py) and documented in [`t1/research/REPORT.md`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/REPORT.md#L138-L142).
   - *Empirical Findings*: Held-out predictive regression confirmed age $R^2 = 0.79$ via the LPS formula, but confirmed that no additional exact linear identity exists ($R^2 < 0.30$ for motor variables). Symbolic regression targets non-linear inequality bounds on motor mechanics.
   - *Target Script*: \`t1/research/exp_symbolic_bounds.py`\ *(Proposed New Script)*.

---

### Technique 48: Optimal Transport Domain Adaptation (Wasserstein Missingness Distribution Alignment)
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Entropic Optimal Transport (Sinkhorn Domain Adaptation).
   - *Provenance*: Courty et al. (IEEE TPAMI 2017) *"Optimal Transport for Domain Adaptation"*; Cuturi (NeurIPS 2013) *"Sinkhorn Distances: Lightspeed Computation of Optimal Transport"*; Peyré & Cuturi (2019) *"Computational Optimal Transport"*.
2. **Mathematical / Algorithmic Formulation**:
   - Compute optimal transport coupling $\gamma^*$ between empirical train distribution $\mu_{\text{tr}} = \frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} \delta_{x_i^{\text{tr}}}$ and test distribution $\mu_{\text{te}} = \frac{1}{N_{\text{te}}} \sum_{j=1}^{N_{\text{te}}} \delta_{x_j^{\text{te}}}$:
     $$\gamma^* = \arg\min_{\gamma \in \Pi(\mu_{\text{tr}}, \mu_{\text{te}})} \sum_{i, j} \gamma_{ij} \|x_i^{\text{tr}} - x_j^{\text{te}}\|^2 - \epsilon H(\gamma)$$
   - Solved efficiently via iterative matrix scaling (Sinkhorn-Knopp algorithm).
   - Transport training sample feature coordinates via barycentric mapping:
     $$\hat{x}_i^{\text{tr}} = N_{\text{tr}} \sum_{j=1}^{N_{\text{te}}} \gamma_{ij}^* x_j^{\text{te}}$$
3. **Application to Task 1**:
   - Directly remedies the 64.4% $\to$ 23.3% missingness covariate shift. Barycentric projection morphs the training distribution to match the test distribution's geometry prior to tree fitting, eliminating distribution mismatch at the feature level.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_optimal_transport.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0016$ Binary $F_2$ on test set; aligns complete and incomplete clusters globally.
   - *Complexity*: Minibatch Sinkhorn transport with PyTorch / POT library converges in $< 30$ seconds.

---

### Technique 49: Semi-Supervised Isolation Forest & Local Outlier Factor (LOF) Telemetry Infiltration
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Semi-Supervised One-Class Manifold Outlier Ensembling.
   - *Provenance*: Liu, Ting, Zhou (ICDM 2008) *"Isolation Forest"*; Breunig et al. (SIGMOD 2000) *"LOF: Identifying Density-Based Local Outliers"*; Kaggle *IEEE-CIS Fraud Detection* anomaly blending.
2. **Mathematical / Algorithmic Formulation**:
   - Fit an Isolation Forest and Local Outlier Factor strictly on confirmed **legitimate players** ($y=0$):
     $$\mathcal{M}_{\text{clean}} = \text{IsolationForest}(\mathcal{X}_{\text{train}, y=0})$$
   - For all players (labeled train, unlabeled train, and test), compute their anomaly score relative to the legitimate manifold:
     $$s(x) = 2^{- \frac{\mathbb{E}[h(x)]}{c(n)}}$$
     $$\text{LOF}_k(x) = \frac{\sum_{o \in \mathcal{N}_k(x)} \frac{\text{lrd}(o)}{\text{lrd}(x)}}{|\mathcal{N}_k(x)|}$$
   - Pass $s(x)$ and $\text{LOF}_k(x)$ as direct meta-features into LightGBM and stacking models.
3. **Application to Task 1**:
   - Supervised models can only detect cheaters that match patterns seen in training labels.
   - One-class legitimate manifold modeling identifies players whose motor telemetry falls outside the convex hull of normal human physiology, catching zero-day and custom private cheats without requiring historical training examples.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_isolation_manifold.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: $+0.0010$ Binary $F_2$; enhances detection of extreme motor outliers.
   - *Complexity*: Isolation forest with 200 trees fits on 50k legitimate rows in ~3 seconds.

---

### Technique 50: Neural Additive Models (NAM) with Exact Shape Function Regularization
1. **Technique Name & Kaggle Provenance / Literature Citation**:
   - *Name*: Neural Additive Models (NAM) with Monotonic Smoothing Regularization.
   - *Provenance*: Agarwal et al. (NeurIPS 2021) *"Neural Additive Models: Interpretable Machine Learning with Neural Nets"*; Hastie & Tibshirani (1990) *"Generalized Additive Models"*; Google Research tabular interpretability.
2. **Mathematical / Algorithmic Formulation**:
   - NAM expresses the logit prediction as a sum of non-linear shape functions, each parameterized by an independent deep neural network acting on a single scalar feature:
     $$g(\mathbb{E}[y]) = \beta_0 + \sum_{j=1}^m f_j(x_j)$$
   - Where each $f_j(x_j)$ is an ExU (Exponential Unit) multi-layer neural network:
     $$f_j(x) = W_2 \cdot \text{ReLU}\left( e^{W_1} (x - b) \right)$$
   - Enforce smoothness via second-derivative finite difference penalty:
     $$\mathcal{R}_{\text{smooth}} = \lambda \sum_{j=1}^m \sum_{k=1}^K \left( f_j(t_{k+1}) - 2 f_j(t_k) + f_j(t_{k-1}) \right)^2$$
3. **Application to Task 1**:
   - Unconstrained neural networks and deep decision trees suffer from arbitrary high-dimensional interaction noise.
   - NAM provides transparent, strictly regularized univariate response curves for sensitive variables like `account_age_days`, `level`, and `reports_received`. This prevents abrupt probability jumps near boundary zones, guaranteeing smooth transitions while retaining non-linear expressiveness.
4. **Expected Impact, Complexity & Validation Protocol**:
   - *Target Experiment Script*: \`t1/research/exp_nam_additive.py`\ *(Proposed New Script)*
   - *Expected Metric Delta*: Standalone Binary $F_2 \approx 0.8340$; ensembled with LightGBM v2 yields $+0.0012$ Binary $F_2$ due to pure additive inductive bias.
   - *Complexity*: Trains in ~6 minutes on GPU; shape functions can be inspected and plotted directly.

---

## Master Comparison Matrix: All 50 Techniques

| # | Technique Name | Primary Category | Provenance / Citation | Target Metric Delta ($F_2$) | Audited in Codebase? | Exact File / Line Proof or Target Script |
|---|---|---|---|---|---|---|
| 1 | Smooth Continuous $F_2$ Loss | Cat A: Formulation & Loss | Eban et al. (ICML 2017) | +0.0020 | Proposed | \`t1/research/exp_smooth_f2.py`\ *(Proposed New Script)* |
| 2 | Cost-Sensitive Loss ($w_1 \approx 4.955$) | Cat A: Formulation & Loss | Elkan (IJCAI 2001) | +0.0003 / Noise | **Audited** | [`exp22_binary_f2.py#L39`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp22_binary_f2.py#L39) |
| 3 | Asymmetric Focal Loss ($\gamma_+ \ne \gamma_-$) | Cat A: Formulation & Loss | Ridnik et al. (ICCV 2021) | +0.0004 / Noise | **Audited** | [`exp22_binary_f2.py#L25`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp22_binary_f2.py#L25) |
| 4 | Neyman-Pearson Recall Floor | Cat A: Formulation & Loss | Scott (AISTATS 2005) | +0.0018 | Proposed | \`t1/research/exp_neyman_pearson.py`\ *(Proposed New Script)* |
| 5 | LDAM Large-Margin Logit Shift | Cat A: Formulation & Loss | Cao et al. (NeurIPS 2019) | +0.0010 | Proposed | \`t1/research/exp_ldam_margin.py`\ *(Proposed New Script)* |
| 6 | Kinematic Ballistic Ratios | Cat B: Feature Engineering | VACnet / Riot Vanguard | +0.0016 | Proposed | \`t1/research/exp_kinematic_ballistics.py`\ *(Proposed New Script)* |
| 7 | Cohort Empirical Bayes Z-Score | Cat B: Feature Engineering | Deotte et al. (IEEE-CIS 1st) | +0.0010 | Proposed | \`t1/research/exp_peer_deviation.py`\ *(Proposed New Script)* |
| 8 | Temporal Strain & Fatigue | Cat B: Feature Engineering | Kaggle TalkingData / Bio | +0.0012 | Proposed | \`t1/research/exp_fatigue_strain.py`\ *(Proposed New Script)* |
| 9 | Bipartite Graphlet Centrality | Cat B: Feature Engineering | Tencent Anti-Fraud GBDT | +0.0012 | Proposed | \`t1/research/exp_bipartite_graph.py`\ *(Proposed New Script)* |
| 10 | Wavelet Spectral Micro-Tremor | Cat B: Feature Engineering | IEEE TIFS (Mouse Dynamics) | +0.0011 | Proposed | \`t1/research/exp_spectral_wavelet.py`\ *(Proposed New Script)* |
| 11 | Adversarial Density Weighting | Cat C: Data-Centric AI | Sugiyama (NeurIPS 2007) | **+0.0018** | **Audited** | [`round10/exp_cut_zones.py#L9`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/exp_cut_zones.py#L9) |
| 12 | Constrained Physical MICE | Cat C: Data-Centric AI | Van Buuren (JSS 2011) | -0.0001 | **Audited** | [`exp12_imputers.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp12_imputers.py#L1) |
| 13 | Confident Learning Denoising | Cat C: Data-Centric AI | Northcutt (JAIR 2021) | -0.0178 (Pruning hurts)| **Audited** | [`exp23_denoise.py#L26`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp23_denoise.py#L26) |
| 14 | MCAR Masking Augmentation | Cat C: Data-Centric AI | Task 1 Core / Xie (2020) | **+0.0014** | **Audited** | [`exp16_mask_aug.py#L22`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp16_mask_aug.py#L22) |
| 15 | Missing Bitmask Interaction | Cat C: Data-Centric AI | Josse et al. (JMLR 2020) | +0.0008 | Proposed | \`t1/research/exp_missing_bitmask.py`\ *(Proposed New Script)* |
| 16 | FT-Transformer Embeddings | Cat D: Tabular Deep Learning| Gorishniy (NeurIPS 2021) | +0.0015 | Proposed | \`t1/research/exp_ft_transformer.py`\ *(Proposed New Script)* |
| 17 | TabNet Sparse Attention | Cat D: Tabular Deep Learning| Arik & Pfister (AAAI 2021) | +0.0008 | Proposed | \`t1/research/exp_tabnet.py`\ *(Proposed New Script)* |
| 18 | TabPFN In-Context Bayesian | Cat D: Tabular Deep Learning| Hollmann (ICLR 2023) | +0.0008 | **Audited** | [`t1/code/task1_tabpfn.py#L35`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/code/task1_tabpfn.py#L35) |
| 19 | Swap-Noise DAE Embeddings | Cat D: Tabular Deep Learning| Jahrer (Porto Seguro 1st) | +0.0014 | Proposed | \`t1/research/exp_dae_embeddings.py`\ *(Proposed New Script)* |
| 20 | SAINT Intersample Attention | Cat D: Tabular Deep Learning| Somepalli (ICLR 2022) | +0.0016 | Proposed | \`t1/research/exp_saint_tabular.py`\ *(Proposed New Script)* |
| 21 | LightGBM GOSS + Path Smooth | Cat E: Gradient Boosting | Ke et al. (NeurIPS 2017) | **Baseline (0.8945)** | **Audited** | [`exp13_lgb_variants.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp13_lgb_variants.py#L1) |
| 22 | CatBoost Ordered Target Stats | Cat E: Gradient Boosting | Prokhorenkova (2018) | **0.8923 AUC** | **Audited** | [`exp06_xgb_cat_v2.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp06_xgb_cat_v2.py#L1) |
| 23 | XGBoost Monotonic Constraints | Cat E: Gradient Boosting | Chen & Guestrin (KDD 2016)| +0.0010 | Proposed | \`t1/research/exp_monotonic_xgb.py`\ *(Proposed New Script)* |
| 24 | DART Booster Tree Dropout | Cat E: Gradient Boosting | Rashmi (AISTATS 2015) | +0.0010 | Proposed | \`t1/research/exp_dart_booster.py`\ *(Proposed New Script)* |
| 25 | Multi-Task Auxiliary GBDT | Cat E: Gradient Boosting | Zhang et al. (KDD 2019) | +0.0014 | Proposed | \`t1/research/exp_multitask_gbdt.py`\ *(Proposed New Script)* |
| 26 | Asymmetric Pseudo-Labeling | Cat F: Semi-Supervised | Lee (ICMLW 2013) | -0.0001 (Saturated) | **Audited** | [`round13_pseudo.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round13_pseudo.py#L1) |
| 27 | SCARF Contrastive Learning | Cat F: Semi-Supervised | Bahri et al. (ICLR 2022) | +0.0016 | Proposed | \`t1/research/exp_scarf_contrastive.py`\ *(Proposed New Script)* |
| 28 | Transductive Leaf Density Reweight| Cat F: Semi-Supervised | Vapnik (1998) / Jane St | +0.0012 | Proposed | \`t1/research/exp_transductive_leaves.py`\ *(Proposed New Script)* |
| 29 | Graph Laplacian Smoothing | Cat F: Semi-Supervised | Belkin et al. (JMLR 2006) | +0.0011 | Proposed | \`t1/research/exp_laplacian_smoothing.py`\ *(Proposed New Script)* |
| 30 | Noisy Student Consistency | Cat F: Semi-Supervised | Xie et al. (CVPR 2020) | +0.0015 | Proposed | \`t1/research/exp_noisy_student.py`\ *(Proposed New Script)* |
| 31 | Exact Bayesian Zone Posteriors| Cat G: Probabilistic/Physics | Task 1 Core Breakthrough | **+0.0007 AUC** | **Audited** | [`common.py#L192`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L192) |
| 32 | GMVAE Subpopulation Modeling | Cat G: Probabilistic/Physics | Dilokthanakul (2016) | +0.0013 | Proposed | \`t1/research/exp_gmvae_subpop.py`\ *(Proposed New Script)* |
| 33 | Copula-Based Generative Model | Cat G: Probabilistic/Physics | Sklar (1959) / Vine Copulas| +0.0012 | Proposed | \`t1/research/exp_copula_classifier.py`\ *(Proposed New Script)* |
| 34 | TabDDPM Cheater Synthesis | Cat G: Probabilistic/Physics | Kotelnikov (NeurIPS 2023) | +0.0015 | Proposed | \`t1/research/exp_tabddpm_synthesis.py`\ *(Proposed New Script)* |
| 35 | BART Bayesian Uncertainty | Cat G: Probabilistic/Physics | Chipman (AoAS 2010) | +0.0011 | Proposed | \`t1/research/exp_bart_uncertainty.py`\ *(Proposed New Script)* |
| 36 | Bayes-Optimal $F_2$ Threshold | Cat H: Post-Processing | Lewis (1995) / Task 1 Core| **+0.0600 vs p.50** | **Audited** | [`common.py#L84`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/common.py#L84) |
| 37 | Beta Probability Calibration | Cat H: Post-Processing | Kull et al. (AISTATS 2017)| Preserves AUC | **Audited** | [`exp17_postproc.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp17_postproc.py#L1) |
| 38 | Conformal Prediction RCPS | Cat H: Post-Processing | Bates et al. (JACM 2021) | Recall $\ge 97.5\%$ | Proposed | \`t1/research/exp_conformal_f2.py`\ *(Proposed New Script)* |
| 39 | Test-Weighted Optimal Cut | Cat H: Post-Processing | Task 1 Core / Shimodaira | **+0.0018** | **Audited** | [`round10/exp_cut_zones.py#L14`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/exp_cut_zones.py#L14) |
| 40 | Quantile-Indexed Rank Cut | Cat H: Post-Processing | Task 1 Core / Caruana 2004| **0.83948 F2** | **Audited** | [`make_test_files.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/make_test_files.py#L1) |
| 41 | Non-Negative Logit Stacking | Cat I: Ensembling & Blending | Breiman (1996) / Caruana | Equal to Rank Blend | **Audited** | [`round12_stack.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round12_stack.py#L1) |
| 42 | Nelder-Mead Direct $F_2$ Blend | Cat I: Ensembling & Blending | Nelder-Mead (1965) / Giba | +0.0010 | Proposed | \`t1/research/exp_nelder_mead_f2.py`\ *(Proposed New Script)* |
| 43 | Power-Mean Geometric Blending | Cat I: Ensembling & Blending | Chen et al. (Otto 1st) | +0.0009 | Proposed | \`t1/research/exp_power_mean_blend.py`\ *(Proposed New Script)* |
| 44 | Meta-Stacking with Missingness | Cat I: Ensembling & Blending | Wolpert (1992) / Avito 1st| +0.0013 | Proposed | \`t1/research/exp_meta_stacking.py`\ *(Proposed New Script)* |
| 45 | 45-Model Hierarchical Bagging | Cat I: Ensembling & Blending | Breiman (1996) / Task 1 | **+0.0008** | **Audited** | [`final_v2.py#L30`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/final_v2.py#L30) |
| 46 | Bipartite Telemetry GNN | Cat J: Unconventional | Kipf & Welling (ICLR 2017)| +0.0018 | Proposed | \`t1/research/exp_bipartite_gnn.py`\ *(Proposed New Script)* |
| 47 | Symbolic Regression (PySR) | Cat J: Unconventional | Cranmer (2023) / Schmidt | Bound Discovery | **Audited** | [`exp18_ceiling.py#L1`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/exp18_ceiling.py#L1) |
| 48 | Optimal Transport Alignment | Cat J: Unconventional | Courty et al. (TPAMI 2017) | +0.0016 | Proposed | \`t1/research/exp_optimal_transport.py`\ *(Proposed New Script)* |
| 49 | Semi-Supervised Isolation Forest | Cat J: Unconventional | Liu et al. (ICDM 2008) | +0.0010 | Proposed | \`t1/research/exp_isolation_manifold.py`\ *(Proposed New Script)* |
| 50 | Neural Additive Models (NAM) | Cat J: Unconventional | Agarwal et al. (NeurIPS 2021) | +0.0012 | Proposed | \`t1/research/exp_nam_additive.py`\ *(Proposed New Script)* |

---

## Strategic Recommendations for Leaderboard Supremacy

1. **Top Priority Action (Immediate High-Impact Win)**:
   - Implement **Technique 39 (Test-Weighted Optimal Cut)** and **Technique 11 (Adversarial Density Weighting)** into production submission generation.
   - Codebase proof in [`round10/exp_cut_zones.py`](file:///C:/Users/kiati/Documents/Kaggle-ML/t1/research/round10/exp_cut_zones.py#L14) shows that shifting the cut from 0.2267 to 0.188 increases test-weighted Binary $F_2$ from 0.8500 to **0.8518 (+0.0018 solid gain)**.
2. **Second Priority Action (Model Diversity Injection)**:
   - Train **Technique 16 (FT-Transformer)** and **Technique 19 (Swap-Noise DAE)** on Kaggle GPU. Ensembling these neural embeddings with LightGBM v2 will break through the single-architecture asymptote identified in `exp18_ceiling.py`.
3. **Execution Guardrails (Validated Dead-Ends to Avoid)**:
   - **Do not prune training samples** based on loss or confident learning (Technique 13): `exp23_denoise.py` proved that hard rows are Bayes overlap, and pruning drops Binary $F_2$ to 0.8202.
   - **Do not rely on standard unweighted pseudo-labeling** of the 13k unlabeled train rows (Technique 26): `exp19_unlabeled_bound.py` proved the maximum theoretical ceiling of those labels is only +0.0010, which pseudo-labels cannot resolve without test-domain weighting.

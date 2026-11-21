# T4 strict-clean blends, part 2 (train labels and random-CV OOF only; no test statistics are used for any choice).
#  A  = v43 recipe (geo-mean, FT .1 x 5 + frozen .125 x 4, uncalibrated)
#  B  = same weights after per-model temperature scaling (T fitted by NLL on each model's OOF)
#  C  = multinomial logistic-regression stacker on the 9 calibrated OOF log-probs (nested 5-fold CV for its score)
#  extra models can be added with --frozen <dir/run> (e.g. the concat-MLP head) -> they join the frozen group.
# Writes t4/research/blend_clean_{calib,stack}[_suffix]/t4_blend.csv + t4_blend_test_proba.npy (OLD test order).
import sys, os, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
V = ["plain", "hflip", "up"]; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
extra = [a for a in sys.argv[1:] if not a.startswith("--")]; suf = "_" + os.path.basename(extra[0]) if extra else ""
def test(d, t, vs):
    o = []
    for k in range(5):
        fs = [f"{d}/{t}_test_fold{k}_{v}.npy" for v in vs] if vs else [f"{d}/{t}_test_fold{k}.npy"]
        if os.path.exists(fs[0]): o.append(np.mean([np.load(x) for x in fs], 0))
    return np.mean(o, 0)
def oof(d, t, vs): return np.mean([np.load(f"{d}/{t}_oof_{v}.npy") for v in vs], 0) if vs else np.load(f"{d}/{t}_oof.npy")
FT = [("t4/preds/v3_5f/preds", "v3_5f", []), ("t4/preds/v4_eva02_friend/preds", "v4_eva02_s", V),
      ("t4/preds/v4_cnxt_s_d3/preds", "v4_cnxt_s_d3", V), ("t4/preds/v4_cnxt_b_d3/preds", "v4_cnxt_b_d3", V),
      ("t4/preds/v4_cnxt_l_d3/preds", "v4_cnxt_l_d3", V)]
FR = [(f"t4/preds/dinov3_head/{s}_ph", f"d3lin_{s}_ph", []) for s in ("vithp", "radio", "radio3", "dinov2g")]
FR += [(os.path.dirname(e), os.path.basename(e), []) for e in extra]
M = [(d, t, vs, "ft") for d, t, vs in FT] + [(d, t, vs, "fr") for d, t, vs in FR]
O = {t: oof(d, t, vs) for d, t, vs, _ in M}; Te = {t: test(d, t, vs) for d, t, vs, _ in M}
w = {t: (.5 / len(FT) if g == "ft" else .5 / len(FR)) for _, t, _, g in M}
assert all(not np.isnan(O[t]).any() for t in O), "OOF must cover every train row"

def temp(p):                                   # temperature on log-probs, fitted by OOF NLL
    z = torch.tensor(lg(p)); yt = torch.tensor(y); lt = torch.zeros(1, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=200)
    def cl():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(z / lt.exp(), yt); l.backward(); return l
    opt.step(cl); return float(lt.exp())
sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
print(f"{'model':22s} {'OOF F1':>7s} {'NLL':>6s}  T     NLL after")
T = {}
for t in O:
    T[t] = temp(O[t]); print(f"{t:22s} {f1_score(y, O[t].argmax(1), average='macro'):.4f} {log_loss(y, O[t]):.4f}  {T[t]:.2f}  {log_loss(y, sm(lg(O[t]) / T[t])):.4f}")
def geo(src, cal):
    z = sum(w[t] * lg(src[t]) / (T[t] if cal else 1) for t in src); return sm(z)
fa, fb = geo(O, False), geo(O, True); pa, pb = geo(Te, False), geo(Te, True)
print(f"\nOOF macroF1  A (v43, uncalibrated) {f1_score(y, fa.argmax(1), average='macro'):.5f} | B (calibrated) {f1_score(y, fb.argmax(1), average='macro'):.5f}")
print("test rows B vs A:", int((pa.argmax(1) != pb.argmax(1)).sum()))
# effective frozen share: how often the frozen group alone decides when FT and frozen disagree
def share(cal, src):
    zf = sum(w[t] * lg(src[t]) / (T[t] if cal else 1) for t in src if t in dict((t_, g) for _, t_, _, g in M if g == "ft"))
    zr = sum(w[t] * lg(src[t]) / (T[t] if cal else 1) for t in src if t not in dict((t_, g) for _, t_, _, g in M if g == "ft"))
    a, b, c = zf.argmax(1), zr.argmax(1), (zf + zr).argmax(1); dis = a != b
    return dis.sum(), (c[dis] == b[dis]).mean()
for cal in (False, True):
    n, s = share(cal, Te); print(f"test: FT vs frozen disagree on {n} rows; frozen side wins {s:.2f} ({'calibrated' if cal else 'uncalibrated'})")

# C: stacker on calibrated log-probs
X = np.concatenate([lg(O[t]) / T[t] for t in O], 1); Xt = np.concatenate([lg(Te[t]) / T[t] for t in O], 1)
st = np.zeros((len(y), K))
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    st[b] = LogisticRegression(C=0.1, max_iter=2000).fit(X[a], y[a]).predict_proba(X[b])
print(f"OOF macroF1  C (stacker, nested) {f1_score(y, st.argmax(1), average='macro'):.5f}")
lr = LogisticRegression(C=0.1, max_iter=2000).fit(X, y); pc = lr.predict_proba(Xt)
imp = np.abs(lr.coef_).reshape(K, len(O), K).sum((0, 2)); imp /= imp.sum()
print("stacker weight share:", {t: round(float(v), 3) for t, v in zip(O, imp)})
print("test rows C vs A:", int((pa.argmax(1) != pc.argmax(1)).sum()), "| C vs B:", int((pb.argmax(1) != pc.argmax(1)).sum()))
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")
for name, P in (("calib", pb), ("stack", pc)):
    od = f"t4/research/blend_clean_{name}{suf}"; os.makedirs(od, exist_ok=True); np.save(f"{od}/t4_blend_test_proba.npy", P.astype(np.float32))
    pd.DataFrame({"id": ref["id"], "id_img": ref["id_img"], "pred_num": P.argmax(1), "pred_str": np.array(C)[P.argmax(1)]}).to_csv(f"{od}/t4_blend.csv", index=False, lineterminator="\n")
    print(name, "share", dict(zip(C, (np.bincount(P.argmax(1), minlength=K) / len(P)).round(3))), "->", od)

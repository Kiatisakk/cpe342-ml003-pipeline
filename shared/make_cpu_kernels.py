# Build self-contained Kaggle CPU kernels that embed repo scripts (base64) and run them on the competition data.
# usage (repo root): python shared/make_cpu_kernels.py   -> writes t1/kaggle/<slug>/ and t3/kaggle/<slug>/, then kaggle kernels push -p <dir>
# No test rows are used for training (Rule 2): the embedded scripts are the same CV recipes as local.
import os, json, base64

T1 = ["t1/research/common.py", "t1/research/final_v2.py", "t1/research/make_test_files.py"]
T3 = ["t3/research/common.py", "t3/research/exp_lgb_hurdle.py", "t3/research/exp_cbq.py", "t3/research/avg_q.py"]

HEAD = r'''
import os, glob, base64, subprocess, sys, time, shutil
W = "/kaggle/working/repo"; T0 = time.time()
for p, b in FILES.items():
    f = os.path.join(W, p); os.makedirs(os.path.dirname(f), exist_ok=True); open(f, "wb").write(base64.b64decode(b))
tr = glob.glob("/kaggle/input/**/01_binary_classification/train.csv", recursive=True)
assert tr, "competition data not found"
os.makedirs(W + "/t3/research/oof", exist_ok=True)   # exp_lgb_hurdle saves p only if oof/ already exists
root = os.path.dirname(os.path.dirname(tr[0])); print("data root", root, os.listdir(root))
LINKS = ["/kaggle/working/cpe342-kaggle-main", W + "/cpe342-kaggle-main"]   # t1 common: ROOT/../../../ ; t3 common: cwd-relative
for l in LINKS:
    if not os.path.exists(l): os.symlink(root, l)
for p, s in PATCH:
    f = os.path.join(W, p); t = open(f).read(); assert s[0] in t, (p, s[0]); open(f, "w").write(t.replace(s[0], s[1]))
def run(*cmd, env=None):
    e = dict(os.environ, **(env or {})); print(">>", " ".join(cmd), env or "", f"[{(time.time()-T0)/3600:.2f} h]", flush=True)
    subprocess.run([sys.executable, *cmd], cwd=W, env=e, check=True)
'''
TAIL = r'''
for l in LINKS:
    if os.path.islink(l): os.unlink(l)
for c in glob.glob(W + "/**/cache", recursive=True): shutil.rmtree(c, ignore_errors=True)
print("DONE", f"{(time.time()-T0)/3600:.2f} h")
'''


def build(slug, task, files, patch, body):
    d = f"{task}/kaggle/{slug}"; os.makedirs(d, exist_ok=True)
    enc = {p: base64.b64encode(open(p, "rb").read()).decode() for p in files}
    code = f"FILES = {json.dumps(enc)}\nPATCH = {json.dumps(patch)}\n" + HEAD + body + TAIL
    fn = slug.replace("-", "_") + ".py"; open(os.path.join(d, fn), "w", newline="\n").write(code)
    meta = {"id": f"kiatisakkk/{slug}", "title": slug, "code_file": fn, "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "dataset_sources": [],
            "competition_sources": ["player-intelligence-system-cpe-342-ml"], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1); print("wrote", d)


# T1: final_v2 recipe exactly (lr .02, 3 bags, NC 4), new fold/model seeds -> OOF per seed + test
for slug, seeds in [("task1-finalv2-seeds-a", "101,102,103,104,105"), ("task1-finalv2-seeds-b", "106,107,108,109,110")]:
    patch = [["t1/research/common.py", ["SEEDS = (42, 7, 2024)",
              "SEEDS = tuple(int(s) for s in os.environ.get('SEEDS', '42,7,2024').split(','))"]]]
    build(slug, "t1", T1, patch, f'run("t1/research/final_v2.py", env={{"SEEDS": "{seeds}", "PREFIX": "s{seeds[:3]}_"}})\n')

R = "t3/research/"
# T3 #1: run_all.sh recipe with 10 seeds of the classifier p and of CatBoost d4 MultiQuantile
b = f'run("{R}exp_lgb_hurdle.py", "lgbh_log")\n'
b += "".join(f'run("{R}exp_lgb_hurdle.py", "clf_s{s}", "--clf_only", "--seed", "{s}")\n' for s in range(1, 10))
b += f'run("{R}exp_cbq.py", "cbq_d4", "--depth", "4", "--lr", "0.05")\n'
b += "".join(f'run("{R}exp_cbq.py", "cbq_d4s{s}", "--depth", "4", "--lr", "0.05", "--seed", "{s}")\n' for s in range(1, 10))
P3, P10 = "lgbh_log,clf_s1,clf_s2", ",".join(["lgbh_log"] + [f"clf_s{s}" for s in range(1, 10)])
Q3, Q10 = "cbq_d4,cbq_d4s1,cbq_d4s2", ",".join(["cbq_d4"] + [f"cbq_d4s{s}" for s in range(1, 10)])
for out, q, p in [("bag_q3p3", Q3, P3), ("bag_q10p3", Q10, P3), ("bag_q3p10", Q3, P10), ("bag_q10p10", Q10, P10)]:
    b += f'run("{R}avg_q.py", "{out}", "--q", "{q}", "--p", "{p}")\n'
build("task3-seedbag10", "t3", T3, [], b)

# T3 #2: CatBoost depth 3 with the Optuna R6 params (opt_t20), 5 seeds, + d4 x3 for the mixed bag
b = f'run("{R}exp_lgb_hurdle.py", "lgbh_log")\n'
b += "".join(f'run("{R}exp_lgb_hurdle.py", "clf_s{s}", "--clf_only", "--seed", "{s}")\n' for s in (1, 2))
D3 = ["--depth", "3", "--lr", "0.05", "--l2", "5.683", "--rs", "0.0514", "--bt", "0.273", "--rsm", "0.745", "--border", "128", "--p_avg"]
b += "".join(f'run("{R}exp_cbq.py", "cbd3o_s{s}", *{D3}, "--seed", "{s}")\n' for s in range(5))
b += f'run("{R}exp_cbq.py", "cbq_d4", "--depth", "4", "--lr", "0.05")\n'
b += "".join(f'run("{R}exp_cbq.py", "cbq_d4s{s}", "--depth", "4", "--lr", "0.05", "--seed", "{s}")\n' for s in (1, 2))
QD3 = ",".join(f"cbd3o_s{s}" for s in range(5))
for out, q in [("bag_d4x3", Q3), ("bag_d3x5", QD3), ("bag_d3x5_d4x3", QD3 + "," + Q3)]:
    b += f'run("{R}avg_q.py", "{out}", "--q", "{q}", "--p", "{P3}")\n'
build("task3-cbd3-bag", "t3", T3, [], b)

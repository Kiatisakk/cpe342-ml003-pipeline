# Helper: build a notebook from (kind, text) cells and execute it from the repo root so outputs/plots are stored.
import os, sys, nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(name, cells, execute=True, timeout=3600):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    path = os.path.join(ROOT, "report_notebooks", name)
    if execute:
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
    nbformat.write(nb, path)
    print("wrote", path)


SETUP = r'''
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
# notebook runs from the repo root (Kaggle-ML/)
if os.path.basename(os.getcwd()) == "report_notebooks": os.chdir("..")
print("cwd:", os.getcwd())
'''


def md(s): return ("md", s)
def code(s): return ("code", s)

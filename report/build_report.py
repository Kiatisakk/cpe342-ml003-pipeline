"""Automated XeLaTeX builder, log/numerical auditor, and PDF page PNG renderer for report/."""
import argparse
import json
import os
import re
import subprocess
import sys
import pymupdf

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(REPORT_DIR)
MAIN_TEX = os.path.join(REPORT_DIR, "main.tex")
MAIN_PDF = os.path.join(REPORT_DIR, "main.pdf")
MAIN_LOG = os.path.join(REPORT_DIR, "main.log")
QA_DIR = os.path.join(REPORT_DIR, "qa")
LEDGER_PATH = os.path.join(REPORT_DIR, "facts_ledger.json")


def compile_xelatex(passes=2):
    cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    for i in range(passes):
        print(f"[build] Running xelatex pass {i + 1}/{passes}...")
        proc = subprocess.run(cmd, cwd=REPORT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            print(proc.stderr[-1000:])
            raise RuntimeError(f"xelatex failed on pass {i + 1} with code {proc.returncode}")
    print(f"[build] Compiled -> {MAIN_PDF}")


def audit_log():
    with open(MAIN_LOG, encoding="utf-8", errors="replace") as f:
        log = f.read()
    errors = []
    for pat, label in [
        (r"^! ", "LaTeX Error"),
        (r"Undefined control sequence", "Undefined control sequence"),
        (r"There were undefined references", "Undefined references"),
        (r"Missing character:", "Missing font character"),
        (r"File `[^']+' not found", "Missing file/figure"),
    ]:
        hits = re.findall(pat, log, flags=re.MULTILINE)
        if hits:
            errors.append(f"{label}: {len(hits)} occurrences")

    overfull = []
    for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log):
        pts = float(m.group(1))
        if pts > 12.0:
            overfull.append(pts)

    print(f"[audit-log] Critical issues: {len(errors)} | Overfull >12pt: {len(overfull)}")
    for e in errors:
        print(f"  [ERROR] {e}")
    if overfull:
        print(f"  [ERROR] Max overfull hbox: {max(overfull):.1f}pt (count={len(overfull)})")
    return len(errors) == 0 and len(overfull) == 0


def audit_numbers_and_scope():
    if not os.path.exists(LEDGER_PATH):
        print("[audit-num] Skipping: facts_ledger.json missing")
        return True
    ledger = json.load(open(LEDGER_PATH, encoding="utf-8"))
    sec_dir = os.path.join(REPORT_DIR, "sections")

    # Check ALL section files for forbidden Task 4 tokens
    forbidden_t4 = ["0.87346", "0.87336", "0.9915", "0.99467", "LoRA", "DINOv3", "ConvNeXt", "EVA02"]
    for fname in sorted(os.listdir(sec_dir)):
        if not fname.endswith(".tex"):
            continue
        with open(os.path.join(sec_dir, fname), encoding="utf-8") as f:
            txt = f.read()
        for tok in forbidden_t4:
            if tok in txt:
                print(f"[ERROR] Forbidden Task 4 token '{tok}' found in {fname}")
                return False
    print("[audit-scope] All sections clear of Task 4 tokens.")

    # Required verified strings in each task section
    checks = {
        "sec_01_task1.tex": ["0.89499", "0.84007", "15,359", "59.33", "0.3769", "0.67890"],
        "sec_02_task2.tex": ["0.75826", "0.75766", "0.5975", "10,576", "3,715", "0.9677"],
        "sec_03_task3.tex": ["0.79537", "0.79191", "0.79391", "0.79505", "50.39", "10,429.55"],
        "sec_05_task5.tex": ["2,104", "827", "981", "244", "52", "0.9968"],
    }
    ok = True
    for fname, tokens in checks.items():
        fpath = os.path.join(sec_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            txt = f.read()
        missing = [t for t in tokens if t not in txt]
        if missing:
            print(f"[audit-num] {fname} missing expected ledger tokens: {missing}")
            ok = False
        else:
            print(f"[audit-num] {fname}: all {len(tokens)} verified anchor numbers present")
    return ok


def render_qa_pngs(dpi=140):
    os.makedirs(QA_DIR, exist_ok=True)
    for old in os.listdir(QA_DIR):
        if old.startswith("page_") and old.endswith(".png"):
            os.remove(os.path.join(QA_DIR, old))
    doc = pymupdf.open(MAIN_PDF)
    print(f"[render] Rendering {len(doc)} pages to {QA_DIR}...")
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        pix.save(os.path.join(QA_DIR, f"page_{i + 1:02d}.png"))
    print(f"[render] Rendered {len(doc)} PNG pages.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Compile smoke check without numerical audit")
    ap.add_argument("--audit", action="store_true", help="Run log + facts_ledger numerical audit")
    ap.add_argument("--render-qa", action="store_true", help="Render PDF pages to report/qa/page_XX.png")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    compile_xelatex(passes=args.passes)
    log_ok = audit_log()
    num_ok = True
    if args.audit and not args.smoke:
        num_ok = audit_numbers_and_scope()
    if args.render_qa:
        render_qa_pngs()
    if not (log_ok and num_ok):
        sys.exit(1)
    print("[done] All checks passed.")


if __name__ == "__main__":
    main()

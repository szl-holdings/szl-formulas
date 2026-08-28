#!/usr/bin/env python3
"""Re-verify the szl-formulas surrogate: sha256 the shipped model against
TRAINING_RECEIPT.json, then deterministically regenerate the seeded dataset via
scripts/forge.py (which resolves the kernel from this repo's own build/ dir) and
compare re-measured 21-class accuracy to the receipt (tolerance 0.02 across lib
versions). Run from repo root: python scripts/eval.py"""
import hashlib, json, subprocess, sys, tempfile, os, shutil
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
receipt = json.load(open(f"{root}/TRAINING_RECEIPT.json"))
got = hashlib.sha256(open(f"{root}/model.joblib", "rb").read()).hexdigest()
want = receipt["model"]["sha256"]
print(f"model.joblib sha256 {'MATCHES receipt' if got==want else 'MISMATCH — refuse'}: {got[:16]}…")
if got != want: sys.exit(1)
with tempfile.TemporaryDirectory() as td:
    copy = f"{td}/repo"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns(".cache"))
    out = subprocess.run([sys.executable, f"{copy}/scripts/forge.py"],
                         capture_output=True, text=True, cwd=copy)
    print(out.stdout[-500:] if out.returncode == 0 else out.stderr[-500:])
    if out.returncode: sys.exit(1)
    re_receipt = json.load(open(f"{copy}/scripts/TRAINING_RECEIPT.json"))
    d = abs(re_receipt["metrics_MEASURED"]["test_accuracy_21_class"]
            - receipt["metrics_MEASURED"]["test_accuracy_21_class"])
    print(f"re-measured 21-class accuracy delta vs receipt: {d:.4f} ({'OK <=0.02' if d<=0.02 else 'FAIL'})")
    sys.exit(0 if d <= 0.02 else 1)

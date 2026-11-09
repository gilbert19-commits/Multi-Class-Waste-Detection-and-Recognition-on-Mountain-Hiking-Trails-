"""GPU part of the evaluation for runs added after the original 13 (YOLOv8l/x, YOLOv12s e200).

Reuses 01_evaluate_test.evaluate() for metrics/params/GFLOPs/ms per image and
02_error_analysis.predict() for the cached validator predictions, so every number is
produced exactly as for the original 13 models. Nothing in results/ is overwritten:
  - metrics  -> results/test_metrics_{overall,per_class}_<suffix>.csv (old rows + new rows)
  - preds    -> results/_preds/<run>_{full,clean}.pkl (new files)
  - val runs -> results/_val_runs/<run>_{full,clean}[_err] (new folders)
The existing test subsets (results/test_sets) are used as they are, not rebuilt.

Usage:
    python scripts/05_eval_tambahan.py <suffix> <run> [<run> ...]
    python scripts/05_eval_tambahan.py 15model v8l v8x
"""
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval01 = load("eval01", "01_evaluate_test.py")
eval02 = load("eval02", "02_error_analysis.py")


def main(suffix, run_names):
    t0 = time.time()
    runs, _ = eval01.inventory()
    by_name = {r["run"]: r for r in runs}
    missing = [r for r in run_names if r not in by_name]
    if missing:
        raise SystemExit(f"run tidak ditemukan di inventaris: {missing}")
    sel = [by_name[r] for r in run_names]

    yamls = {s: OUT / "test_sets" / f"test_{s}.yaml" for s in ("full", "clean")}
    names = yaml.safe_load((ROOT / "TA-1" / "data.yaml").read_text())["names"]
    sets = {s: eval02.load_test_set(s) for s in ("full", "clean")}

    overall, per_class = [], []
    for i, rec in enumerate(sel, 1):
        print(f"[{i}/{len(sel)}] {rec['run']} ({rec['model']}{rec['varian']}, aug={rec['augmentasi']}, "
              f"epochs={rec['epochs']}) {rec['best_pt'].relative_to(ROOT)}", flush=True)
        o, p = eval01.evaluate(rec, yamls, names)
        for r in o + p:
            r["run"] = rec["run"]
        overall += o
        per_class += p
        for s, (imgs, gts, _) in sets.items():
            preds, gts_val = eval02.predict(rec, s, imgs)
            n_own, n_val = sum(len(g) for g in gts), sum(len(g) for g in gts_val)
            if n_own != n_val:
                raise RuntimeError(f"jumlah GT berbeda: label {n_own} vs validator {n_val}")
            print(f"      prediksi {s}: {len(preds)} citra -> results/_preds/{rec['run']}_{s}.pkl", flush=True)

    old_o = pd.read_csv(OUT / "test_metrics_overall.csv")
    old_p = pd.read_csv(OUT / "test_metrics_per_class.csv")
    ov = pd.concat([old_o, pd.DataFrame(overall)], ignore_index=True)
    pc = pd.concat([old_p, pd.DataFrame(per_class)], ignore_index=True)
    ov.to_csv(OUT / f"test_metrics_overall_{suffix}.csv", index=False)
    pc.to_csv(OUT / f"test_metrics_per_class_{suffix}.csv", index=False)
    print(f"\nSelesai dalam {(time.time() - t0) / 60:.1f} menit. "
          f"Output: results/test_metrics_overall_{suffix}.csv, results/test_metrics_per_class_{suffix}.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

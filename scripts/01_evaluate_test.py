"""Evaluate every trained run (YOLOv12 non-aug / aug, YOLOv8) on two test sets.

  full  : TA-1/test (550 images)
  clean : TA-1/test minus images whose D0 scene group also appears in train/valid

Read-only w.r.t. weights and dataset. Test subsets are built as hardlinks under
results/test_sets/ so Ultralytics writes its labels.cache there, never into TA-1/.

Usage:
    python scripts/01_evaluate_test.py
"""
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SETS_DIR = OUT / "test_sets"
GROUPS_CSV = ROOT / "analysis" / "tables" / "D0_scene_groups.csv"

SEED = 0
EVAL_ARGS = dict(split="test", imgsz=640, conf=0.001, iou=0.6, batch=16, device=0,
                 plots=False, save_json=False, verbose=False, deterministic=True, seed=SEED)
MIN_EPOCHS = 10  # runs trained for fewer epochs are smoke tests, not results
VARIANT_RE = re.compile(r"v(\d+)([nsmlx])(?![a-z0-9])", re.I)


# ---------------------------------------------------------------- inventory
def find_data_yaml(args_data):
    """Resolve the data.yaml a run was trained on (args.yaml may hold a relative path)."""
    p = Path(str(args_data))
    for cand in (p, ROOT / p):
        if cand.exists():
            return cand.resolve()
    return None


def inventory():
    runs, excluded = [], []
    for best in sorted(ROOT.glob("runs/**/weights/best.pt")):
        run_dir = best.parent.parent
        args_f = run_dir / "args.yaml"
        args = yaml.safe_load(args_f.read_text()) if args_f.exists() else {}
        m = VARIANT_RE.search(str(args.get("model", ""))) or VARIANT_RE.search(run_dir.name)
        data = find_data_yaml(args.get("data", ""))
        rec = dict(
            run=run_dir.name,
            model=f"YOLOv{m.group(1)}" if m else "?",
            varian=m.group(2).lower() if m else "?",
            augmentasi="ya" if data and data.parent.name.lower() == "augmented" else "tidak",
            epochs=args.get("epochs"),
            data=str(data.relative_to(ROOT)) if data else str(args.get("data")),
            best_pt=best,
        )
        if not m:
            excluded.append((rec, "varian tidak terbaca dari args.yaml / nama run"))
        elif isinstance(rec["epochs"], int) and rec["epochs"] < MIN_EPOCHS:
            excluded.append((rec, f"smoke test ({rec['epochs']} epoch)"))
        else:
            runs.append(rec)
    order = {k: i for i, k in enumerate("nsmlx")}
    runs.sort(key=lambda r: (r["model"] != "YOLOv12", r["augmentasi"], order.get(r["varian"], 9)))
    return runs, excluded


# ---------------------------------------------------------------- test sets
def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_test_sets(base_yaml):
    base = yaml.safe_load(base_yaml.read_text())
    test_dir = (base_yaml.parent / base.get("test", "test/images")).resolve()
    lbl_dir = test_dir.parent / "labels"
    all_imgs = sorted(p.name for p in test_dir.iterdir() if p.is_file())

    g = pd.read_csv(GROUPS_CSV)
    splits_per_group = g.groupby("group_id")["split"].agg(set)
    cross = {k for k, v in splits_per_group.items() if "test" in v and len(v) > 1}
    t = g[g["split"] == "test"]
    leaked = set(t.loc[t["group_id"].isin(cross), "filename"])
    missing = set(t["filename"]) - set(all_imgs)
    if missing:
        raise RuntimeError(f"{len(missing)} citra di D0_scene_groups.csv tidak ada di {test_dir}")

    subsets = {"full": all_imgs, "clean": [f for f in all_imgs if f not in leaked]}
    yamls = {}
    for name, files in subsets.items():
        d = SETS_DIR / name
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        for f in files:
            link(test_dir / f, d / "images" / f)
            lbl = lbl_dir / (Path(f).stem + ".txt")
            if lbl.exists():
                link(lbl, d / "labels" / lbl.name)
        y = SETS_DIR / f"test_{name}.yaml"
        y.write_text(yaml.safe_dump(dict(path=str(d), train="images", val="images", test="images",
                                         nc=base["nc"], names=base["names"]), sort_keys=False))
        yamls[name] = y
    return yamls, {k: len(v) for k, v in subsets.items()}, len(leaked), base["names"]


# ---------------------------------------------------------------- evaluation
def n_valid_images(y):
    """Images Ultralytics actually evaluated (it drops images whose label file it rejects)."""
    from ultralytics.data.utils import load_dataset_cache_file

    cache = load_dataset_cache_file(Path(yaml.safe_load(Path(y).read_text())["path"]) / "labels.cache")
    return len(cache["labels"])


def rejected_images(y):
    from ultralytics.data.utils import load_dataset_cache_file

    cache = load_dataset_cache_file(Path(yaml.safe_load(Path(y).read_text())["path"]) / "labels.cache")
    return [m for m in cache.get("msgs", []) if "ignoring corrupt" in m]


def evaluate(rec, yamls, names):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params, init_seeds

    model = YOLO(str(rec["best_pt"]))
    model_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if model_names != list(names):
        raise RuntimeError(f"nama kelas model {model_names} != data.yaml {names}")
    params = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=EVAL_ARGS["imgsz"])

    overall, per_class = [], []
    for set_name, y in yamls.items():
        init_seeds(SEED, deterministic=True)
        m = model.val(data=str(y), project=str(OUT / "_val_runs"),
                      name=f"{rec['run']}_{set_name}", exist_ok=True, **EVAL_ARGS)
        box = m.box
        present = list(m.ap_class_index)
        row = dict(model=rec["model"], varian=rec["varian"], augmentasi=rec["augmentasi"],
                   test_set=set_name, precision=box.mp, recall=box.mr, mAP50=box.map50,
                   mAP50_95=box.map, params=params, GFLOPs=gflops,
                   ms_per_img=m.speed["inference"],
                   ms_per_img_total=sum(m.speed[k] for k in ("preprocess", "inference", "postprocess")),
                   n_images=n_valid_images(y), n_kelas_dievaluasi=len(present))
        cls_rows = {}
        for c, cname in enumerate(names):
            r = dict(model=rec["model"], varian=rec["varian"], augmentasi=rec["augmentasi"],
                     test_set=set_name, kelas=cname, images=int(m.nt_per_image[c]),
                     instances=int(m.nt_per_class[c]),
                     precision=np.nan, recall=np.nan, mAP50=np.nan, mAP50_95=np.nan)
            if c in present:
                p, rr, ap50, ap = box.class_result(present.index(c))
                r.update(precision=p, recall=rr, mAP50=ap50, mAP50_95=ap)
            cls_rows[cname] = r
            per_class.append(r)
        # 3-class mean (without puntung_rokok) so full and clean can be compared like-for-like
        three = [cls_rows[n] for n in names if n != "puntung_rokok" and cls_rows[n]["instances"] > 0]
        row["mAP50_3kelas"] = float(np.mean([r["mAP50"] for r in three]))
        row["mAP50_95_3kelas"] = float(np.mean([r["mAP50_95"] for r in three]))
        overall.append(row)
        print(f"      {set_name:5s}: {row['n_images']} citra, {len(present)} kelas | "
              f"P {row['precision']:.3f} R {row['recall']:.3f} "
              f"mAP50 {row['mAP50']:.3f} mAP50-95 {row['mAP50_95']:.3f} | "
              f"{row['ms_per_img']:.2f} ms/img", flush=True)
    return overall, per_class


# ---------------------------------------------------------------- reports
def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def signed(x):
    return "–" if pd.isna(x) else f"{x:+.3f}"


def label(r):
    return f"{r['model']}{r['varian']}" + (" aug" if r["augmentasi"] == "ya" else "")


def write_rekap(ov):
    full = ov[ov.test_set == "full"].set_index(["model", "varian", "augmentasi"])
    clean = ov[ov.test_set == "clean"].set_index(["model", "varian", "augmentasi"])
    L = ["# Rekap Evaluasi Test Set", "",
         "Pengaturan identik untuk semua model: imgsz=640, conf=0.001, iou=0.6, batch=16, "
         "device=0, seed=0, deterministic=True.", "",
         f"- **Full**: TA-1/test, {int(full['n_images'].iloc[0])} citra dievaluasi, 4 kelas "
         "(citra yang labelnya ditolak Ultralytics tidak dihitung, lihat ringkasan.md).",
         f"- **Clean**: {int(clean['n_images'].iloc[0])} citra, tanpa scene group yang juga ada di "
         "train/val (D0). puntung_rokok tidak punya instance, jadi metrik clean = rata-rata 3 kelas.",
         "- **Δ** = clean − full (3 kelas). Pembanding full memakai 3 kelas yang sama supaya setara.", "",
         "## Akurasi", "",
         "| Model | Varian | Augmentasi | P full | R full | mAP@50 full (4 kls) | mAP@50-95 full (4 kls) "
         "| mAP@50 full (3 kls) | mAP@50-95 full (3 kls) | P clean | R clean | mAP@50 clean (3 kls) "
         "| mAP@50-95 clean (3 kls) | Δ mAP@50 | Δ mAP@50-95 |",
         "|" + "---|" * 15]
    for k in full.index:
        a, b = full.loc[k], clean.loc[k] if k in clean.index else None
        cells = [k[0], k[1], k[2], f3(a.precision), f3(a.recall), f3(a.mAP50), f3(a.mAP50_95),
                 f3(a.mAP50_3kelas), f3(a.mAP50_95_3kelas)]
        if b is not None:
            cells += [f3(b.precision), f3(b.recall), f3(b.mAP50), f3(b.mAP50_95),
                      signed(b.mAP50 - a.mAP50_3kelas), signed(b.mAP50_95 - a.mAP50_95_3kelas)]
        else:
            cells += ["–"] * 6
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Efisiensi", "",
          "ms/img = waktu inference per citra (batch 16, GPU), diukur pada test full.", "",
          "| Model | Varian | Augmentasi | Params | GFLOPs | ms/img |", "|---|---|---|---|---|---|"]
    for k in full.index:
        a = full.loc[k]
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {a.params / 1e6:.2f} M | {a.GFLOPs:.1f} | {a.ms_per_img:.2f} |")
    (OUT / "tabel_rekap.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_ringkasan(ov, pc, n_sets, n_leaked, names, skipped, yamls):
    L = ["# Ringkasan Evaluasi Test Set", ""]
    n_clean = n_sets["clean"]
    L += ["## Ukuran test set", "",
          f"- Full: {n_sets['full']} citra, **{n_valid_images(yamls['full'])} dievaluasi**.",
          f"- Clean: **{n_clean} citra** ({n_sets['full']} − {n_leaked} citra di scene group lintas split), "
          f"{n_valid_images(yamls['clean'])} dievaluasi. "
          + ("Sesuai target 192." if n_clean == 192 else f"**Tidak sama dengan target 192** (selisih {n_clean - 192})."),
          ""]
    for name, y in yamls.items():
        rej = rejected_images(y)
        if rej:
            L += [f"Citra yang ditolak Ultralytics di test {name} (tidak ikut dihitung, untuk semua model sama):", ""]
            for m in rej:
                hit = re.search(r"([^\\/\s]+\.(?:jpe?g|png|bmp|webp)): ignoring corrupt image/label: (.*)", m, re.I)
                L.append(f"- `{hit.group(1)}`: {hit.group(2)}" if hit else f"- {m}")
            L.append("")

    clean_inst = pc[pc.test_set == "clean"].groupby("kelas")["instances"].first()
    zero = [c for c in names if clean_inst.get(c, 0) == 0]
    L += ["## Catatan: jumlah kelas berbeda", "",
          f"Di test clean, kelas {', '.join(zero) if zero else '(tidak ada)'} tidak punya instance, "
          f"sehingga mAP clean dihitung dari {len(names) - len(zero)} kelas, sedangkan mAP full dari "
          f"{len(names)} kelas. Kedua angka itu **tidak setara** dan tidak boleh dibandingkan langsung. "
          "Untuk perbandingan full vs clean dipakai mAP full versi 3 kelas (tanpa puntung_rokok).", ""]

    clean = ov[ov.test_set == "clean"]
    v12 = clean[clean.model == "YOLOv12"]
    v8 = clean[clean.model == "YOLOv8"]
    if len(v12):
        best = v12.loc[v12.mAP50_95.idxmax()]
        full_best = ov[(ov.test_set == "full") & (ov.model == best.model) & (ov.varian == best.varian)
                       & (ov.augmentasi == best.augmentasi)].iloc[0]
        L += ["## Varian YOLOv12 terbaik (mAP@50-95, test clean)", "",
              f"**{label(best)}**: mAP@50-95 clean {best.mAP50_95:.3f} (3 kelas), "
              f"mAP@50 clean {best.mAP50:.3f}. Di test full: mAP@50-95 {full_best.mAP50_95:.3f} (4 kelas) / "
              f"{full_best.mAP50_95_3kelas:.3f} (3 kelas).", "",
              "Peringkat YOLOv12 di test clean:", "",
              "| Peringkat | Model | mAP@50-95 clean | mAP@50 clean |", "|---|---|---|---|"]
        for i, (_, r) in enumerate(v12.sort_values("mAP50_95", ascending=False).iterrows(), 1):
            L.append(f"| {i} | {label(r)} | {r.mAP50_95:.3f} | {r.mAP50:.3f} |")
        L.append("")

        if len(v8):
            same = v8[v8.varian == best.varian]
            v8_best = v8.loc[v8.mAP50_95.idxmax()]
            comps = [("YOLOv8 terbaik", v8_best)]
            if len(same) and same.iloc[0].varian != v8_best.varian:
                comps.append(("YOLOv8 skala sama", same.iloc[0]))
            elif not len(same):
                L.append(f"YOLOv8{best.varian} tidak tersedia, jadi pembanding skala sama tidak ada.\n")
            for title, ref in comps:
                L += [f"## {label(best)} vs {label(ref)} ({title}), per kelas", ""]
                for ts in ("full", "clean"):
                    L += [f"**Test {ts}**", "",
                          "| Kelas | Instances | mAP@50-95 " + label(best) + " | mAP@50-95 " + label(ref)
                          + " | Δ | mAP@50 " + label(best) + " | mAP@50 " + label(ref) + " | Δ |",
                          "|---|---|---|---|---|---|---|---|"]
                    for c in names:
                        def get(r):
                            return pc[(pc.test_set == ts) & (pc.kelas == c) & (pc.model == r.model)
                                      & (pc.varian == r.varian) & (pc.augmentasi == r.augmentasi)].iloc[0]
                        a, b = get(best), get(ref)
                        L.append(f"| {c} | {a.instances} | {f3(a.mAP50_95)} | {f3(b.mAP50_95)} | "
                                 f"{signed(a.mAP50_95 - b.mAP50_95)} | {f3(a.mAP50)} | {f3(b.mAP50)} | "
                                 f"{signed(a.mAP50 - b.mAP50)} |")
                    L.append("")

    f = ov[ov.test_set == "full"].set_index(["model", "varian", "augmentasi"])
    c = ov[ov.test_set == "clean"].set_index(["model", "varian", "augmentasi"])
    idx = f.index.intersection(c.index)
    d50 = (c.loc[idx, "mAP50"] - f.loc[idx, "mAP50_3kelas"])
    d95 = (c.loc[idx, "mAP50_95"] - f.loc[idx, "mAP50_95_3kelas"])
    d95_raw = (c.loc[idx, "mAP50_95"] - f.loc[idx, "mAP50_95"])
    L += ["## Dampak overlap antar split (clean − full)", "",
          f"Rata-rata dari {len(idx)} model, pembanding full 3 kelas (setara):", "",
          f"- Δ mAP@50: **{d50.mean():+.3f}** (min {d50.min():+.3f}, maks {d50.max():+.3f})",
          f"- Δ mAP@50-95: **{d95.mean():+.3f}** (min {d95.min():+.3f}, maks {d95.max():+.3f})", "",
          f"Sebagai referensi saja (tidak setara, 3 vs 4 kelas): Δ mAP@50-95 rata-rata {d95_raw.mean():+.3f}.", "",
          "Nilai negatif berarti model lebih buruk pada citra yang adegannya tidak pernah terlihat saat "
          "training. Besarnya selisih itu adalah ukuran seberapa angka test full terdongkrak oleh overlap.", "",
          "Catatan: subset clean juga berbeda komposisi sumbernya dari full, jadi Δ mencampur efek "
          "leakage dan efek perbedaan tingkat kesulitan citra.", ""]
    if skipped:
        L += ["## Run yang dilewati", ""] + [f"- {r}: {why}" for r, why in skipped] + [""]
    (OUT / "ringkasan.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    runs, excluded = inventory()

    print("=" * 100)
    print("INVENTARIS RUN")
    print("=" * 100)
    print(f"{'run':22s} {'model':8s} {'varian':6s} {'augmentasi':10s} {'epochs':6s} {'data':22s} best.pt")
    for r in runs:
        print(f"{r['run']:22s} {r['model']:8s} {r['varian']:6s} {r['augmentasi']:10s} "
              f"{str(r['epochs']):6s} {r['data']:22s} {r['best_pt'].relative_to(ROOT)}")
    for r, why in excluded:
        print(f"[DIKECUALIKAN] {r['run']}: {why}  ({r['best_pt'].relative_to(ROOT)})")

    data_yamls = sorted(p.relative_to(ROOT) for p in ROOT.glob("*/data.yaml"))
    print(f"\ndata.yaml ditemukan: {', '.join(map(str, data_yamls))}")
    base_yaml = ROOT / "TA-1" / "data.yaml"
    yamls, n_sets, n_leaked, names = build_test_sets(base_yaml)
    print(f"Test set: full {n_sets['full']} citra | clean {n_sets['clean']} citra "
          f"({n_leaked} dibuang karena scene group lintas split)")
    print(f"Subset dibuat sebagai hardlink di {SETS_DIR.relative_to(ROOT)} (file asli tidak diubah)\n")

    overall, per_class, skipped = [], [], [(r["run"], why) for r, why in excluded]
    t0 = time.time()
    for i, rec in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {rec['run']} ({rec['model']}{rec['varian']}, aug={rec['augmentasi']})", flush=True)
        try:
            o, p = evaluate(rec, yamls, names)
            overall += o
            per_class += p
            print(f"      selesai ({time.time() - t0:.0f} s sejak mulai)", flush=True)
        except Exception as e:  # missing / corrupt weights must not stop the whole run
            msg = f"{type(e).__name__}: {e}".splitlines()[0][:200]
            print(f"      [SKIP] {msg}", flush=True)
            skipped.append((rec["run"], msg))

    if not overall:
        print("Tidak ada model yang berhasil dievaluasi.")
        sys.exit(1)

    ov = pd.DataFrame(overall)
    pc = pd.DataFrame(per_class)
    ov.to_csv(OUT / "test_metrics_overall.csv", index=False)
    pc.to_csv(OUT / "test_metrics_per_class.csv", index=False)
    write_rekap(ov)
    write_ringkasan(ov, pc, n_sets, n_leaked, names, skipped, yamls)

    print("\n" + "=" * 100)
    print(f"Selesai dalam {(time.time() - t0) / 60:.1f} menit. Output di {OUT.relative_to(ROOT)}/")
    if skipped:
        print("Run yang dilewati / dikecualikan:")
        for r, why in skipped:
            print(f"  - {r}: {why}")
    else:
        print("Semua run berhasil dievaluasi.")


if __name__ == "__main__":
    main()

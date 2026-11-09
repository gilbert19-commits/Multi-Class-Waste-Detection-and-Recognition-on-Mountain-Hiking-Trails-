"""CPU-only analysis for YOLOv8l/x added to the original 13 models (15 models in total).

Needs results/test_metrics_*_15model.csv and results/_preds/v8{l,x}_*.pkl from
05_eval_tambahan.py. No GPU, no training, nothing old is overwritten.

Uses the functions of 02_error_analysis.py (matching, confusion matrix, AP, bootstrap weights)
and 04_label_noise_check.py (kertas<->plastik events, source mapping), so the numbers are
computed exactly like the original 13-model analysis.

Outputs (results/):
  confusion_matrix/YOLOv8{l,x}_aug_{clean,full}.png, confusion_matrix_v8lx_15model.csv
  fp_fn_per_class_15model.csv, bootstrap_head_to_head_15model.csv,
  label_noise_per_sumber_15model.csv, label_noise_agregat_15model.csv,
  selisih_full_clean_15model.csv, peringkat_augmented_15model.csv, training_v8lx_15model.csv

Usage:
    python scripts/06_analisis_15model.py
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SUFFIX = "15model"
NEW_RUNS = ["v8l", "v8x"]
PAIRS = [(("YOLOv12", "l", "ya"), ("YOLOv8", "l", "ya")),
         (("YOLOv12", "x", "ya"), ("YOLOv8", "x", "ya"))]

spec = importlib.util.spec_from_file_location("lbl04", ROOT / "scripts" / "04_label_noise_check.py")
lbl04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lbl04)
eval02 = lbl04.eval02
eval01 = lbl04.eval01
SEED, N_BOOT, CM_CONF, CM_IOU = eval02.SEED, eval02.N_BOOT, eval02.CM_CONF, eval02.CM_IOU
SOURCES = ["UAVVaste", "Burangrang", "TACO"]


def key_of(rec):
    return rec["model"], rec["varian"], rec["augmentasi"]


def md(df, floatfmt=repr):
    """Markdown table; floats printed at full precision."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return out


# ---------------------------------------------------------------- training info
def training_info(run):
    d = ROOT / "runs" / "detect" / "runs_baseline" / run
    res = pd.read_csv(d / "results.csv")
    res.columns = res.columns.str.strip()
    best_i = int(res["metrics/mAP50-95(B)"].values.argmax())  # Ultralytics 8.4 fitness = mAP50-95 (val)
    ck = torch.load(d / "weights" / "best.pt", map_location="cpu", weights_only=False)
    tm = ck.get("train_metrics", {})
    args = yaml.safe_load((d / "args.yaml").read_text())
    log = (ROOT / "train_v8lx.log").read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    # the log holds both runs in order; slice the part of this run
    start = log.find(f"name={run},")
    nxt = log.find("name=v8", start + 10)
    part = log[start: nxt if nxt > 0 else None]
    hours = re.search(r"(\d+) epochs completed in ([\d.]+) hours", part)
    opt = re.search(r"optimizer:\S*\s+(\w+\(lr=[^)]*\))", part)
    batch_note = [x for x in (ROOT / "catatan_batch.txt").read_text(encoding="utf-8").splitlines()
                  if f"run={run} " in x]
    return dict(run=run, epochs_selesai=int(res.epoch.iloc[-1]),
                lama_training_jam_log=float(hours.group(2)) if hours else "TIDAK TERSEDIA",
                waktu_results_csv_detik=float(res["time"].iloc[-1]),
                epoch_terbaik=int(res.epoch.iloc[best_i]),
                val_mAP50_95_epoch_terbaik=float(res["metrics/mAP50-95(B)"].iloc[best_i]),
                val_mAP50_epoch_terbaik=float(res["metrics/mAP50(B)"].iloc[best_i]),
                best_pt_train_metrics_mAP50_95=float(tm.get("metrics/mAP50-95(B)", np.nan)),
                batch_args_yaml=args["batch"], batch_train_args_ckpt=ck.get("train_args", {}).get("batch"),
                catatan_batch=batch_note[0] if batch_note else "TIDAK TERSEDIA",
                optimizer_args_yaml=args["optimizer"], lr0_args_yaml=args["lr0"],
                momentum_args_yaml=args["momentum"],
                optimizer_aktual_log=opt.group(1) if opt else "TIDAK TERSEDIA")


# ---------------------------------------------------------------- bootstrap
def head_to_head(results, sets, names, nc, pc, a_key, b_key):
    """Same procedure as section 1 of 02_error_analysis.py; fresh rng(SEED) per pair so every pair
    uses the same resampled images. mAP50 is also reported over the 3 classes without puntung_rokok."""
    rng = np.random.default_rng(SEED)
    A, B = eval02.label(*a_key), eval02.label(*b_key)
    three = [c for c, n in enumerate(names) if n != "puntung_rokok"]
    rows, L = [], [f"### {A} vs {B}", "", "Δ = YOLOv12 − YOLOv8. Untuk FN dan FP, Δ negatif berarti YOLOv12 lebih baik.", ""]
    for s, (imgs, gts, _) in sets.items():
        ra, rb = results[a_key + (s,)], results[b_key + (s,)]
        n = len(gts)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        w_all = np.stack([np.bincount(ix, minlength=n) for ix in idx])
        tab = []
        for c, cname in enumerate(names + ["TOTAL"]):
            cs = slice(None) if cname == "TOTAL" else c
            gt_n = int(ra["tp"][:, cs].sum() + ra["fn"][:, cs].sum())
            fa = ra["fn"][:, cs].reshape(n, -1).sum(1); fb = rb["fn"][:, cs].reshape(n, -1).sum(1)
            pa = ra["fp"][:, cs].reshape(n, -1).sum(1); pb = rb["fp"][:, cs].reshape(n, -1).sum(1)
            dfn, dfp = fa.sum() - fb.sum(), pa.sum() - pb.sum()
            ci_fn = np.percentile(w_all @ (fa - fb), [2.5, 97.5])
            ci_fp = np.percentile(w_all @ (pa - pb), [2.5, 97.5])
            no_gt = gt_n == 0 and cname != "TOTAL"
            tab.append({"Kelas": cname, "GT": gt_n,
                        "FN v12": "–" if no_gt else int(fa.sum()), "FN v8": "–" if no_gt else int(fb.sum()),
                        "ΔFN": "–" if no_gt else f"{int(dfn):+d}",
                        "95% CI ΔFN": "–" if no_gt else f"[{ci_fn[0]:+.1f}, {ci_fn[1]:+.1f}]",
                        "FP v12": int(pa.sum()), "FP v8": int(pb.sum()), "ΔFP": f"{int(dfp):+d}",
                        "95% CI ΔFP": f"[{ci_fp[0]:+.1f}, {ci_fp[1]:+.1f}]",
                        "Kesimpulan FN": "tidak ada GT" if no_gt else eval02.verdict(*ci_fn),
                        "Kesimpulan FP": eval02.verdict(*ci_fp)})
            for met, va, vb, dd, ci in (("FN", fa.sum(), fb.sum(), dfn, ci_fn), ("FP", pa.sum(), pb.sum(), dfp, ci_fp)):
                if met == "FN" and no_gt:
                    continue
                rows.append(dict(pasangan=f"{A} vs {B}", test_set=s, kelas=cname, metrik=met, v12=int(va), v8=int(vb),
                                 delta=int(dd), ci_lo=ci[0], ci_hi=ci[1], kesimpulan=eval02.verdict(*ci)))
        # mAP50: all classes with GT (as in 02) and 3 classes without puntung_rokok
        for tag, cls in (("mAP50_3kelas", three), ("mAP50_semua_kelas_ber_GT", None)):
            ma, mb = eval02.map50(gts, ra, nc), eval02.map50(gts, rb, nc)
            pick = lambda d: [c for c in d if (cls is None or c in cls) and not np.isnan(d[c])]
            va = float(np.mean([ma[c] for c in pick(ma)])); vb = float(np.mean([mb[c] for c in pick(mb)]))
            bm = []
            for w in w_all:
                xa, xb = eval02.map50(gts, ra, nc, img_w=w), eval02.map50(gts, rb, nc, img_w=w)
                common = [c for c in pick(xa) if c in pick(xb)]
                bm.append(np.mean([xa[c] for c in common]) - np.mean([xb[c] for c in common]))
            ci = np.nanpercentile(bm, [2.5, 97.5])
            rows.append(dict(pasangan=f"{A} vs {B}", test_set=s, kelas=f"{tag} ({len(pick(ma))} kelas)", metrik="mAP50",
                             v12=va, v8=vb, delta=va - vb, ci_lo=ci[0], ci_hi=ci[1],
                             kesimpulan=eval02.verdict(*ci, lower_is_better=False)))
        L += [f"#### Test {s} ({n} citra, bootstrap {N_BOOT}× berpasangan pada level citra)", ""]
        L += md(pd.DataFrame(tab)) + [""]
        for r in rows[-2:]:
            L.append(f"- {r['kelas']}: {A} {r['v12']!r}, {B} {r['v8']!r}, Δ {r['delta']!r}, "
                     f"95% CI [{r['ci_lo']!r}, {r['ci_hi']!r}] → {r['kesimpulan']}")
        sub = pc[(pc.test_set == s) & (pc["gt"] > 0) & pc.apply(
            lambda r: (r.model, r.varian, r.augmentasi) in (a_key, b_key), axis=1)]
        br = sub.pivot_table(index="kelas", columns="model",
                             values=["FN_terlewat", "FN_salah_kelas", "FP_latar", "FP_salah_kelas"], aggfunc="sum")
        br.columns = [f"{a} {b}" for a, b in br.columns]
        L += ["", "Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas), kelas yang punya GT:", ""]
        L += md(br.reset_index()) + [""]
    return rows, L


# ---------------------------------------------------------------- label noise
def label_noise(runs, sets, names):
    k, p = names.index("kertas_tisu"), names.index("plastik")
    groups, *_ = lbl04.d0_copies()
    test_rows = groups[groups.split == "test"].set_index("filename")
    src = lambda f: test_rows.sumber.get(f, lbl04.source_heuristic(f))
    ev = lbl04.collect_events(runs, sets, k, p)
    ev["sumber"] = ev.file.map(src)
    ev["arah"] = ["Kertas/Tisu → Plastik" if g == k else "Plastik → Kertas/Tisu" for g in ev.gt_cls]
    gtab = lbl04.gt_table(sets, k, p, src)
    clean = ev[ev.test_set == "clean"]
    g = gtab[gtab.test_set == "clean"]
    gt_total = int(g.n_kertas.sum() + g.n_plastik.sum())

    def per_source(e, n_models, label):
        obj = e.groupby(["file", "gt_idx"], as_index=False).first()
        rows = []
        for so in SOURCES + ["TOTAL"]:
            gs = g if so == "TOTAL" else g[g.sumber == so]
            es = e if so == "TOTAL" else e[e.sumber == so]
            os_ = obj if so == "TOTAL" else obj[obj.sumber == so]
            n_gt = int(gs.n_kertas.sum() + gs.n_plastik.sum())
            rows.append({"model": label, "sumber": so, "GT kertas+plastik": n_gt, "% GT": n_gt / gt_total,
                         "kasus": len(es), "% kasus": len(es) / max(len(e), 1),
                         "kasus per GT per model": len(es) / max(n_gt * n_models, 1),
                         "objek GT unik tertukar (min. 1 model)": len(os_),
                         "% objek GT tertukar (min. 1 model)": len(os_) / max(n_gt, 1),
                         "Kertas/Tisu → Plastik": int((es.arah == "Kertas/Tisu → Plastik").sum()),
                         "Plastik → Kertas/Tisu": int((es.arah == "Plastik → Kertas/Tisu").sum())})
        return rows

    labels13 = {lbl04.model_label(r) for r in runs if r["run"] not in NEW_RUNS}
    out = []
    for r in runs:
        if r["run"] in NEW_RUNS:
            lab = lbl04.model_label(r)
            out += per_source(clean[clean.model == lab], 1, lab)
    out += per_source(clean[clean.model.isin(labels13)], len(labels13), "13 model (lama)")
    out += per_source(clean, len(runs), f"{len(runs)} model")
    return pd.DataFrame(out)


# ---------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names = yaml.safe_load((ROOT / "TA-1" / "data.yaml").read_text())["names"]
    nc = len(names)
    runs, _ = eval01.inventory()
    runs = [r for r in runs if r["best_pt"].exists() and r["epochs"] == 100]  # 13 lama + v8l, v8x
    assert len(runs) == 15, [r["run"] for r in runs]
    sets = {s: eval02.load_test_set(s) for s in ("full", "clean")}
    ov = pd.read_csv(OUT / f"test_metrics_overall_{SUFFIX}.csv")

    # ------------------------------------------------ per-model analysis (as in 02)
    results, rows, cm_rows = {}, [], []
    for rec in runs:
        key = key_of(rec)
        for s, (imgs, gts, rej) in sets.items():
            preds, gts_val = eval02.predict(rec, s, imgs)  # cache only (no GPU): all 15 are cached
            sets[s] = (imgs, gts_val, rej)
            res = eval02.analyse(gts_val, preds, nc)
            results[key + (s,)] = res
            aps = eval02.map50(gts_val, res, nc)
            for c, n in enumerate(names):
                tp, fn, fp = res["tp"][:, c].sum(), res["fn"][:, c].sum(), res["fp"][:, c].sum()
                rows.append(dict(model=rec["model"], varian=rec["varian"], augmentasi=rec["augmentasi"],
                                 test_set=s, kelas=n, gt=int(tp + fn), TP=int(tp), FP=int(fp), FN=int(fn),
                                 FN_terlewat=int(res["cm"][c, nc]), FN_salah_kelas=int(fn - res["cm"][c, nc]),
                                 FP_latar=int(res["cm"][nc, c]), FP_salah_kelas=int(fp - res["cm"][nc, c]),
                                 precision=tp / (tp + fp) if tp + fp else np.nan,
                                 recall=tp / (tp + fn) if tp + fn else np.nan, AP50=aps.get(c, np.nan)))
            if rec["run"] in NEW_RUNS:
                eval02.plot_cm(res["cm"], names, f"{eval02.label(*key)} — test {s} ({len(imgs)} citra)",
                               eval02.CM_DIR / f"{eval02.fname(*key)}_{s}.png")
                lab = [*names, "latar"]
                for i, t in enumerate(lab):
                    for j, pr in enumerate(lab):
                        cm_rows.append(dict(model=eval02.label(*key), test_set=s, sebenarnya=t, prediksi=pr,
                                            jumlah=int(res["cm"][i, j])))
    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / f"fp_fn_per_class_{SUFFIX}.csv", index=False)
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(OUT / f"confusion_matrix_v8lx_{SUFFIX}.csv", index=False)

    # check: re-implemented matching reproduces the old fp_fn_per_class.csv for the 13 old models
    old_fp = pd.read_csv(OUT / "fp_fn_per_class.csv")
    chk = old_fp.merge(pc, on=["model", "varian", "augmentasi", "test_set", "kelas"], suffixes=("_lama", ""))
    same = all((chk[f"{c}_lama"] == chk[c]).all() for c in ("TP", "FP", "FN"))
    print(f"Cek ulang 13 model lama vs fp_fn_per_class.csv: {len(chk)} baris, TP/FP/FN identik = {same}")

    # ------------------------------------------------ training info
    tr = pd.DataFrame([training_info(r) for r in NEW_RUNS])
    tr.to_csv(OUT / f"training_v8lx_{SUFFIX}.csv", index=False)

    # ------------------------------------------------ bootstrap
    boot, boot_md = [], []
    for a, b in PAIRS:
        r, L = head_to_head(results, sets, names, nc, pc, a, b)
        boot += r
        boot_md += L
    pd.DataFrame(boot).to_csv(OUT / f"bootstrap_head_to_head_{SUFFIX}.csv", index=False)

    # ------------------------------------------------ label noise
    ln = label_noise(runs, sets, names)
    ln.to_csv(OUT / f"label_noise_per_sumber_{SUFFIX}.csv", index=False)

    # ------------------------------------------------ full - clean
    f = ov[ov.test_set == "full"].set_index(["model", "varian", "augmentasi"])
    c = ov[ov.test_set == "clean"].set_index(["model", "varian", "augmentasi"])
    dd = pd.DataFrame({"mAP50_full_3kelas": f.mAP50_3kelas, "mAP50_clean": c.mAP50,
                       "mAP50_95_full_3kelas": f.mAP50_95_3kelas, "mAP50_95_clean": c.mAP50_95})
    dd["selisih_full_minus_clean_mAP50"] = dd.mAP50_full_3kelas - dd.mAP50_clean
    dd["selisih_full_minus_clean_mAP50_95"] = dd.mAP50_95_full_3kelas - dd.mAP50_95_clean
    dd = dd.reset_index()
    dd.to_csv(OUT / f"selisih_full_clean_{SUFFIX}.csv", index=False)
    is_new = (dd.model == "YOLOv8") & dd.varian.isin(["l", "x"])
    agg = []
    for grp, m in (("13 model (lama)", ~is_new), ("15 model", pd.Series(True, index=dd.index)),
                   ("5 model YOLOv8", dd.model == "YOLOv8"), ("3 model YOLOv8 lama (n/s/m)", (dd.model == "YOLOv8") & ~is_new)):
        for col in ("selisih_full_minus_clean_mAP50", "selisih_full_minus_clean_mAP50_95"):
            x = dd.loc[m, col]
            agg.append(dict(kelompok=grp, n=int(m.sum()), metrik=col, rata_rata=float(x.mean()),
                            minimum=float(x.min()), model_min=eval02.label(*dd.loc[x.idxmin(), ["model", "varian", "augmentasi"]]),
                            maksimum=float(x.max()), model_maks=eval02.label(*dd.loc[x.idxmax(), ["model", "varian", "augmentasi"]])))
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT / f"selisih_full_clean_agregat_{SUFFIX}.csv", index=False)

    # ------------------------------------------------ ranking augmented
    aug = ov[ov.augmentasi == "ya"].copy()
    aug["model_label"] = [eval02.label(m, v, a) for m, v, a in zip(aug.model, aug.varian, aug.augmentasi)]
    rk_full = aug[aug.test_set == "full"].sort_values("mAP50", ascending=False)
    rk_clean = aug[aug.test_set == "clean"].sort_values("mAP50", ascending=False)
    rk = pd.concat([
        rk_full.assign(peringkat=range(1, len(rk_full) + 1), kriteria="mAP50 4 kelas, test full")[
            ["kriteria", "peringkat", "model_label", "mAP50", "mAP50_95", "mAP50_3kelas", "mAP50_95_3kelas", "ms_per_img"]],
        rk_clean.assign(peringkat=range(1, len(rk_clean) + 1), kriteria="mAP50 3 kelas, test clean")[
            ["kriteria", "peringkat", "model_label", "mAP50", "mAP50_95", "mAP50_3kelas", "mAP50_95_3kelas", "ms_per_img"]]])
    rk.to_csv(OUT / f"peringkat_augmented_{SUFFIX}.csv", index=False)

    # save the markdown fragments for hasil_v8lx.md
    (OUT / f"_bootstrap_{SUFFIX}.md").write_text("\n".join(boot_md), encoding="utf-8")
    print("Selesai. CSV di results/*_15model.csv")


if __name__ == "__main__":
    sys.exit(main())

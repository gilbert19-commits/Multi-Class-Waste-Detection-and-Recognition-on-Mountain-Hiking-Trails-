"""LANGKAH 2 (tabel) dan 4: tabel metrik, sanity check, paired bootstrap 1000x level gambar memakai ap_per_class Ultralytics.

mAP dihitung ulang dari prediksi per gambar (_cache/*.pkl) dengan ultralytics.utils.metrics.ap_per_class, untuk 550 gambar
penuh (harus sama dengan validator, selisih maksimal 0.001), untuk 548/592 gambar (cara default Ultralytics, cross-check
dengan angka training), dan untuk test tanpa gambar yang sumbernya juga ada di train (tabel pelengkap, bukan angka utama).
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hk_common import CACHE, FAMILIES, MODELS, NAMES, OUT, RUNS, SCALES, SEED, best_epoch, family, label, read_results

N_BOOT = 1000
N_WORKERS = 6
TOL = 0.001


def load_images(m, split):
    d = pickle.loads((CACHE / f"{m}_2kelas_{split}.pkl").read_bytes())["images"]
    files = sorted(d)
    return files, [d[f] for f in files]


def ap_matrix(imgs):
    """Return AP (nc, 10) dari ap_per_class. Baris kelas yang tidak punya GT = NaN."""
    from ultralytics.utils.metrics import ap_per_class

    tp = np.concatenate([x["tp"] for x in imgs])
    conf = np.concatenate([x["pr_conf"] for x in imgs])
    cls = np.concatenate([x["pr_cls"] for x in imgs])
    target = np.concatenate([x["gt_cls"] for x in imgs])
    r = ap_per_class(tp, conf, cls, target)
    ap, uc = r[5], r[6]
    out = np.full((len(NAMES), 10), np.nan)
    out[uc.astype(int)] = ap
    return out


def summarize(ap):
    return dict(mAP50=float(np.nanmean(ap[:, 0])), mAP75=float(np.nanmean(ap[:, 5])), mAP50_95=float(np.nanmean(ap.mean(1))))


def boot_worker(m):
    f = CACHE / f"boot_{m}.npy"
    if f.exists():
        return m
    _, imgs = load_images(m, "test")
    idx = np.load(CACHE / "boot_idx.npy")
    res = np.stack([ap_matrix([imgs[i] for i in ix]) for ix in idx])
    np.save(f, res)
    return m


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw = json.loads((OUT / "_eval_raw.json").read_text())
    overlap = pd.read_csv(OUT / "tumpang_tindih_train.csv")
    stats = pd.read_csv(OUT / "dataset_stats.csv")
    thr = int(stats[stats.item == "ambang_inlier_ORB_maks_kontrol_acak"].nilai.iloc[0])
    fixed = pd.read_csv(OUT / "label_campuran_diperbaiki.csv")
    leak = set(overlap[(overlap.split == "test") & (overlap.inlier_nama_sama > thr)].file)

    # ---- tabel metrik dan sanity check
    t_rows, c_rows, v_rows, sens, check = [], [], [], [], []
    for m in MODELS:
        r = raw[f"{m}|test"]
        files, imgs = load_images(m, "test")
        full = summarize(ap_matrix(imgs))
        drop = set(fixed[fixed.split == "test"].file.str.replace(".txt", ".jpg"))
        keep548 = [x for f, x in zip(files, imgs) if f not in drop]
        if len(keep548) != 548:
            raise RuntimeError(f"subset 548 salah: {len(keep548)}")
        s548 = summarize(ap_matrix(keep548))
        nl = [x for f, x in zip(files, imgs) if f not in leak]
        s_nl = summarize(ap_matrix(nl))
        diff = max(abs(full["mAP50"] - r["mAP50"]), abs(full["mAP50_95"] - r["mAP50_95"]), abs(full["mAP75"] - r["mAP75"]))
        check.append(dict(model=label(m), split="test", n_gambar=len(imgs), mAP50_validator=r["mAP50"], mAP50_hitung_ulang=full["mAP50"],
                          mAP50_95_validator=r["mAP50_95"], mAP50_95_hitung_ulang=full["mAP50_95"], selisih_maks=diff))
        t_rows.append(dict(model=label(m), family=family(m), skala=m[-1], n_gambar=r["n_images"], precision=r["precision"],
                           recall=r["recall"], mAP50=r["mAP50"], mAP75=r["mAP75"], mAP50_95=r["mAP50_95"],
                           params_M=r["params_fused"] / 1e6, GFLOPs_512=r["GFLOPs_512_fused"],
                           params_M_tanpa_fuse=r["params_tanpa_fuse"] / 1e6, GFLOPs_512_tanpa_fuse=r["GFLOPs_512_tanpa_fuse"],
                           mode_postprocess="end-to-end tanpa NMS" if r["end2end_validator"] else "NMS",
                           end2end_validator=r["end2end_validator"], nms_end2end_arg=r.get("nms_end2end_arg"),
                           bentuk_output_model="x".join(map(str, r.get("bentuk_output_model", []))),
                           mAP50_548=s548["mAP50"], mAP75_548=s548["mAP75"], mAP50_95_548=s548["mAP50_95"]))
        for cname, c in r["per_class"].items():
            from hk_common import PAPER

            c_rows.append(dict(model=label(m), family=family(m), skala=m[-1], kelas=cname, kelas_paper=PAPER[cname], gambar=c["images"],
                               objek=c["instances"], precision=c["precision"], recall=c["recall"], AP50=c["AP50"], AP75=c["AP75"],
                               AP50_95=c["AP50_95"]))
        sens.append(dict(model=label(m), family=family(m), skala=m[-1], n_gambar=len(nl), n_gambar_dibuang=len(imgs) - len(nl),
                         mAP50=s_nl["mAP50"], mAP75=s_nl["mAP75"], mAP50_95=s_nl["mAP50_95"],
                         selisih_mAP50_95_terhadap_550=s_nl["mAP50_95"] - full["mAP50_95"]))
        # valid
        rv = raw[f"{m}|val"]
        fv, iv = load_images(m, "val")
        full_v = summarize(ap_matrix(iv))
        dropv = set(fixed[fixed.split == "valid"].file.str.replace(".txt", ".jpg"))
        keep592 = [x for f, x in zip(fv, iv) if f not in dropv]
        s592 = summarize(ap_matrix(keep592))
        d = read_results(RUNS / f"{m}_2kelas" / "results.csv")
        be = best_epoch(d)
        tr50 = float(d.loc[d.epoch == be, "metrics/mAP50(B)"].iloc[0])
        tr95 = float(d.loc[d.epoch == be, "metrics/mAP50-95(B)"].iloc[0])
        v_rows.append(dict(model=label(m), family=family(m), skala=m[-1], n_gambar=rv["n_images"], precision=rv["precision"],
                           recall=rv["recall"], mAP50=rv["mAP50"], mAP75=rv["mAP75"], mAP50_95=rv["mAP50_95"],
                           mAP50_592=s592["mAP50"], mAP50_95_592=s592["mAP50_95"], best_epoch=be, mAP50_training=tr50,
                           mAP50_95_training=tr95, selisih_mAP50_592_vs_training=s592["mAP50"] - tr50,
                           selisih_mAP50_95_592_vs_training=s592["mAP50_95"] - tr95))
        check.append(dict(model=label(m), split="val", n_gambar=len(iv), mAP50_validator=rv["mAP50"], mAP50_hitung_ulang=full_v["mAP50"],
                          mAP50_95_validator=rv["mAP50_95"], mAP50_95_hitung_ulang=full_v["mAP50_95"],
                          selisih_maks=max(abs(full_v["mAP50"] - rv["mAP50"]), abs(full_v["mAP50_95"] - rv["mAP50_95"]))))
        print(f"[tabel] {label(m)} selisih hitung ulang {diff:.2e}", flush=True)
    T = pd.DataFrame(t_rows)
    T.to_csv(OUT / "metrics_test.csv", index=False)
    pd.DataFrame(c_rows).to_csv(OUT / "metrics_per_class.csv", index=False)
    pd.DataFrame(v_rows).to_csv(OUT / "metrics_val.csv", index=False)
    pd.DataFrame(sens).to_csv(OUT / "metrics_test_tanpa_tumpang_tindih.csv", index=False)
    C = pd.DataFrame(check)
    C.to_csv(OUT / "sanity_check_map.csv", index=False)
    if C.selisih_maks.max() > TOL:
        print("[BERHENTI] sanity check gagal\n", C.to_string())
        sys.exit(1)
    print("sanity check OK, selisih maksimum:", C.selisih_maks.max(), flush=True)

    # ---- bootstrap
    files, _ = load_images(MODELS[0], "test")
    for m in MODELS[1:]:
        if load_images(m, "test")[0] != files:
            raise RuntimeError(f"himpunan gambar {m} berbeda")
    n = len(files)
    rng = np.random.default_rng(SEED)
    np.save(CACHE / "boot_idx.npy", rng.integers(0, n, size=(N_BOOT, n)))
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for done in ex.map(boot_worker, MODELS):
            print(f"[bootstrap] {label(done)} selesai", flush=True)
    boots = {m: np.load(CACHE / f"boot_{m}.npy") for m in MODELS}
    full = {m: ap_matrix(load_images(m, "test")[1]) for m in MODELS}
    metric = {"mAP50": lambda a: np.nanmean(a[..., 0], -1), "mAP50_95": lambda a: np.nanmean(a.mean(-1), -1)}
    best = {f: T[T.family == f].sort_values("mAP50_95", ascending=False).iloc[0] for f in FAMILIES}
    lab2m = {label(m): m for m in MODELS}
    pairs = []
    for s in SCALES:
        pairs += [("per_skala", s, f"yolo26{s}", f"yolov12{s}"), ("per_skala", s, f"yolo26{s}", f"yolov8{s}"),
                  ("per_skala", s, f"yolov12{s}", f"yolov8{s}")]
    b26, b12, b8 = (lab2m[best[f]["model"]] for f in ("YOLO26", "YOLOv12", "YOLOv8"))
    pairs += [("terbaik_family", "-", b26, b12), ("terbaik_family", "-", b26, b8), ("terbaik_family", "-", b12, b8)]
    rows = []
    for kind, s, a, b in pairs:
        for name, fn in metric.items():
            dist = (fn(boots[a]) - fn(boots[b])) * 100
            lo, hi = np.percentile(dist, [2.5, 97.5])
            va, vb = float(fn(full[a])) * 100, float(fn(full[b])) * 100
            cross0 = bool(lo <= 0 <= hi)
            rows.append(dict(jenis=kind, skala=s, model_A=label(a), model_B=label(b), metrik=name, A_persen=va, B_persen=vb,
                             delta_poin_persen=va - vb, ci95_bawah=lo, ci95_atas=hi, ci_melewati_nol=cross0,
                             lebih_baik="setara" if cross0 else (label(a) if lo > 0 else label(b)), n_gambar=n,
                             n_resample=N_BOOT, seed=SEED))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "bootstrap.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(T[["model", "n_gambar", "precision", "recall", "mAP50", "mAP75", "mAP50_95", "params_M", "GFLOPs_512", "mode_postprocess",
             "mAP50_95_548"]].round(5).to_string(index=False))
    print(B[["model_A", "model_B", "metrik", "delta_poin_persen", "ci95_bawah", "ci95_atas", "lebih_baik"]].round(3).to_string(index=False))
    print("[SELESAI s4]", flush=True)


if __name__ == "__main__":
    main()

"""BAGIAN A: empat versi mAP di 550 citra test, efek bobot dan efek salah kelas, verifikasi a, b, d, dan rincian per kelas.

Semua prediksi berasal dari hasil_2kelas/_cache (best.pt, imgsz 512, conf 0.001, iou 0.7). Tidak ada inferensi baru di sini.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from ultralytics.utils.metrics import box_iou

from up_common import CACHE, HK, METRICS, MODELS, NAMES, NMS_IOU, OUT, PAPER, agnostic_keep, ap_matrix, arrays, best_std, family, image_versions, label, load_preds, summarize

sys.path.insert(0, str(HK / "scripts"))
from s5_error import match_image  # noqa: E402  (conf 0.25, IoU 0.5, satu prediksi untuk satu GT, sama dengan recall_breakdown.csv)

TOL_A, TOL_B = 0.001, 0.005


def ap_rumus_lama(imgs, version):
    """AP dengan titik penutup lama, mrec = [0, r..., 1], mpre = [1, p..., 0]. Hanya untuk melihat besar pengaruh rumus."""
    tp, conf, cls, target = arrays(imgs, version)
    order = np.argsort(-conf)
    tp, cls = tp[order], cls[order]
    x = np.linspace(0, 1, 101)
    out = []
    for c in np.unique(target):
        sel = cls == c
        ap = np.zeros(tp.shape[1])
        if sel.sum():
            tpc, fpc = tp[sel].cumsum(0), (1 - tp[sel]).cumsum(0)
            for j in range(tp.shape[1]):
                mrec = np.r_[0.0, tpc[:, j] / (target == c).sum(), 1.0]
                mpre = np.r_[1.0, tpc[:, j] / (tpc[:, j] + fpc[:, j]), 0.0]
                mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                ap[j] = np.trapezoid(np.interp(x, mrec, mpre), x)
        out.append(ap)
    return np.array(out)


def nms_diagnosis(m):
    """Prediksi yang dihapus NMS ulang: adakah prediksi yang bertahan dengan IoU > 0.7 dan kelas berbeda, atau hanya kelas sama."""
    beda = sama = 0
    for x in load_preds(m).values():
        n = len(x["pr_conf"])
        keep = agnostic_keep(x["pr_box"], x["pr_conf"])
        gone = np.setdiff1d(np.arange(n), keep)
        if len(gone):
            iou = box_iou(torch.from_numpy(x["pr_box"][gone]), torch.from_numpy(x["pr_box"][keep])).numpy() > NMS_IOU
            b = (iou & (x["pr_cls"][gone][:, None] != x["pr_cls"][keep][None, :])).any(1)
            beda += int(b.sum())
            sama += int((~b).sum())
    return dict(n_dihapus_nms_ulang_ada_penekan_kelas_beda=beda, n_dihapus_nms_ulang_hanya_kelas_sama=sama)


def build_versions():
    CACHE.mkdir(parents=True, exist_ok=True)
    diff_tp = []
    for m in MODELS:
        d = load_preds(m)
        files, recs = [], []
        n_tp_beda = 0
        for f, x in d.items():
            r = image_versions(x["gt_cls"], x["gt_box"], x["pr_cls"], x["pr_conf"], x["pr_box"])
            n_tp_beda += int((r["tp"] != x["tp"]).sum())
            files.append(f)
            recs.append(r)
        diff_tp.append(dict(model=label(m), sel_tp_berbeda_dari_validator=n_tp_beda, n_sel=int(sum(x["tp"].size for x in d.values()))))
        (CACHE / f"versi_{m}.pkl").write_bytes(pickle.dumps(dict(files=files, images=recs)))
    return pd.DataFrame(diff_tp)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 240, "display.max_columns", 40)
    dest = OUT / "class_agnostic"
    dest.mkdir(parents=True, exist_ok=True)
    tp_check = build_versions()
    raw = json.loads((HK / "_eval_raw.json").read_text())
    single_f = OUT / "_singlecls_raw.json"
    single = json.loads(single_f.read_text()) if single_f.exists() else {}

    rows, va, vb, vd, lama = [], [], [], [], []
    for m in MODELS:
        v = pickle.loads((CACHE / f"versi_{m}.pkl").read_bytes())["images"]
        s = {k: summarize(ap_matrix(v, k)) for k in ("v1", "v2", "v3", "v4")}
        row = dict(model=label(m), family=family(m), skala=m[-1], n_gambar=len(v), n_prediksi=int(sum(len(x["conf"]) for x in v)),
                   n_prediksi_setelah_nms_ulang=int(sum(len(x["keep3"]) for x in v)))
        for met in METRICS:
            for k, nm in (("v1", "1_standar"), ("v2", "2_pooled"), ("v3", "3_agnostic"), ("v4", "4_agnostic_tanpa_nms")):
                row[f"{met}_{nm}"] = s[k][met]
            row[f"{met}_efek_bobot"] = s["v2"][met] - s["v1"][met]
            row[f"{met}_efek_salah_kelas"] = s["v3"][met] - s["v2"][met]
            row[f"{met}_efek_nms_ulang"] = s["v3"][met] - s["v4"][met]
        row.update(nms_diagnosis(m))
        rows.append(row)
        # verifikasi a: script vs Ultralytics
        r = raw[f"{m}|test"]
        va.append(dict(model=label(m), **{f"{met}_ultralytics": r[met] for met in METRICS}, **{f"{met}_script": s["v1"][met] for met in METRICS},
                       selisih_maks=max(abs(r[met] - s["v1"][met]) for met in METRICS)))
        # verifikasi b: single_cls=True vs versi 3 dan 4
        if m in single:
            q = single[m]
            vb.append(dict(model=label(m), family=family(m), mode_validator="end-to-end, tanpa NMS" if q["end2end"] else "NMS agnostic",
                           n_gambar=q["n_gambar"], n_objek=q["n_objek"], kelas_gt=str(q["kelas_gt"]),
                           n_prediksi_single_cls=q["n_prediksi"], n_prediksi_v3=row["n_prediksi_setelah_nms_ulang"], n_prediksi_v4=row["n_prediksi"],
                           **{f"{met}_single_cls": q[met] for met in METRICS},
                           **{f"{met}_selisih_v3": q[met] - s["v3"][met] for met in METRICS},
                           **{f"{met}_selisih_v4": q[met] - s["v4"][met] for met in METRICS},
                           selisih_maks_v3=max(abs(q[met] - s["v3"][met]) for met in METRICS),
                           selisih_maks_v4=max(abs(q[met] - s["v4"][met]) for met in METRICS)))
        # verifikasi d: versi 3 >= versi 2
        for met in METRICS:
            vd.append(dict(model=label(m), family=family(m), metrik=met, v2=s["v2"][met], v3=s["v3"][met], v4=s["v4"][met], selisih_v3_v2=s["v3"][met] - s["v2"][met],
                           v3_lebih_rendah=s["v3"][met] < s["v2"][met]))
        # pengaruh rumus AP lama
        for k, nm in (("v1", "standar"), ("v3", "agnostic")):
            sl = summarize(ap_rumus_lama(v, k))
            lama.append(dict(model=label(m), versi=nm, **{f"{met}_terpasang": s[k][met] for met in METRICS}, **{f"{met}_rumus_lama": sl[met] for met in METRICS},
                             **{f"{met}_selisih_poin": (sl[met] - s[k][met]) * 100 for met in METRICS}))
    T = pd.DataFrame(rows)
    T.to_csv(dest / "metrics_4versi.csv", index=False)
    A = pd.DataFrame(va).merge(tp_check, on="model")
    A.to_csv(dest / "verifikasi_a_standar_vs_ultralytics.csv", index=False)
    D = pd.DataFrame(vd)
    D.to_csv(dest / "verifikasi_d_v3_vs_v2.csv", index=False)
    pd.DataFrame(lama).to_csv(dest / "efek_rumus_ap_lama.csv", index=False)
    print(T[["model"] + [c for c in T.columns if c.startswith("mAP50_95")]].round(4).to_string(index=False))
    print(T[["model"] + [c for c in T.columns if c.startswith("mAP50_") and not c.startswith("mAP50_95")]].round(4).to_string(index=False))
    print("VERIF a: selisih maks", A.selisih_maks.max(), "| sel TP berbeda:", A.sel_tp_berbeda_dari_validator.sum())
    print("VERIF d: lebih rendah\n", D[D.v3_lebih_rendah].round(5).to_string(index=False))
    if vb:
        B = pd.DataFrame(vb)
        B.to_csv(dest / "verifikasi_b_single_cls.csv", index=False)
        print("VERIF b\n", B[["model", "mode_validator", "n_prediksi_single_cls", "n_prediksi_v3", "n_prediksi_v4", "mAP50_95_single_cls", "mAP50_selisih_v3", "mAP50_95_selisih_v3",
                              "selisih_maks_v3", "selisih_maks_v4"]].round(6).to_string(index=False))

    # rincian per kelas GT, model terbaik tiap family (conf 0.25, IoU 0.5)
    best = best_std()
    out = []
    for f, m in best.items():
        d = load_preds(m)
        cnt = {c: np.zeros(3, int) for c in range(len(NAMES))}
        for x in d.values():
            gs, _, _ = match_image(x)
            for c, g in zip(x["gt_cls"], gs):
                cnt[int(c)][g] += 1
        rec = pd.read_csv(HK / "recall_breakdown.csv")
        for c, n in cnt.items():
            tot = int(n.sum())
            cek = rec[(rec.model == label(m)) & (rec.dimensi == "kelas") & (rec.kelas == PAPER[NAMES[c]])].iloc[0]
            if int(cek.TP) != int(n[1]) or int(cek.FN_salah_kelas) != int(n[2]) or int(cek.FN_terlewat) != int(n[0]):
                raise RuntimeError(f"rincian {label(m)} {NAMES[c]} tidak sama dengan recall_breakdown.csv")
            out.append(dict(model=label(m), family=f, kelas_gt=PAPER[NAMES[c]], n_objek=tot, n_kelas_benar=int(n[1]), n_kelas_salah=int(n[2]), n_tidak_terdeteksi=int(n[0]),
                            persen_kelas_benar=n[1] / tot * 100, persen_kelas_salah=n[2] / tot * 100, persen_tidak_terdeteksi=n[0] / tot * 100, conf=0.25, iou=0.5))
    R = pd.DataFrame(out)
    R.to_csv(dest / "rincian_per_kelas.csv", index=False)
    print(R.round(2).to_string(index=False))
    print("[SELESAI a3]")


if __name__ == "__main__":
    main()

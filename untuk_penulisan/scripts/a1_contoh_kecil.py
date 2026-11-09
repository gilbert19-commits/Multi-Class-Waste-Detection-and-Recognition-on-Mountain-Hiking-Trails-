"""VERIFIKASI c: contoh kecil satu gambar 512x512. Hitungan manual tidak memakai ap_per_class maupun fungsi script.

Ada dua rumus titik penutup kurva PR:
- terpasang (Ultralytics 8.4.123, compute_ap): mrec = [0, r..., r[-1], 1], mpre = [1, p..., 0, 0]. Precision jatuh ke 0 tepat
  setelah recall maksimum.
- lama: mrec = [0, r..., 1], mpre = [1, p..., 0]. Ada garis miring dari titik terakhir ke (1, 0).
Script (dan seluruh angka paper) memakai rumus terpasang. Perkiraan 0,29 dan 0,78 di permintaan cocok dengan rumus lama.
"""
import sys

import numpy as np
import pandas as pd

from up_common import OUT, ap_matrix, image_versions, summarize

LOGAM, RINGAN = 1, 0  # urutan NAMES: 0 kertas_plastik_puntungRokok (ringan), 1 logam_kaleng_kaca
GT_CLS = np.array([LOGAM, RINGAN, RINGAN])
GT_BOX = np.array([[60, 60, 140, 140], [200, 60, 300, 160], [100, 300, 130, 320]], dtype=np.float32)
PR_CLS = np.array([RINGAN, RINGAN, RINGAN])
PR_CONF = np.array([0.9, 0.8, 0.4], dtype=np.float32)
PR_BOX = np.array([[205, 65, 300, 160], [65, 65, 140, 140], [350, 350, 420, 400]], dtype=np.float32)


def manual_ap(tp_sorted, n_gt, rumus):
    """AP interpolasi 101 titik + trapezoid, ditulis langsung dari definisinya."""
    tpc = np.cumsum(tp_sorted)
    fpc = np.cumsum(~np.asarray(tp_sorted))
    rec, pre = tpc / n_gt, tpc / (tpc + fpc)
    if rumus == "terpasang":
        mrec, mpre = np.r_[0.0, rec, rec[-1], 1.0], np.r_[1.0, pre, 0.0, 0.0]
    else:
        mrec, mpre = np.r_[0.0, rec, 1.0], np.r_[1.0, pre, 0.0]
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def manual_all(rumus):
    # mAP@50: P1 cocok G2 (IoU 0.9025), P2 menimpa G1 (IoU 0.8789) tapi beda kelas, P3 tidak menimpa apa pun
    ap_ringan = manual_ap([True, False, False], 2, rumus)  # kelas ringan: 2 GT
    ap_logam = 0.0                                         # kelas logam: 1 GT, tanpa prediksi
    return dict(v1=(ap_ringan + ap_logam) / 2, v2=manual_ap([True, False, False], 3, rumus), v3=manual_ap([True, True, False], 3, rumus),
                v4=manual_ap([True, True, False], 3, rumus), ap_ringan=ap_ringan)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    im = image_versions(GT_CLS, GT_BOX, PR_CLS, PR_CONF, PR_BOX)
    res = {v: summarize(ap_matrix([im], v)) for v in ("v1", "v2", "v3", "v4")}
    man, lama = manual_all("terpasang"), manual_all("lama")
    tp_txt = lambda b: "TP" if b else "FP"  # noqa: E731
    rows = [
        dict(pengecekan="P1 di standar (IoU 0.5)", harapan="TP", hasil=tp_txt(im["tp"][0, 0]), cocok=bool(im["tp"][0, 0])),
        dict(pengecekan="P2 di standar (IoU 0.5)", harapan="FP", hasil=tp_txt(im["tp"][1, 0]), cocok=not im["tp"][1, 0]),
        dict(pengecekan="P2 di class-agnostic (IoU 0.5)", harapan="TP", hasil=tp_txt(im["tp3"][1, 0]), cocok=bool(im["tp3"][1, 0])),
        dict(pengecekan="P3 di class-agnostic (IoU 0.5)", harapan="FP", hasil=tp_txt(im["tp3"][2, 0]), cocok=not im["tp3"][2, 0]),
        dict(pengecekan="NMS ulang mempertahankan ketiga prediksi", harapan=3, hasil=len(im["keep3"]), cocok=len(im["keep3"]) == 3),
    ]
    for name, v in (("standar (v1)", "v1"), ("pooled class-aware (v2)", "v2"), ("class-agnostic (v3)", "v3"), ("class-agnostic tanpa NMS (v4)", "v4")):
        rows.append(dict(pengecekan=f"mAP@50 {name}: script vs manual rumus terpasang", harapan=round(man[v], 6), hasil=round(res[v]["mAP50"], 6),
                         cocok=abs(res[v]["mAP50"] - man[v]) < 1e-6))
    R = pd.DataFrame(rows)
    # Baris informasi: perkiraan di permintaan dibandingkan dengan rumus lama, bukan syarat lulus
    info = pd.DataFrame([
        dict(pengecekan="Perkiraan permintaan standar 0.29 vs manual rumus lama", harapan=0.29, hasil=round(lama["v1"], 6), cocok=abs(lama["v1"] - 0.29) < 0.01),
        dict(pengecekan="Perkiraan permintaan class-agnostic 0.78 vs manual rumus lama", harapan=0.78, hasil=round(lama["v3"], 6), cocok=abs(lama["v3"] - 0.78) < 0.01),
        dict(pengecekan="Perkiraan permintaan standar 0.29 vs script (rumus terpasang)", harapan=0.29, hasil=round(res["v1"]["mAP50"], 6), cocok=abs(res["v1"]["mAP50"] - 0.29) < 0.01),
        dict(pengecekan="Perkiraan permintaan class-agnostic 0.78 vs script (rumus terpasang)", harapan=0.78, hasil=round(res["v3"]["mAP50"], 6), cocok=abs(res["v3"]["mAP50"] - 0.78) < 0.01),
    ])
    (OUT / "class_agnostic").mkdir(parents=True, exist_ok=True)
    pd.concat([R.assign(jenis="syarat"), info.assign(jenis="informasi")]).to_csv(OUT / "class_agnostic" / "verifikasi_c_contoh_kecil.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print(R.to_string(index=False))
    print(info.to_string(index=False))
    print("manual terpasang:", man, "\nmanual lama:", lama)
    print("script v1:", res["v1"], "\nscript v3:", res["v3"])
    if not R.cocok.all():
        sys.exit(1)
    print("[SELESAI a1]")


if __name__ == "__main__":
    main()

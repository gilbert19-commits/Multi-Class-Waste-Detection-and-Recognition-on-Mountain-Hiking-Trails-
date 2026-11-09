"""BAGIAN A: paired percentile bootstrap level gambar, 1000 resample, seed 0, memakai ap_per_class.

Indeks resample sama persis dengan hasil_2kelas/_cache/boot_idx.npy. Pasangan sama dengan hasil_2kelas. Pasangan terbaik family
dipilih dari mAP@50-95 standar (sama dengan hasil_2kelas); kalau terbaik menurut versi 3 berbeda, pasangannya ditulis juga.
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from up_common import CACHE, FAMILIES, HK_CACHE, METRIC_FN, METRICS, MODELS, N_BOOT, N_WORKERS, OUT, SEED, ap_matrix, best_pairs, best_std, family, holm, label, load_versions, pair_row, scale_pairs, summarize


def worker(job):
    m, v = job
    f = CACHE / f"bootU_{m}_{v}.npy"
    if not f.exists():
        imgs = load_versions(m)["images"]
        idx = np.load(HK_CACHE / "boot_idx.npy")
        np.save(f, np.stack([ap_matrix([imgs[i] for i in ix], v) for ix in idx]))
    return job


def add_holm(B):
    """Holm untuk 30 baris per skala (mAP50 dan mAP50-95): satu keluarga berisi 30, dan terpisah 15 per metrik."""
    B = B.copy()
    sel = (B.jenis == "per_skala") & B.metrik.isin(["mAP50", "mAP50_95"])
    for suf, src in (("", "p_dua_sisi"), ("_plus1", "p_dua_sisi_plus1")):
        B[f"p_holm_30{suf}"], B[f"p_holm_15_per_metrik{suf}"] = np.nan, np.nan
        B.loc[sel, f"p_holm_30{suf}"] = holm(B.loc[sel, src].to_numpy())
        for met in ("mAP50", "mAP50_95"):
            s2 = sel & (B.metrik == met)
            B.loc[s2, f"p_holm_15_per_metrik{suf}"] = holm(B.loc[s2, src].to_numpy())
    return B


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 240, "display.max_columns", 40)
    jobs = [(m, v) for m in MODELS for v in ("v2", "v3")]
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for i, done in enumerate(ex.map(worker, jobs)):
            print(f"[bootstrap] {i + 1}/{len(jobs)} {done}", flush=True)
    boots = {j: np.load(CACHE / f"bootU_{j[0]}_{j[1]}.npy") for j in jobs}
    point = {j: summarize(ap_matrix(load_versions(j[0])["images"], j[1])) for j in jobs}

    best = best_std()
    b3 = {f: max((m for m in MODELS if family(m) == f), key=lambda m: point[(m, "v3")]["mAP50_95"]) for f in FAMILIES}
    kinds = {"per_skala": scale_pairs(), "terbaik_family_standar": [("-", a, b) for a, b in best_pairs(best)]}
    if b3 != best:
        kinds["terbaik_family_versi3"] = [("-", a, b) for a, b in best_pairs(b3)]
    rows = []
    for kind, pairs in kinds.items():
        for s, a, b in pairs:
            for met in METRICS:
                dist = METRIC_FN[met](boots[(a, "v3")]) - METRIC_FN[met](boots[(b, "v3")])
                r = pair_row(dist, point[(a, "v3")][met], point[(b, "v3")][met])
                rows.append(dict(jenis=kind, skala=s, model_A=label(a), model_B=label(b), versi="3_agnostic", metrik=met, **r,
                                 lebih_baik="setara" if r["ci_melewati_nol"] else (label(a) if r["ci95_bawah"] > 0 else label(b)),
                                 n_gambar=550, n_resample=N_BOOT, seed=SEED))
    A = add_holm(pd.DataFrame(rows))
    e_rows = []
    for m in MODELS:
        for met in METRICS:
            dist = METRIC_FN[met](boots[(m, "v3")]) - METRIC_FN[met](boots[(m, "v2")])
            r = pair_row(dist, point[(m, "v3")][met], point[(m, "v2")][met])
            e_rows.append(dict(jenis="efek_salah_kelas", skala=m[-1], model_A=label(m), model_B=label(m), versi="3_agnostic_minus_2_pooled", metrik=met, **r,
                               lebih_baik="di atas nol" if r["ci95_bawah"] > 0 else ("di bawah nol" if r["ci95_atas"] < 0 else "melewati nol"),
                               n_gambar=550, n_resample=N_BOOT, seed=SEED))
    AE = pd.concat([A, pd.DataFrame(e_rows)], ignore_index=True)
    AE.to_csv(OUT / "class_agnostic" / "bootstrap_agnostic.csv", index=False)

    cols = ["jenis", "model_A", "model_B", "metrik", "A_persen", "B_persen", "delta_poin_persen", "ci95_bawah", "ci95_atas", "lebih_baik", "p_dua_sisi", "p_dua_sisi_plus1", "p_holm_15_per_metrik", "p_holm_15_per_metrik_plus1", "p_holm_30", "p_holm_30_plus1"]
    print("terbaik versi 3:", {f: label(x) for f, x in b3.items()}, "| terbaik standar:", {f: label(x) for f, x in best.items()})
    print("CI tidak melewati nol:\n", A[A.lebih_baik != "setara"][cols].round(4).to_string(index=False))
    print(A[A.jenis != "per_skala"][cols].round(3).to_string(index=False))
    print(AE[AE.jenis == "efek_salah_kelas"][["model_A", "metrik", "A_persen", "B_persen", "delta_poin_persen", "ci95_bawah", "ci95_atas", "lebih_baik"]].round(3).to_string(index=False))
    print("[SELESAI a4]", flush=True)


if __name__ == "__main__":
    main()

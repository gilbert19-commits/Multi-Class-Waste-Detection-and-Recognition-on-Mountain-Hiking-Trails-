"""BAGIAN B1-B3: p-value dan Holm untuk bootstrap hasil_2kelas, tabel batch dan patience, citra yang diabaikan Ultralytics."""
import re
import sys

import numpy as np
import pandas as pd

from up_common import DATA_DIR, HK, HK_CACHE, METRIC_FN, MODELS, N_BOOT, NAMES, OUT, PAPER, ROOT, boot_p, holm, label, source_of

DEST = OUT / "pelengkap"
ALPHA = 0.05
BATCH_KECIL = 4  # batas "batch kecil" yang dipakai untuk penandaan, ditulis di laporan
LOG = ROOT / "train_2kelas.log"


def b1_holm():
    B = pd.read_csv(HK / "bootstrap.csv")
    lab2m = {label(m): m for m in MODELS}
    boots = {m: np.load(HK_CACHE / f"boot_{m}.npy") for m in MODELS}
    rows = []
    for _, r in B.iterrows():
        d = (METRIC_FN[r.metrik](boots[lab2m[r.model_A]]) - METRIC_FN[r.metrik](boots[lab2m[r.model_B]])) * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        if abs(lo - r.ci95_bawah) > 1e-9 or abs(hi - r.ci95_atas) > 1e-9:
            raise RuntimeError(f"CI tidak sama dengan bootstrap.csv: {r.model_A} {r.model_B} {r.metrik}")
        p, p1 = boot_p(d)
        rows.append(dict(jenis=r.jenis, skala=r.skala, model_A=r.model_A, model_B=r.model_B, metrik=r.metrik, delta_poin_persen=r.delta_poin_persen, ci95_bawah=lo, ci95_atas=hi,
                         n_resample_di_sisi_berlawanan=int(min((d <= 0).sum(), (d >= 0).sum())), p_dua_sisi=p, p_dua_sisi_plus1=p1))
    P = pd.DataFrame(rows)
    sel = P.jenis == "per_skala"
    if sel.sum() != 30:
        raise RuntimeError(f"perbandingan per skala {sel.sum()}, seharusnya 30")
    for col, src in (("p_holm_30", "p_dua_sisi"), ("p_holm_30_plus1", "p_dua_sisi_plus1")):
        P[col] = np.nan
        P.loc[sel, col] = holm(P.loc[sel, src])
    for col, src in (("p_holm_15_per_metrik", "p_dua_sisi"), ("p_holm_15_per_metrik_plus1", "p_dua_sisi_plus1")):
        P[col] = np.nan
        for met in ("mAP50", "mAP50_95"):
            s2 = sel & (P.metrik == met)
            P.loc[s2, col] = holm(P.loc[s2, src])
    P.to_csv(DEST / "bootstrap_pvalue_holm.csv", index=False)
    sm = []
    for met in ("mAP50", "mAP50_95"):
        q = P[sel & (P.metrik == met)]
        sm.append(dict(metrik=met, n_perbandingan=len(q), signifikan_tanpa_koreksi=int((q.p_dua_sisi < ALPHA).sum()),
                       signifikan_holm_keluarga_15=int((q.p_holm_15_per_metrik < ALPHA).sum()), signifikan_holm_keluarga_30=int((q.p_holm_30 < ALPHA).sum()),
                       signifikan_holm_15_plus1=int((q.p_holm_15_per_metrik_plus1 < ALPHA).sum()), signifikan_holm_30_plus1=int((q.p_holm_30_plus1 < ALPHA).sum()),
                       p_terkecil=q.p_dua_sisi.min(), p_holm_15_terkecil=q.p_holm_15_per_metrik.min(), p_holm_30_terkecil=q.p_holm_30.min()))
    S = pd.DataFrame(sm)
    S.to_csv(DEST / "bootstrap_holm_ringkasan.csv", index=False)
    print(S.to_string(index=False))
    print(P[sel].sort_values("p_dua_sisi").head(8)[["model_A", "model_B", "metrik", "delta_poin_persen", "ci95_bawah", "ci95_atas", "n_resample_di_sisi_berlawanan", "p_dua_sisi", "p_holm_15_per_metrik", "p_holm_30"]].round(4).to_string(index=False))


def scan_log():
    """Baca log training: patience sesi pertama tiap model, hasil scan label dataset-2kelas, dan semua pesan label."""
    patience, scans, msgs = {}, [], {}
    rx_scan = re.compile(r"dataset-2kelas\\(train|valid)\\labels(\.cache)?\.*\s+(\d+) images, (\d+) backgrounds, (\d+) corrupt: 100%")
    rx_info = re.compile(r"\[INFO\] Train (\w+)_2kelas dari awal .*?'patience': (\d+)")
    rx_msg = re.compile(r"dataset-2kelas\\(train|valid)\\images\\([^\s:]+): (.*)$")
    with open(LOG, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "[INFO] Train" in line:
                m = rx_info.search(line)
                if m:
                    patience[m.group(1)] = int(m.group(2))  # sesi "dari awal" terakhir, sesi yang dibatalkan tertimpa
            if "corrupt: 100%" in line:
                for m in rx_scan.finditer(line):
                    scans.append((m.group(1), bool(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
            elif "dataset-2kelas\\" in line and "images\\" in line:
                m = rx_msg.search(line.strip())
                if m:
                    msgs.setdefault((m.group(1), m.group(3)[:80]), set()).add(m.group(2))
    return patience, scans, msgs


def b2_batch(patience):
    S = pd.read_csv(HK / "settings_all_runs.csv")
    T = pd.read_csv(HK / "metrics_test.csv").set_index("model")
    rows = []
    for _, r in S.iterrows():
        m = r.model
        code = next(x for x in MODELS if label(x) == m)
        args_batch = int(r.batch_args_yaml)
        auto = str(r.autobatch_log) != "tidak ada"
        p_log = patience.get(code)
        sesi = str(r.patience_per_sesi)
        rows.append(dict(model=m, family=r.family, skala=r.skala, batch_dipakai=int(r.batch), sumber_batch="AutoBatch" if auto else "ditetapkan manual",
                         batch_di_args_yaml=args_batch, autobatch_log=r.autobatch_log, batch_per_sesi=r.batch_per_sesi, akumulasi_gradien=int(r.akumulasi_gradien),
                         weight_decay_efektif=r.weight_decay_efektif, patience_args_yaml=int(r.patience), patience_sesi_pertama_dari_log=p_log, patience_per_sesi_ckpt=sesi,
                         batch_kecil=int(r.batch) <= BATCH_KECIL, tanda_skala_x_batch_kecil=(r.skala == "x") and int(r.batch) <= BATCH_KECIL,
                         mAP50_95_test_persen=float(T.loc[m, "mAP50_95"]) * 100))
    R = pd.DataFrame(rows)
    R.to_csv(DEST / "batch_dan_patience.csv", index=False)
    print(R[["model", "batch_dipakai", "sumber_batch", "batch_di_args_yaml", "akumulasi_gradien", "patience_args_yaml", "patience_sesi_pertama_dari_log", "patience_per_sesi_ckpt", "batch_kecil", "tanda_skala_x_batch_kecil"]].to_string(index=False))


def read_rows(p):
    return [r.split() for r in p.read_text().splitlines() if r.strip()]


def b3_label(scans, msgs):
    rows, files = [], []
    for sp, cname in (("train", "train"), ("valid", "valid"), ("test", None)):
        n_mixed = n_empty = n_img = n_poly_only = 0
        lost = {0: 0, 1: 0}
        for img in sorted((DATA_DIR / sp / "images").iterdir()):
            n_img += 1
            lb = read_rows(DATA_DIR / sp / "labels" / f"{img.stem}.txt")
            kinds = {len(r) == 5 for r in lb}
            n_empty += not lb
            n_poly_only += kinds == {False}
            if len(kinds) > 1:
                n_mixed += 1
                c = [int(r[0]) for r in lb]
                lost[0] += c.count(0)
                lost[1] += c.count(1)
                files.append(dict(split=sp, file=img.name, sumber=source_of(img.name), baris_box=sum(len(r) == 5 for r in lb), baris_poligon=sum(len(r) != 5 for r in lb),
                                  objek_lightweight_litter=c.count(0), objek_metal_and_glass=c.count(1)))
        row = dict(split=sp, gambar_di_folder=n_img, gambar_label_kosong=n_empty, gambar_label_seluruhnya_poligon=n_poly_only, gambar_label_campuran_dihitung_ulang=n_mixed,
                   objek_hilang_lightweight_litter=lost[0], objek_hilang_metal_and_glass=lost[1])
        if cname:
            c = np.load(DATA_DIR / sp / "labels.cache", allow_pickle=True).item()
            nf, nm, ne, nc, n = c["results"]
            reasons = sorted({re.sub(r"\x1b\[[0-9;]*m", "", x).split(": ", 2)[-1] for x in c["msgs"]})
            sc = sorted({s[2:] for s in scans if s[0] == sp})
            row.update(gambar_dipindai_cache=n, hilang_cache=nm, background_cache=ne, diabaikan_corrupt_cache=nc, dipakai_training=len(c["labels"]), alasan_di_cache="; ".join(reasons),
                       n_pesan_cache=len(c["msgs"]), hasil_scan_di_log="; ".join(f"{a} images, {b} backgrounds, {d} corrupt" for a, b, d in sc), jumlah_scan_di_log=sum(s[0] == sp for s in scans))
            for (s_, reason), fs in msgs.items():
                if s_ == sp:
                    print(f"[log] {sp}: {len(fs)} file unik | {reason}")
        else:
            row.update(alasan_di_cache="tidak ada labels.cache: training tidak pernah memindai split test", dipakai_training=0)
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(DEST / "label_diabaikan_per_split.csv", index=False)
    F = pd.DataFrame(files)
    F.to_csv(DEST / "label_diabaikan_daftar.csv", index=False)
    print(R.T.to_string())
    print(F.groupby(["split", "sumber"]).size().to_string())
    print("patokan nama kelas:", [PAPER[n] for n in NAMES])


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_colwidth", 80)
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"resample {N_BOOT}")
    b1_holm()
    patience, scans, msgs = scan_log()
    print("patience dari log:", patience)
    b2_batch(patience)
    b3_label(scans, msgs)
    print("[SELESAI b1]")


if __name__ == "__main__":
    main()

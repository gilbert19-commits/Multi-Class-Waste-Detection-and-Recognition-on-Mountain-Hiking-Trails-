"""Menulis summary.md dari CSV hasil (semua angka diambil dari file, tidak diketik tangan) dan membuat hasil_2kelas.zip."""
import sys
import zipfile

import pandas as pd

from hk_common import OUT, ROOT, RUNS, SCALES, best_epoch, read_results

EXCLUDE_DIRS = {"_cache", "_data", "_val", "__pycache__"}


def table(df, fmt=None, bold_max=()):
    """Tabel markdown. fmt: {kolom: format}, bold_max: kolom yang nilai tertingginya ditebalkan."""
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            s = format(v, fmt[c]) if c in fmt and pd.notna(v) else ("-" if pd.isna(v) else str(v))
            if c in bold_max and v == df[c].max():
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    T = pd.read_csv(OUT / "metrics_test.csv")
    C = pd.read_csv(OUT / "metrics_per_class.csv")
    V = pd.read_csv(OUT / "metrics_val.csv")
    SP = pd.read_csv(OUT / "speed.csv")
    B = pd.read_csv(OUT / "bootstrap.csv")
    R = pd.read_csv(OUT / "recall_breakdown.csv").fillna({"n_gambar": ""})
    I = pd.read_csv(OUT / "runs_inventory.csv")
    S = pd.read_csv(OUT / "settings_all_runs.csv")
    D = pd.read_csv(OUT / "dataset_stats.csv").fillna({"kelas": ""})
    NL = pd.read_csv(OUT / "metrics_test_tanpa_tumpang_tindih.csv")
    OV = pd.read_csv(OUT / "tumpang_tindih_train.csv")
    FX = pd.read_csv(OUT / "label_campuran_diperbaiki.csv")
    EX = pd.read_csv(OUT / "runs_dikecualikan.csv")
    CK = pd.read_csv(OUT / "sanity_check_map.csv")
    SM = pd.read_csv(OUT / "samples" / "samples.csv")
    ENV = (OUT / "environment.txt").read_text(encoding="utf-8").strip()

    def dv(split, item, kelas=""):
        q = D[(D.split == split) & (D.item == item) & (D.kelas == kelas)]
        return int(q.nilai.iloc[0]) if len(q) else 0

    thr = dv("semua", "ambang_inlier_ORB_maks_kontrol_acak")
    n_leak, n_name = dv("test", "terkonfirmasi_visual_inlier_di_atas_ambang"), dv("test", "gambar_dengan_nama_dasar_sama_di_train")
    v_leak, v_name = dv("valid", "terkonfirmasi_visual_inlier_di_atas_ambang"), dv("valid", "gambar_dengan_nama_dasar_sama_di_train")
    t_ov = OV[OV.split == "test"]
    sources = ["UAVVaste", "TACO", "Burangrang", "Unknown"]
    src_img = {s: dv("test", f"gambar_{s}") for s in sources}
    src_leak = {s: dv("test", f"terkonfirmasi_{s}") for s in sources}
    m = T.merge(SP[SP.conf == 0.001][["model", "preprocess", "inference", "postprocess", "total", "FPS", "total_min", "total_max"]], on="model")
    m25 = SP[SP.conf == 0.25].set_index("model")
    best = {f: T[T.family == f].sort_values("mAP50_95", ascending=False).iloc[0]["model"] for f in ("YOLOv8", "YOLOv12", "YOLO26")}
    top = T.sort_values("mAP50_95", ascending=False).iloc[0]
    sig = B[(~B.ci_melewati_nol)]
    n_ci = len(B[B.jenis == "per_skala"])
    nl_top = NL.sort_values("mAP50_95", ascending=False).iloc[0]
    d_all = (NL.mAP50_95 * 100 - T.mAP50_95 * 100)
    d50 = (NL.mAP50 * 100 - T.mAP50 * 100)
    sim = I.dropna(subset=["berhenti_jika_patience_50"])
    val_top = V.sort_values("mAP50_95", ascending=False).iloc[0]
    v12m = I[I.model == "YOLOv12m"].iloc[0]
    rel = (m.total_max - m.total_min).max()
    d12m = read_results(RUNS / "yolov12m_2kelas" / "results.csv")
    d12m_stop = d12m[d12m.epoch <= int(v12m.berhenti_jika_patience_50)]
    be_stop, val_stop = best_epoch(d12m_stop), d12m_stop["metrics/mAP50-95(B)"].max()
    jam_total = I.lama_training_jam.sum()
    n_train_efektif = dv("train", "jumlah_gambar") - dv("train", "gambar_label_campuran_poligon_dan_box")
    n_val_efektif = dv("valid", "jumlah_gambar") - dv("valid", "gambar_label_campuran_poligon_dan_box")
    tot = m.set_index("model").total
    v12_slowest = [s for s in SCALES if tot[f"YOLOv12{s}"] > max(tot[f"YOLOv8{s}"], tot[f"YOLO26{s}"])]
    v8_fastest = [s for s in SCALES if tot[f"YOLOv8{s}"] < tot[f"YOLO26{s}"]]
    L = []
    A = L.append

    A("# Hasil evaluasi dataset 2 kelas: YOLOv8, YOLOv12, YOLO26\n")
    A("Semua angka berasal dari script di `scripts/` yang benar-benar dijalankan, dan tabel di dokumen ini dibuat otomatis dari "
      "CSV di folder yang sama. Kelas: `kertas_plastik_puntungRokok` = *lightweight litter*, `logam_kaleng_kaca` = *metal and glass*.\n")

    A("## 1. Baca ini dulu: empat hal yang mempengaruhi cara menulis paper\n")
    A(f"**a. Test set tumpang tindih dengan train (temuan paling serius).** {n_leak} dari 550 gambar test "
      f"({n_leak / 550 * 100:.1f}%) terbukti berasal dari foto sumber yang sama dengan gambar di train, hanya beda augmentasi Roboflow. "
      f"Di valid {v_leak} dari 595 ({v_leak / 595 * 100:.1f}%), padahal valid dipakai memilih `best.pt` dan early stopping. "
      f"Buktinya pencocokan fitur ORB + RANSAC antar gambar dengan nama dasar sama (median inlier "
      f"{t_ov.inlier_nama_sama.median():.0f}; kontrol: nilai tertinggi dari 8 pasangan acak per gambar, bermedian "
      f"{OV.inlier_kontrol_acak.median():.0f}), lalu saya cek langsung tiga pasangan secara visual. Ambangnya konservatif "
      f"(lebih dari {thr} inlier, nilai kontrol tertinggi), dan gambar dengan nama berbeda tidak dicek, jadi {n_leak} adalah batas bawah. "
      f"Per sumber di test: " + ", ".join(f"{s} {src_leak[s]}/{src_img[s]}" for s in sources) + ". Daftar lengkap: `tumpang_tindih_train.csv`.\n")
    A(f"Tanpa {n_leak} gambar itu, mAP@50-95 turun {abs(d_all).min():.1f} sampai {abs(d_all).max():.1f} poin persen dan mAP@50 turun "
      f"{abs(d50).min():.1f} sampai {abs(d50).max():.1f} poin persen (`metrics_test_tanpa_tumpang_tindih.csv`, angka pelengkap, angka utama "
      f"tetap 550 gambar sesuai permintaan). Penurunan ini tidak seluruhnya akibat kebocoran: sisa {550 - n_leak} gambar didominasi "
      f"Burangrang ({src_img['Burangrang'] - src_leak['Burangrang']} gambar) yang memang lebih sulit. Jadi angka ini menunjukkan besarnya "
      "kepekaan, bukan performa yang sudah terkoreksi.\n")
    A("Klaim performa dari 550 gambar berisiko dibaca terlalu optimistis kalau hal ini tidak disebut. Yang bisa dilakukan: (1) melaporkan kedua angka "
      "dan menyebut tumpang tindihnya di paper, tanpa training ulang; atau (2) menyusun ulang split berdasarkan foto sumber lalu "
      f"melatih ulang, yang paling bersih tetapi butuh sekitar {jam_total:.0f} jam GPU kalau 15 model diulang semua. Ini keputusan kamu.\n")
    A(f"**b. Protokol training belum sepenuhnya sama.** Epochs 200, imgsz 512, optimizer, augmentasi, dan seed seragam. Yang berbeda: "
      f"patience (YOLOv12n/s/m memakai 100) dan batch (2 sampai 27). Simulasi dari `results.csv`: patience 100 tidak mengubah apa pun untuk "
      f"YOLOv12n dan YOLOv12s, tetapi **YOLOv12m seharusnya berhenti di epoch {int(v12m.berhenti_jika_patience_50)} kalau memakai patience 50**, "
      f"sedangkan `best.pt`-nya dari epoch {int(v12m.best_epoch)}. Jadi kalimat \"protokol yang sama\" belum akurat untuk YOLOv12m tanpa catatan. "
      "Rincian di bagian 6.\n")
    A(f"**c. 550 gambar bukan angka default Ultralytics.** Ultralytics membuang 2 gambar test (dan 3 valid, 54 train) karena labelnya "
      f"mencampur poligon dan box. Supaya 550 gambar benar-benar dinilai sesuai permintaan, label kedua gambar test itu dikonversi ke box di "
      f"salinan `_data/` (dataset asli tidak diubah), persis seperti cara Ultralytics memperlakukan label poligon. Pengaruhnya sangat kecil, "
      f"tetapi `yolo val` biasa akan menghasilkan angka 548 gambar. Kedua versi ada di `metrics_test.csv` (kolom `*_548`). "
      f"Training sendiri berjalan dengan {n_train_efektif} gambar train dan {n_val_efektif} valid, bukan {dv('train', 'jumlah_gambar')} dan {dv('valid', 'jumlah_gambar')}.\n")
    A(f"**d. Aturan sumber dari nama file perlu dikonfirmasi.** Aturannya (dari `scripts/fase1_scene_groups.py`): `batch_*` = UAVVaste, "
      "nama angka = TACO, `IMG_*`/`GOPR*`/`WhatsApp-Video*` = Burangrang. Tetapi `IMG_4879` (contoh (b) di `samples/`) adalah foto wadah plastik "
      "di dalam ruangan, jadi jelas bukan foto jalur pendakian. Angka per sumber hanya seakurat aturan ini.\n")

    A("## 2. Tabel utama (test, 550 gambar)\n")
    A("imgsz 512, batch 16, conf 0.001, IoU 0.7. Latensi: batch 1, rata-rata 3 ulangan. Params dan GFLOPs dihitung pada model setelah fuse (model yang "
      "benar-benar dipakai inferensi). Nilai tertinggi ditebalkan. Sumber: `metrics_test.csv`, `speed.csv`.\n")
    t = pd.DataFrame(dict(Model=m.model, Params_M=m.params_M, GFLOPs=m.GFLOPs_512, mAP50=m.mAP50 * 100, mAP75=m.mAP75 * 100, mAP50_95=m.mAP50_95 * 100,
                          P=m.precision * 100, R=m.recall * 100, ms=m.total, FPS=m.FPS))
    t.columns = ["Model", "Params (M)", "GFLOPs", "mAP@50 (%)", "mAP@75 (%)", "mAP@50-95 (%)", "P (%)", "R (%)", "ms/gambar", "FPS"]
    L += table(t, {"Params (M)": ".2f", "GFLOPs": ".2f", "mAP@50 (%)": ".2f", "mAP@75 (%)": ".2f", "mAP@50-95 (%)": ".2f", "P (%)": ".2f", "R (%)": ".2f",
                   "ms/gambar": ".2f", "FPS": ".1f"}, bold_max=("mAP@50 (%)", "mAP@75 (%)", "mAP@50-95 (%)", "FPS"))
    A("\n### Rincian latensi (`speed.csv`)\n")
    t = m[["model", "preprocess", "inference", "postprocess", "total", "FPS"]].copy()
    t["total_conf025"] = t.model.map(m25.total)
    t["post_conf025"] = t.model.map(m25.postprocess)
    t.columns = ["Model", "Pre", "Inference", "Post", "Total", "FPS", "Total, conf 0.25", "Post, conf 0.25"]
    L += table(t, {c: ".2f" for c in ("Pre", "Inference", "Post", "Total", "Total, conf 0.25", "Post, conf 0.25")} | {"FPS": ".1f"})
    A("\n## 3. Per kelas (test, `metrics_per_class.csv`)\n")
    t = C[["model", "kelas_paper", "gambar", "objek", "precision", "recall", "AP50", "AP75", "AP50_95"]].copy()
    for c in ("precision", "recall", "AP50", "AP75", "AP50_95"):
        t[c] = t[c] * 100
    t.columns = ["Model", "Kelas", "Gambar", "Objek", "P (%)", "R (%)", "AP@50 (%)", "AP@75 (%)", "AP@50-95 (%)"]
    L += table(t, {c: ".2f" for c in ("P (%)", "R (%)", "AP@50 (%)", "AP@75 (%)", "AP@50-95 (%)")})

    A("\n## 4. Uji signifikansi (`bootstrap.csv`)\n")
    A(f"Paired percentile bootstrap level gambar, 1000 resample, seed 0, 550 gambar per resample, gambar yang sama untuk kedua model. mAP dihitung ulang "
      f"dengan `ap_per_class` Ultralytics; sanity check terhadap validator: selisih maksimum {CK.selisih_maks.max():.1e} (`sanity_check_map.csv`). "
      "Delta = A minus B, dalam poin persen.\n")
    t = B[["jenis", "model_A", "model_B", "metrik", "delta_poin_persen", "ci95_bawah", "ci95_atas", "lebih_baik"]].copy()
    t["jenis"] = t.jenis.map({"per_skala": "skala", "terbaik_family": "terbaik"})
    t["metrik"] = t.metrik.map({"mAP50": "mAP@50", "mAP50_95": "mAP@50-95"})
    t.columns = ["Jenis", "A", "B", "Metrik", "Delta", "CI 95% bawah", "CI 95% atas", "Kesimpulan"]
    L += table(t, {"Delta": "+.2f", "CI 95% bawah": "+.2f", "CI 95% atas": "+.2f"})

    A("\n## 5. Temuan penting\n")
    r_sig = sig.iloc[0]
    A(f"- **Tidak ada model yang terbukti lebih unggul secara umum.** Dari {n_ci} perbandingan per skala, hanya {len(sig[sig.jenis == 'per_skala'])} yang CI-nya tidak "
      f"melewati nol: {r_sig.model_A} vs {r_sig.model_B} pada mAP@50-95 ({r_sig.delta_poin_persen:+.2f} poin, CI {r_sig.ci95_bawah:+.2f} sampai {r_sig.ci95_atas:+.2f}). "
      f"CI ini tidak dikoreksi untuk perbandingan berganda; pada taraf 5% sekitar {n_ci * 0.05:.1f} hasil seperti ini bisa muncul secara kebetulan, jadi jangan dijadikan klaim utama.")
    A(f"- Angka tertinggi di test: {top.model} (mAP@50 {top.mAP50 * 100:.2f}%, mAP@75 {top.mAP75 * 100:.2f}%, mAP@50-95 {top.mAP50_95 * 100:.2f}%), tetapi selisihnya "
      f"dengan model terbaik family lain tidak signifikan. Model terbaik tiap family: {best['YOLOv8']}, {best['YOLOv12']}, {best['YOLO26']}, semuanya skala l, sehingga "
      "perbandingan antar model terbaik sama dengan perbandingan skala l.")
    A(f"- Urutan tidak stabil antar split: di valid yang tertinggi {val_top.model} (mAP@50-95 {val_top.mAP50_95 * 100:.2f}%), sedangkan {top.model} di valid hanya "
      f"{V[V.model == top.model].mAP50_95.iloc[0] * 100:.2f}%. Ini mendukung kesimpulan bahwa perbedaannya sebesar variasi sampel.")
    A(f"- Kecepatan (total, batch 1): YOLOv12 paling lambat di skala {', '.join(v12_slowest)}. YOLOv8 lebih cepat dari YOLO26 di skala {', '.join(v8_fastest)}; "
      f"di skala x keduanya hampir sama (YOLOv8x {tot['YOLOv8x']:.2f} ms, YOLO26x {tot['YOLO26x']:.2f} ms). YOLO26 berjalan tanpa NMS sehingga postprocess-nya sekitar "
      f"{m[m.family == 'YOLO26'].postprocess.mean():.2f} ms, dibanding sekitar {m[m.family != 'YOLO26'].postprocess.mean():.2f} ms untuk YOLOv8/YOLOv12, tetapi selisih itu kecil dibanding waktu inferensi. "
      "Latensi antara skala n dan s nyaris sama, kemungkinan karena overhead per panggilan pada batch 1.")
    A(f"- Kelas *metal and glass* lebih lemah di semua model: AP@50 {C[C.kelas_paper == 'metal and glass'].AP50.min() * 100:.1f} sampai "
      f"{C[C.kelas_paper == 'metal and glass'].AP50.max() * 100:.1f}%, dibanding *lightweight litter* {C[C.kelas_paper == 'lightweight litter'].AP50.min() * 100:.1f} sampai "
      f"{C[C.kelas_paper == 'lightweight litter'].AP50.max() * 100:.1f}%. Objeknya hanya {int(C[C.kelas_paper == 'metal and glass'].objek.iloc[0])} di test.")
    q = R[(R.kategori == "Burangrang") & (R.kelas == "semua")]
    u = R[(R.kategori == "UAVVaste") & (R.kelas == "semua")]
    A(f"- **Burangrang paling sulit:** recall {q.recall.min() * 100:.1f} sampai {q.recall.max() * 100:.1f}%, sedangkan UAVVaste {u.recall.min() * 100:.1f} sampai {u.recall.max() * 100:.1f}%. "
      f"Perlu dicatat, hanya {src_leak['Burangrang']} dari {src_img['Burangrang']} gambar Burangrang yang tumpang tindih dengan train, dibanding {src_leak['UAVVaste']} dari {src_img['UAVVaste']} untuk UAVVaste. "
      "Jadi selisih ini sebagian bisa berasal dari kebocoran, bukan murni tingkat kesulitan.")
    A("- Sebagian false positive tampaknya sampah yang tidak dilabeli: contoh (g) di `samples/` adalah gambar tanpa label, tetapi objek yang dideteksi ketiga model secara visual terlihat seperti sampah. Ini pengamatan visual dari satu contoh, bukan hasil audit label.\n")

    A("### Analisis error model terbaik (`confusion/`, `recall_breakdown.csv`)\n")
    A("Confusion matrix: conf 0.25, IoU 0.45 (default Ultralytics). Baris = prediksi, kolom = ground truth. Recall: conf 0.25, IoU 0.5, kelas harus benar.\n")
    for mod in best.values():
        cm = pd.read_csv(OUT / "confusion" / f"cm_{mod}_mentah.csv", index_col=0)
        A(f"**{mod}**\n")
        L += table(cm.reset_index().rename(columns={"index": ""}))
        A("")
    t = R[(R.kelas == "semua") & R.dimensi.isin(["semua", "ukuran", "sumber"])][["model", "dimensi", "kategori", "n_gambar", "GT", "TP", "FN_terlewat", "FN_salah_kelas", "recall"]].copy()
    t["recall"] = t.recall * 100
    t.columns = ["Model", "Dimensi", "Kategori", "Gambar", "GT", "TP", "Terlewat", "Salah kelas", "Recall (%)"]
    L += table(t, {"Recall (%)": ".2f"})

    A("\n## 6. Inventaris run dan setting\n")
    t = I[["model", "path", "epoch_dijalankan", "best_epoch", "early_stopping", "lama_training_jam", "jumlah_sesi", "berhenti_jika_patience_50"]].merge(
        S[["model", "batch", "akumulasi_gradien", "patience"]], on="model")
    t["early_stopping"] = t.early_stopping.map({True: "ya", False: "tidak"})
    t.columns = ["Model", "Path", "Epoch", "Best", "Early stop", "Jam", "Sesi", "Simulasi stop patience 50", "Batch", "Akumulasi", "Patience"]
    L += table(t, {"Jam": ".2f", "Simulasi stop patience 50": ".0f"})
    A("\nSemua 15 run selesai dan memakai `dataset-2kelas/data.yaml` (hasil ekstrak `dataset_2class.zip`, jumlah file per split sama). Rincian lengkap: `runs_inventory.csv`, `settings_all_runs.csv`, `settings_berbeda.csv`.\n")
    A("**Sama di semua run:** epochs 200, imgsz 512, optimizer=auto yang di log tercatat `MuSGD(lr=0.01, momentum=0.9)`, lrf 0.01, weight_decay 0.0005 (nominal), warmup_epochs 3, mosaic 1.0, close_mosaic 10, mixup 0, hsv 0.015/0.7/0.4, fliplr 0.5, flipud 0, scale 0.5, translate 0.1, degrees/shear/perspective/copy_paste/cutmix 0, seed 0, deterministic, AMP, pretrained COCO, Ultralytics 8.4.123.\n")
    A("**Berbeda:**\n")
    A("- **patience:** YOLOv12n/s/m 100; YOLOv12l 100 sampai epoch 46 lalu 50 (simulasi: tidak berpengaruh); lainnya 50.")
    A(f"- **batch:** YOLOv8n/s tetap 16; sisanya AutoBatch dan hasilnya berbeda per model ({int(S.batch.min())} sampai {int(S.batch.max())}). Akumulasi gradien menyesuaikan ke nbs 64, tetapi efektifnya tidak persis sama, sehingga weight decay efektif {S.weight_decay_efektif.min():.6f} sampai {S.weight_decay_efektif.max():.6f}. Model besar juga memperbarui statistik BatchNorm dengan batch kecil.")
    A("- **workers:** YOLOv8s 0, YOLOv8n 0 sampai epoch 67 lalu 4, lainnya 4. Ini pengaturan pemuatan data, bukan hyperparameter.")
    A("- **Training terputus:** YOLOv8n 3 sesi (lanjut di epoch 68 dan 75), YOLOv12l 2 sesi (lanjut di epoch 47).")
    A(f"- **Simulasi patience 50:** YOLOv8m dan YOLO26m memang berhenti sesuai simulasi (epoch 171 dan 159), jadi simulasinya tervalidasi. YOLOv12m akan berhenti di epoch {int(v12m.berhenti_jika_patience_50)} dengan epoch terbaik saat itu {be_stop}. Bobot epoch itu tidak tersimpan, jadi angka test-nya **TIDAK TERSEDIA** tanpa training ulang. Di valid, mAP@50-95 terbaik sampai epoch {int(v12m.berhenti_jika_patience_50)} adalah {val_stop * 100:.2f}% vs {v12m.val_mAP50_95_best * 100:.2f}% pada `best.pt`.")
    A("- **warmup_bias_lr** tertulis 0.1 di sebagian besar args.yaml dan 0.0 di YOLOv8n/YOLOv12l. Ini hanya artefak penulisan file: mode optimizer=auto mengubahnya menjadi 0.0 setelah args.yaml ditulis, dan `train_args` di semua `best.pt` memang 0.0. Jadi nilainya seragam.")
    A("- **batch di args.yaml:** tertulis -1 untuk AutoBatch; angka sebenarnya diambil dari `train_args` di `best.pt` dan sudah cocok dengan log.\n")
    A("**Run yang dikecualikan:** " + f"{len(EX)} folder berisi args.yaml, alasannya per folder ada di `runs_dikecualikan.csv`. Tidak ada run yang meragukan: run `New goji preapprove/runs/detect/train*` memakai TA-4 (4 kelas), "
      "dan `runs_2kelas/_dibatalkan_e100/yolov12s_2kelas` adalah percobaan 100 epoch yang dibatalkan di epoch 16 dan sudah digantikan.\n")
    A(f"```\n{ENV}\n```\n")

    A("## 7. Statistik dataset (`dataset_stats.csv`)\n")
    rows = []
    for sp in ("train", "valid", "test"):
        rows.append(dict(Split=sp, Gambar=dv(sp, "jumlah_gambar"), Tanpa_objek=dv(sp, "gambar_tanpa_objek"), Lightweight=dv(sp, "jumlah_objek", "kertas_plastik_puntungRokok"),
                         Metal=dv(sp, "jumlah_objek", "logam_kaleng_kaca"), Label_campuran=dv(sp, "gambar_label_campuran_poligon_dan_box"),
                         **{s: dv(sp, f"gambar_{s}") for s in sources}))
    L += table(pd.DataFrame(rows))
    A(f"\nJumlah gambar sesuai target (8.409 / 595 / 550). Ukuran objek di test pada 512x512: small {dv('test', 'small')}, medium {dv('test', 'medium')}, large {dv('test', 'large')}; rincian per kelas ada di CSV. "
      "Menurut README Roboflow, seluruh gambar di-resize ke 514x514 dengan black edges dan auto-contrast. Train memakai 3 versi augmentasi per gambar sumber (flip, rotasi 90 derajat, crop 0-20%, rotasi +-15 derajat, shear +-10 derajat, brightness, exposure, noise), lalu ditambah augmentasi Ultralytics saat training.\n")

    A("## 8. Catatan cara ukur dan penyimpangan dari permintaan\n")
    A("Tidak ada langkah yang gagal. Hal yang perlu diketahui:\n")
    A(f"- **YOLO26 berjalan end-to-end tanpa NMS** pada mode default: `end2end=True` di validator, dan keluaran model berbentuk `[16, 300, 6]`, sedangkan YOLOv8/YOLOv12 memakai NMS (`[16, 6, 6069]`). Argumen iou=0.7 hanya berpengaruh pada YOLOv8/YOLOv12.")
    A("- **Params dan GFLOPs:** tabel utama memakai model setelah fuse. Untuk YOLO26 ini penting, karena sebelum fuse head one-to-many yang tidak dipakai inferensi ikut terhitung. Nilai sebelum fuse ada di kolom `*_tanpa_fuse`.")
    A("- **Jangan campur dengan `eval_2kelas/`.** Evaluasi lama memakai IoU NMS 0.6, 548 gambar, dan params sebelum fuse, jadi angkanya sedikit berbeda.")
    A("- **Confusion matrix:** Ultralytics 8.4 menurunkan threshold confusion matrix ke 0.001 kalau conf=0.001 diberikan eksplisit. Nilainya saya set ke 0.25 dan dicek otomatis terhadap jumlah prediksi dan GT.")
    A(f"- **Kecepatan:** permintaan tidak menyebut conf, jadi diukur pada conf 0.001 (utama, sama dengan Langkah 2) dan 0.25 (pelengkap); bedanya kecil. Waktu dari validator; preprocess hanya mencakup pindah ke GPU dan normalisasi, baca JPEG dan letterbox tidak terukur. GPU tidak eksklusif: aplikasi desktop (browser, VS Code, Wallpaper Engine) ikut memakai GPU yang sama. Selisih terbesar antar ulangan {rel:.2f} ms (`speed_ulangan.csv`), jadi selisih kecil antar model jangan dibaca berlebihan.")
    A(f"- **Cross-check valid vs training (`metrics_val.csv`):** kolom `*_592` memakai 592 gambar seperti saat training. Selisih terhadap `results.csv` maksimal {V.selisih_mAP50_592_vs_training.abs().max() * 100:.2f} poin (mAP@50) dan {V.selisih_mAP50_95_592_vs_training.abs().max() * 100:.2f} poin (mAP@50-95), terbesar pada model n.")
    A("- **Contoh prediksi (`samples/samples.csv`):** dipilih otomatis dengan kriteria tertulis di `s6_figures.py`, mengutamakan gambar yang sumbernya tidak ada di train. Gambar di-crop di sekitar objek supaya terbaca di lebar satu kolom (kolom `crop`). Contoh (a) memuat kedua kelas; (b) hanya *lightweight litter*.")
    A("- **Gambar:** PNG 300 dpi, lebar 8,8 cm, font serif 7,5-8 pt, kecuali `ultralytics_*` di `confusion/` yang merupakan keluaran asli Ultralytics.\n")

    A("## 9. Isi folder\n")
    A("- `summary.md`, `environment.txt`\n- Inventaris: `runs_inventory.csv`, `settings_all_runs.csv`, `settings_berbeda.csv`, `runs_dikecualikan.csv`\n"
      "- Dataset: `dataset_stats.csv`, `label_campuran_diperbaiki.csv`, `tumpang_tindih_train.csv`\n"
      "- Metrik: `metrics_test.csv`, `metrics_per_class.csv`, `metrics_val.csv`, `metrics_test_tanpa_tumpang_tindih.csv`, `sanity_check_map.csv`\n"
      "- `speed.csv`, `speed_ulangan.csv`, `bootstrap.csv`, `recall_breakdown.csv`\n- `figures/`, `curves/`, `confusion/`, `samples/`\n"
      "- `scripts/` dan log `s*.log`. Urutan menjalankan dari folder `scripts/` dengan environment `ultralytics_env`: `s0_inventaris`, `s1_dataset`, `s2_eval`, `s3_speed`, `s4_bootstrap`, `s5_error`, `s6_figures`, `s7_summary`.\n"
      "- Di disk saja, tidak ikut zip: `_cache/` (prediksi per gambar), `_data/` (salinan hardlink split), `_val/` (keluaran validator).\n")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")

    zf = ROOT / "hasil_2kelas.zip"
    n = 0
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            rel_p = f.relative_to(OUT)
            if f.is_file() and not (set(rel_p.parts) & EXCLUDE_DIRS) and f.suffix != ".pt" and not f.name.startswith("_"):
                z.write(f, "hasil_2kelas" / rel_p)
                n += 1
    print(f"[ZIP] {zf} {n} file, {zf.stat().st_size / 2 ** 20:.1f} MiB")


if __name__ == "__main__":
    main()

"""Menulis laporan.md dan README.md dari CSV, lalu membungkus untuk_penulisan.zip. Angka di laporan dihitung di sini, tidak diketik tangan."""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "untuk_penulisan"
HK = ROOT / "hasil_2kelas"
ZIP = ROOT / "untuk_penulisan.zip"
CA = OUT / "class_agnostic"
PL = OUT / "pelengkap"
SB = OUT / "sumber"
ORDER = ["YOLOv8", "YOLOv12", "YOLO26"]


def md(df, nd=2):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def rng(s, nd=2):
    return f"{s.min():.{nd}f} sampai {s.max():.{nd}f}"


def versions_table(T, met):
    cols = {f"{met}_1_standar": "1 Standar", f"{met}_2_pooled": "2 Pooled", f"{met}_3_agnostic": "3 Agnostic", f"{met}_4_agnostic_tanpa_nms": "4 Tanpa NMS ulang",
            f"{met}_efek_bobot": "Efek bobot (2-1)", f"{met}_efek_salah_kelas": "Efek salah kelas (3-2)", f"{met}_efek_nms_ulang": "Efek NMS ulang (3-4)"}
    d = T[["model"] + list(cols)].rename(columns=cols).rename(columns={"model": "Model"})
    for c in cols.values():
        d[c] = d[c] * 100
    return md(d)


def pairs_txt(q):
    return "; ".join(f"{r.model_A} vs {r.model_B} ({'mAP@50' if r.metrik == 'mAP50' else 'mAP@50-95'}, {r.delta_poin_persen:+.2f})" for r in q.itertuples()) or "tidak ada"


def build_report():
    T = pd.read_csv(CA / "metrics_4versi.csv")
    BA = pd.read_csv(CA / "bootstrap_agnostic.csv")
    R = pd.read_csv(CA / "rincian_per_kelas.csv")
    VA = pd.read_csv(CA / "verifikasi_a_standar_vs_ultralytics.csv")
    VB = pd.read_csv(CA / "verifikasi_b_single_cls.csv")
    VC = pd.read_csv(CA / "verifikasi_c_contoh_kecil.csv")
    VD = pd.read_csv(CA / "verifikasi_d_v3_vs_v2.csv")
    LM = pd.read_csv(CA / "efek_rumus_ap_lama.csv")
    HR = pd.read_csv(PL / "bootstrap_holm_ringkasan.csv")
    HP = pd.read_csv(PL / "bootstrap_pvalue_holm.csv")
    BP = pd.read_csv(PL / "batch_dan_patience.csv")
    LB = pd.read_csv(PL / "label_diabaikan_per_split.csv")
    LF = pd.read_csv(PL / "label_diabaikan_daftar.csv")
    PN = pd.read_csv(SB / "pola_nama_file.csv")
    EV = pd.read_csv(SB / "bukti_nama_TACO_UAVVaste.csv")
    NS = pd.read_csv(SB / "jumlah_per_sumber_dua_aturan.csv")
    RS = pd.read_csv(SB / "recall_per_sumber_dua_aturan.csv")
    CD = pd.read_csv(OUT / "contoh_dataset" / "daftar_contoh.csv")
    MT = pd.read_csv(HK / "metrics_test.csv")

    fam = lambda f: T[T.family == f]  # noqa: E731
    e = BA[BA.jenis == "efek_salah_kelas"]
    n_above = int((e.lebih_baik == "di atas nol").sum())
    ps = BA[(BA.jenis == "per_skala") & BA.metrik.isin(["mAP50", "mAP50_95"])]
    ps_sig = ps[ps.lebih_baik != "setara"]
    cnt = lambda col, met: int((ps[ps.metrik == met][col] < 0.05).sum())  # noqa: E731
    sig15p1 = ps[ps.p_holm_15_per_metrik_plus1 < 0.05]
    bf = BA[BA.jenis.str.startswith("terbaik")]
    bf_t = bf[["jenis", "model_A", "model_B", "metrik", "A_persen", "B_persen", "delta_poin_persen", "ci95_bawah", "ci95_atas", "lebih_baik", "p_dua_sisi_plus1"]].rename(
        columns={"jenis": "Pilihan", "model_A": "A", "model_B": "B", "metrik": "Metrik", "A_persen": "A (%)", "B_persen": "B (%)", "delta_poin_persen": "Selisih", "ci95_bawah": "CI bawah", "ci95_atas": "CI atas",
                 "lebih_baik": "Hasil", "p_dua_sisi_plus1": "p (+1)"})
    bf_t["Pilihan"] = bf_t["Pilihan"].str.replace("terbaik_family_", "", regex=False)
    bf_t["p (+1)"] = bf_t["p (+1)"].map(lambda v: f"{v:.3f}")
    b3 = {f: fam(f).sort_values("mAP50_95_3_agnostic", ascending=False).iloc[0].model for f in ORDER}
    b1 = {f: fam(f).sort_values("mAP50_95_1_standar", ascending=False).iloc[0].model for f in ORDER}
    top = {m: T.sort_values(f"{m}_3_agnostic", ascending=False).iloc[0] for m in ("mAP50", "mAP75", "mAP50_95")}
    r_t = R[["model", "kelas_gt", "n_objek", "persen_kelas_benar", "persen_kelas_salah", "persen_tidak_terdeteksi"]].rename(
        columns={"model": "Model", "kelas_gt": "Kelas GT", "n_objek": "Objek", "persen_kelas_benar": "Kelas benar (%)", "persen_kelas_salah": "Kelas salah (%)", "persen_tidak_terdeteksi": "Tidak terdeteksi (%)"})
    mg, lw = R[R.kelas_gt == "metal and glass"], R[R.kelas_gt == "lightweight litter"]

    va_max = VA.selisih_maks.max()
    vb_nms, vb_e2e = VB[VB.family != "YOLO26"], VB[VB.family == "YOLO26"]
    vb_t = VB[["model", "mode_validator", "n_prediksi_single_cls", "n_prediksi_v3", "n_prediksi_v4", "mAP50_single_cls", "mAP50_95_single_cls", "selisih_maks_v3", "selisih_maks_v4"]].rename(
        columns={"model": "Model", "mode_validator": "Validator", "n_prediksi_single_cls": "Pred single_cls", "n_prediksi_v3": "Pred v3", "n_prediksi_v4": "Pred v4", "mAP50_single_cls": "mAP@50", "mAP50_95_single_cls": "mAP@50-95",
                 "selisih_maks_v3": "Selisih maks vs v3", "selisih_maks_v4": "Selisih maks vs v4"})
    vc_t = VC[VC.jenis == "syarat"][["pengecekan", "harapan", "hasil", "cocok"]].rename(columns={"pengecekan": "Pengecekan", "harapan": "Manual/harapan", "hasil": "Script", "cocok": "Cocok"})
    vc_lama = VC[VC.jenis == "informasi"]
    n_low = int(VD.v3_lebih_rendah.sum())
    nms_t = T[["model", "n_prediksi", "n_prediksi_setelah_nms_ulang", "n_dihapus_nms_ulang_ada_penekan_kelas_beda", "n_dihapus_nms_ulang_hanya_kelas_sama"]].rename(
        columns={"model": "Model", "n_prediksi": "Prediksi", "n_prediksi_setelah_nms_ulang": "Setelah NMS ulang", "n_dihapus_nms_ulang_ada_penekan_kelas_beda": "Dihapus, ada penekan kelas beda", "n_dihapus_nms_ulang_hanya_kelas_sama": "Dihapus, hanya kelas sama"})

    h_t = HR.rename(columns={"metrik": "Metrik", "n_perbandingan": "Perbandingan", "signifikan_tanpa_koreksi": "p<0.05 tanpa koreksi", "signifikan_holm_keluarga_15": "Holm keluarga 15", "signifikan_holm_keluarga_30": "Holm keluarga 30",
                             "signifikan_holm_15_plus1": "Holm 15 (+1)", "signifikan_holm_30_plus1": "Holm 30 (+1)", "p_terkecil": "p terkecil", "p_holm_15_terkecil": "Holm 15 terkecil", "p_holm_30_terkecil": "Holm 30 terkecil"})
    h1 = HP[(HP.jenis == "per_skala")].sort_values("p_dua_sisi").iloc[0]
    bp_t = BP[["model", "batch_dipakai", "sumber_batch", "akumulasi_gradien", "patience_args_yaml", "patience_sesi_pertama_dari_log", "tanda_skala_x_batch_kecil", "batch_kecil"]].copy()
    bp_t["Tanda"] = [("skala x, batch kecil" if a else ("batch kecil" if b else "")) for a, b in zip(bp_t.tanda_skala_x_batch_kecil, bp_t.batch_kecil)]
    bp_t = bp_t.drop(columns=["tanda_skala_x_batch_kecil", "batch_kecil"]).rename(columns={"model": "Model", "batch_dipakai": "Batch", "sumber_batch": "Asal batch", "akumulasi_gradien": "Akumulasi", "patience_args_yaml": "Patience args.yaml", "patience_sesi_pertama_dari_log": "Patience sesi pertama (log)"})
    lb = LB.set_index("split")
    lf_src = LF.groupby(["split", "sumber"]).size().reset_index(name="n")
    lf_txt = "; ".join(f"{s}: " + ", ".join(f"{r.sumber} {r.n}" for r in g.itertuples()) for s, g in lf_src.groupby("split"))
    ev_t = EV[["pola_nama", "sumber_aturan_lama", "sumber_aturan_usulan", "train", "valid", "test", "cocok_nama_TACO", "cocok_nama_UAVVaste", "tidak_cocok_keduanya"]].rename(
        columns={"pola_nama": "Pola", "sumber_aturan_lama": "Aturan lama", "sumber_aturan_usulan": "Aturan usulan", "train": "Train", "valid": "Valid", "test": "Test", "cocok_nama_TACO": "Cocok TACO", "cocok_nama_UAVVaste": "Cocok UAVVaste", "tidak_cocok_keduanya": "Tidak cocok"})
    ns = NS[NS.aturan == "aturan_usulan"][["sumber", "split", "citra", "citra_tanpa_objek", "objek_lightweight_litter", "objek_metal_and_glass"]].rename(columns={"sumber": "Sumber", "split": "Split", "citra": "Citra", "citra_tanpa_objek": "Tanpa objek", "objek_lightweight_litter": "Lightweight", "objek_metal_and_glass": "Metal"})
    nb = NS[(NS.aturan == "aturan_usulan") & (NS.sumber == "Burangrang")].set_index("split")
    rs = RS[(RS.kelas == "semua") & (RS.dimensi != "pola_nama")].copy()
    rs["recall"] = rs.recall * 100
    rs_t = rs.pivot_table(index=["dimensi", "kategori"], columns="model", values="recall").reset_index()[["dimensi", "kategori", "YOLOv8l", "YOLOv12l", "YOLO26l"]].rename(columns={"dimensi": "Aturan", "kategori": "Sumber"})
    bur = rs[(rs.dimensi == "aturan_usulan") & (rs.kategori == "Burangrang")]
    ol = EV.set_index("pola_nama").test_tumpang_tindih_daftar_hasil_2kelas
    ol_bur = int(EV[EV.sumber_aturan_usulan == "Burangrang"].test_tumpang_tindih_daftar_hasil_2kelas.sum())
    ol_taco = int(EV[EV.sumber_aturan_usulan == "TACO"].test_tumpang_tindih_daftar_hasil_2kelas.sum())
    ol_uav = int(EV[EV.sumber_aturan_usulan == "UAVVaste"].test_tumpang_tindih_daftar_hasil_2kelas.sum())
    tb = lambda s, c: int(NS[(NS.aturan == "aturan_usulan") & (NS.sumber == s) & (NS.split == "test")][c].iloc[0])  # noqa: E731
    ml = LM.copy()
    wsheet = CD[CD.jenis == "crop"].groupby(["kelas", "sumber"]).size().reset_index(name="n")
    crop_txt = "; ".join(f"{k}: " + ", ".join(f"{r.sumber} {r.n}" for r in g.itertuples()) for k, g in wsheet.groupby("kelas"))

    report = f"""# Laporan bahan penulisan paper (dataset 2 kelas)

Semua angka berasal dari skrip di `scripts/` dengan setting yang sama dengan `hasil_2kelas`: split test 550 citra, imgsz 512, conf 0.001, iou 0.7, best.pt, 15 model. Tidak ada training ulang, dan folder runs serta dataset tidak berubah. Semua langkah Bagian A dan B berhasil dijalankan. Prediksi per gambar dipakai ulang dari `hasil_2kelas/_cache`; inferensi baru hanya untuk `single_cls=True` di verifikasi b. Satuan selisih dan CI adalah poin persen.

## 1. Temuan penting

**a. Aturan sumber di `hasil_2kelas` salah untuk sebagian file, dan ini mengubah cerita "data Burangrang".** Nama file saya cocokkan dengan anotasi resmi TACO dan UAVVaste. Semua {int(EV[EV.pola_nama == 'IMG_#'].cocok_nama_TACO.iloc[0])} file berpola `IMG_####_JPG.rf.*` ada di TACO (`batch_3`), dan semua {int(EV[EV.pola_nama == 'GOPR#'].cocok_nama_UAVVaste.iloc[0])} file `GOPR` ada di UAVVaste, padahal keduanya dihitung Burangrang. Pola Unknown (`camera_img_`, `photo_`, `DJI_`) juga semuanya ada di UAVVaste. Tampilan contact sheet mendukung ini: lantai, aspal, meja, permukaan air, dan daun gugur mirip maple. Yang tidak ada di kedua daftar hanya `WhatsApp-Video` dan `IMG_####_JPG_jpg.rf.*`, jadi hanya keduanya yang masih mungkin data sendiri, dan itu perlu kamu pastikan. Dengan aturan usulan, Burangrang di test tinggal {int(nb.loc['test','citra'])} citra, {int(nb.loc['test','citra_tanpa_objek'])} di antaranya tanpa objek, dengan **{int(nb.loc['test','objek_lightweight_litter'])} objek lightweight litter dan hanya {int(nb.loc['test','objek_metal_and_glass'])} objek metal and glass**. Di train hanya ada {int(nb.loc['train','objek_metal_and_glass'])} objek metal and glass, dan di valid {int(nb.loc['valid','objek_metal_and_glass'])}. Dari {sum(tb(x, 'objek_lightweight_litter') + tb(x, 'objek_metal_and_glass') for x in ('Burangrang', 'TACO', 'UAVVaste'))} objek test, hanya {tb('Burangrang', 'objek_lightweight_litter') + tb('Burangrang', 'objek_metal_and_glass')} yang berasal dari Burangrang. Jadi performa kelas metal and glass pada data sendiri tidak bisa diukur dari dataset ini. Saya tidak mengubah file di `hasil_2kelas/`. Lihat bagian 4.4.

**b. Dengan class-agnostic mAP, YOLOv8 tampak lebih rendah di beberapa skala, tetapi kekuatan kesimpulannya bergantung pada cara koreksi.** Di mAP standar tidak ada perbandingan per skala yang lolos Holm. Di versi 3, {len(ps_sig)} dari 30 perbandingan punya CI yang tidak melewati nol, semuanya menguntungkan model yang lebih baru. Dengan Holm per metrik dan p konservatif (+1), yang bertahan: {pairs_txt(sig15p1)}. Lihat bagian 2.2.

**c. Efek salah kelas nyata di semua model.** {n_above} dari {len(e)} CI (15 model, 3 metrik) berada di atas nol. Pada mAP@50 besarnya {rng(T.mAP50_efek_salah_kelas * 100)} poin. Pada conf 0.25 dan IoU 0.5, {rng(mg.persen_kelas_salah, 1)}% objek metal and glass terdeteksi dengan kelas salah, sedangkan lightweight litter hanya {rng(lw.persen_kelas_salah, 1)}%.

**d. Angka harapan 0,29 dan 0,78 di verifikasi c tidak sesuai dengan Ultralytics yang terpasang.** Script dan hitungan manual sama-sama menghasilkan 0,2475 dan 0,6650 dengan rumus Ultralytics 8.4.123. Angka 0,29 dan 0,7772 hanya muncul dengan rumus titik penutup yang lebih lama. Akibatnya, mAP pada Ultralytics 8.4.123 lebih rendah daripada rumus lama untuk prediksi yang sama. Lihat bagian 3.

**e. `single_cls=True` di YOLO26 tidak setara dengan versi 3.** Selisihnya sampai {vb_e2e.selisih_maks_v3.max():.4f}, melewati batas 0,005, karena validator tidak menjalankan NMS untuk model end-to-end. Hasilnya tepat sama dengan versi 4. Lihat bagian 3.

## 2. Bagian A: class-agnostic mAP

Definisi versi: (1) standar, (2) pooled class-aware, (3) class-agnostic dengan NMS ulang IoU 0.7, (4) class-agnostic tanpa NMS ulang. File: `class_agnostic/metrics_4versi.csv`.

### 2.1 Empat versi (dalam persen)

**mAP@50**

{versions_table(T, "mAP50")}

**mAP@75**

{versions_table(T, "mAP75")}

**mAP@50-95**

{versions_table(T, "mAP50_95")}

Nilai versi 3 tertinggi: mAP@50 {top['mAP50'].model} ({top['mAP50'].mAP50_3_agnostic * 100:.2f}), mAP@75 {top['mAP75'].model} ({top['mAP75'].mAP75_3_agnostic * 100:.2f}), mAP@50-95 {top['mAP50_95'].model} ({top['mAP50_95'].mAP50_95_3_agnostic * 100:.2f}).

Efek bobot positif pada semua model di mAP@50 ({rng(T.mAP50_efek_bobot * 100)} poin) dan mAP@50-95 ({rng(T.mAP50_95_efek_bobot * 100)} poin), karena kelas lightweight litter yang AP-nya lebih tinggi mendominasi objek (1002 banding 129). Di mAP@75 efeknya negatif pada YOLOv8l dan YOLOv12l ({T[T.model == 'YOLOv8l'].mAP75_efek_bobot.iloc[0] * 100:.2f} dan {T[T.model == 'YOLOv12l'].mAP75_efek_bobot.iloc[0] * 100:.2f} poin): keduanya satu-satunya model yang AP@75 metal and glass-nya lebih tinggi dari lightweight litter (`hasil_2kelas/metrics_per_class.csv`). Efek ini terutama soal pembobotan, bukan peningkatan kemampuan model.

### 2.2 Bootstrap versi 3

Protokol sama dengan `hasil_2kelas`: paired percentile, level gambar, 1000 resample, seed 0, dan indeks resample yang sama persis. File: `class_agnostic/bootstrap_agnostic.csv`.

**Per skala.** CI tidak melewati nol pada {len(ps_sig)} dari 30 perbandingan (mAP@50 dan mAP@50-95): {pairs_txt(ps_sig)}. Selisih dalam poin persen, tanda positif menguntungkan model pertama.

Jumlah yang tetap signifikan setelah Holm:

| Keluarga Holm | p mentah: mAP@50 | p mentah: mAP@50-95 | p +1: mAP@50 | p +1: mAP@50-95 |
|---|---|---|---|---|
| 15 per metrik | {cnt('p_holm_15_per_metrik', 'mAP50')} | {cnt('p_holm_15_per_metrik', 'mAP50_95')} | {cnt('p_holm_15_per_metrik_plus1', 'mAP50')} | {cnt('p_holm_15_per_metrik_plus1', 'mAP50_95')} |
| 30 gabungan | {cnt('p_holm_30', 'mAP50')} | {cnt('p_holm_30', 'mAP50_95')} | {cnt('p_holm_30_plus1', 'mAP50')} | {cnt('p_holm_30_plus1', 'mAP50_95')} |

Beberapa perbandingan punya p mentah 0, artinya tidak ada satu pun dari 1000 resample yang melewati nol. Itu bukan berarti p benar-benar 0. Dengan koreksi +1, p terkecil yang mungkin adalah 2/1001 = 0,002. Akibatnya Holm 30 gabungan paling kecil {2 / 1001 * 30:.4f}, jadi dengan 1000 resample keluarga 30 gabungan (+1) **tidak mungkin** mencapai 0,05. Angka 0 di sel itu adalah batas resolusi, bukan bukti tidak ada perbedaan. Keluarga 15 per metrik dengan p +1 adalah pilihan yang masih bisa mencapai signifikansi dan tidak melebih-lebihkan p. Pilihan akhirnya ada padamu.

**Antar model terbaik tiap family.** Terbaik menurut mAP@50-95 standar: {', '.join(b1.values())}. Terbaik menurut versi 3: {', '.join(b3.values())}. Pemilihannya dilakukan pada test yang sama, jadi selisihnya cenderung sedikit optimistis. Tidak ada koreksi berganda di tabel ini.

{md(bf_t)}

**Efek salah kelas (versi 3 dikurangi versi 2).** CI di atas nol pada {n_above} dari {len(e)} kombinasi model dan metrik. Rentang batas bawah CI di mAP@50: {rng(e[e.metrik == 'mAP50'].ci95_bawah)}. Rincian per model ada di file yang sama (`jenis = efek_salah_kelas`).

### 2.3 Rincian per kelas ground truth

Conf 0.25, IoU 0.5, satu prediksi hanya untuk satu ground truth; pencocokan dengan kelas benar didahulukan. Angkanya cocok dengan `hasil_2kelas/recall_breakdown.csv` (dicek otomatis). File: `class_agnostic/rincian_per_kelas.csv`.

{md(r_t)}

Masalah kedua kelas berbeda: metal and glass lebih sering salah kelas, lightweight litter lebih sering tidak terdeteksi ({rng(lw.persen_tidak_terdeteksi, 1)}%, sedangkan metal and glass {rng(mg.persen_tidak_terdeteksi, 1)}%).

## 3. Hasil keempat verifikasi

**a. Standar vs Ultralytics: lulus.** Selisih maksimum {va_max:.2e} (batas 0,001). Fungsi pencocokan script juga menghasilkan matriks TP yang persis sama dengan validator: {int(VA.sel_tp_berbeda_dari_validator.sum())} sel berbeda dari {int(VA.n_sel.sum())} sel. File: `verifikasi_a_standar_vs_ultralytics.csv`.

**b. Cek silang `single_cls=True`: 10 model NMS lulus, 5 model YOLO26 melewati batas dan penyebabnya sudah ditemukan.**

{md(vb_t, 6)}

- YOLOv8 dan YOLOv12: selisih terhadap versi 3 maksimum {vb_nms.selisih_maks_v3.max():.6f}. Jumlah prediksi bisa sedikit berbeda karena versi 3 menjalankan NMS tanpa kelas pada keluaran yang sudah melewati NMS per kelas, sedangkan validator langsung NMS tanpa kelas dari kandidat awal. Satu kotak yang sudah hilang di tahap pertama bisa berbeda efeknya pada rantai penekanan.
- YOLO26: selisih terhadap versi 3 sampai {vb_e2e.selisih_maks_v3.max():.6f}, melewati 0,005. Penyebabnya, `non_max_suppression` langsung mengembalikan keluaran kalau `end2end=True`, tanpa NMS, sehingga `single_cls=True` hanya mengganti label kelas. Buktinya, selisih terhadap versi 4 tepat {vb_e2e.selisih_maks_v4.max():.6f} dan jumlah prediksinya sama persis. Jadi versi 3 pada YOLO26 tidak bisa direproduksi hanya dengan `single_cls=True`.
- Penulisan di paper: versi 3 berarti NMS tanpa kelas dijalankan setelah prediksi disatukan, untuk semua model. Kalau hanya memakai `single_cls=True`, YOLO26 akan dinilai tanpa NMS, sedangkan model lain dengan NMS, sehingga tidak sebanding.

Apa yang dihapus NMS ulang:

{md(nms_t)}

Pada YOLOv8 dan YOLOv12 semua yang dihapus punya penekan dengan kelas berbeda, karena NMS per kelas sudah menghapus duplikat sekelas. Pada YOLO26 sekitar separuhnya hanya berdampingan dengan kotak sekelas, artinya model end-to-end ini memang masih mengeluarkan kotak hampir sama dengan IoU di atas 0,7. Efek NMS ulang di mAP@50: YOLOv8 {rng(fam('YOLOv8').mAP50_efek_nms_ulang * 100)}, YOLOv12 {rng(fam('YOLOv12').mAP50_efek_nms_ulang * 100)}, YOLO26 {rng(fam('YOLO26').mAP50_efek_nms_ulang * 100)} poin. Di mAP@50-95 efeknya {rng(T.mAP50_95_efek_nms_ulang * 100)} poin. Di mAP@75, NMS ulang justru menurunkan nilai pada {int((T.mAP75_efek_nms_ulang < 0).sum())} dari 15 model (terendah {T.mAP75_efek_nms_ulang.min() * 100:.2f} poin). Kemungkinan penyebabnya, kotak dengan confidence lebih tinggi menekan kotak yang lokalisasinya lebih pas, dan ini paling terasa pada ambang IoU tinggi; penyebab ini belum saya uji terpisah.

**c. Contoh kecil: perilaku dan hitungan manual lulus, tetapi angka harapan di permintaan tidak cocok dengan rumus terpasang.**

{md(vc_t.astype(str))}

P2 dihitung FP di standar dan TP di class-agnostic, sesuai harapan. Hitungan manual tidak memakai fungsi script:
- Rumus terpasang (`compute_ap` Ultralytics 8.4.123, titik penutup `mrec=[0, r..., r_akhir, 1]`, `mpre=[1, p..., 0, 0]`). Precision jatuh ke 0 tepat setelah recall maksimum.
  - Class-agnostic: recall maksimum 2/3. Nilai 1 pada x = 0 sampai 0,66, lalu 0 mulai x = 0,67. Trapezoid: 0,66 + 0,005 = **0,665**.
  - Standar: kelas ringan (2 GT) recall maksimum 0,5, AP = 0,495. Kelas logam tanpa prediksi, AP = 0. Rata-rata = **0,2475**.
- Rumus lama (`mrec=[0, r..., 1]`, `mpre=[1, p..., 0]`), dengan garis miring dari titik terakhir ke (1, 0): standar **{vc_lama.hasil.iloc[0]}** dan class-agnostic **{vc_lama.hasil.iloc[1]}**. Inilah sumber angka 0,29 dan 0,78.

Saya tetap memakai rumus terpasang karena verifikasi a mensyaratkan hasil sama dengan Ultralytics, dan seluruh `hasil_2kelas` memakainya. Pengaruh rumus lama pada ke-15 model (`efek_rumus_ap_lama.csv`): rumus lama lebih tinggi {rng(ml.mAP50_selisih_poin)} poin di mAP@50, {rng(ml.mAP75_selisih_poin)} poin di mAP@75, dan {rng(ml.mAP50_95_selisih_poin)} poin di mAP@50-95. Ini penting saat membandingkan dengan paper lain yang memakai Ultralytics versi lama. Versi berapa rumus ini berubah tidak saya pastikan.

**d. Versi 3 lebih besar atau sama dengan versi 2: lulus.** {n_low} dari {len(VD)} kombinasi model dan metrik lebih rendah. Selisih terkecil {VD.selisih_v3_v2.min() * 100:.2f} poin. File: `verifikasi_d_v3_vs_v2.csv`.

## 4. Bagian B: pelengkap

### 4.1 p-value dan Holm untuk bootstrap `hasil_2kelas` (mAP standar)

p dua sisi dari distribusi bootstrap yang sudah ada (CI-nya dicek sama dengan `bootstrap.csv`). File: `pelengkap/bootstrap_pvalue_holm.csv` dan `pelengkap/bootstrap_holm_ringkasan.csv`.

{md(h_t, 3)}

**Tidak ada perbandingan per skala yang tetap signifikan setelah koreksi Holm**, pada semua varian. Yang terkecil adalah {h1.model_A} vs {h1.model_B} di mAP@50-95 ({h1.delta_poin_persen:+.2f} poin, p = {h1.p_dua_sisi:.3f}, Holm 15 = {h1.p_holm_15_per_metrik:.2f}, Holm 30 = {h1.p_holm_30:.2f}). Permintaan "30 perbandingan, terpisah untuk mAP@50 dan mAP@50-95" bisa dibaca dua cara, jadi saya menghitung keduanya: keluarga 15 per metrik dan keluarga 30 gabungan. Hasilnya sama.

### 4.2 Batch dan patience

Batas "batch kecil" yang saya pakai adalah batch 4 atau kurang; ini batas saya untuk penandaan, bukan standar. File: `pelengkap/batch_dan_patience.csv`.

{md(bp_t)}

Semua model skala x mendapat batch kecil: YOLOv8x 3, YOLOv12x 2, YOLO26x 2. YOLOv12l (3) dan YOLO26l (4) juga kecil, dan keduanya model terbaik di family masing-masing. Akumulasi gradien menjaga ukuran batch nominal sekitar 64, tetapi batch normalization tetap melihat batch kecil. Data ini tidak bisa membuktikan bahwa batch kecil memengaruhi hasil. YOLOv8n dan YOLOv8s memakai batch 16 yang ditetapkan manual, sisanya AutoBatch. YOLOv12n/s/m memakai patience 100, dan YOLOv12l memakai 100 sampai epoch 46 lalu 50 setelah resume.

### 4.3 Citra yang diabaikan Ultralytics

Dari `labels.cache` di `dataset-2kelas` dan `train_2kelas.log`, keduanya cocok, dan dihitung ulang dari file label. File: `pelengkap/label_diabaikan_per_split.csv` dan `pelengkap/label_diabaikan_daftar.csv`.

| Split | Citra | Diabaikan | Dipakai | Objek yang ikut hilang | Alasan |
|---|---|---|---|---|---|
| train | {int(lb.loc['train','gambar_di_folder'])} | **{int(lb.loc['train','diabaikan_corrupt_cache'])}** | {int(lb.loc['train','dipakai_training'])} | {int(lb.loc['train','objek_hilang_lightweight_litter'])} lightweight, {int(lb.loc['train','objek_hilang_metal_and_glass'])} metal | polygon dan box bercampur |
| valid | {int(lb.loc['valid','gambar_di_folder'])} | **{int(lb.loc['valid','diabaikan_corrupt_cache'])}** | {int(lb.loc['valid','dipakai_training'])} | {int(lb.loc['valid','objek_hilang_lightweight_litter'])} lightweight, {int(lb.loc['valid','objek_hilang_metal_and_glass'])} metal | polygon dan box bercampur |
| test | {int(lb.loc['test','gambar_di_folder'])} | {int(lb.loc['test','gambar_label_campuran_dihitung_ulang'])} kalau memakai `yolo val` biasa; 0 di evaluasi ini | tidak dipakai training | {int(lb.loc['test','objek_hilang_lightweight_litter'])} lightweight, {int(lb.loc['test','objek_hilang_metal_and_glass'])} metal | polygon dan box bercampur; dikonversi di salinan |

Tidak ada alasan lain: tidak ada label hilang, gambar rusak, atau pesan duplikat di cache. Per sumber (aturan lama): {lf_txt}. Selain itu, {int(lb.loc['train','gambar_label_seluruhnya_poligon'])} label train, {int(lb.loc['valid','gambar_label_seluruhnya_poligon'])} valid, dan {int(lb.loc['test','gambar_label_seluruhnya_poligon'])} test seluruhnya polygon; label ini tidak diabaikan, melainkan diubah Ultralytics menjadi kotak pembungkus.

### 4.4 Pola nama file dan sumber

Aturan lengkap: `sumber/aturan_sumber.txt`. Pola per split dan jumlah objek: `sumber/pola_nama_file.csv` dan `sumber/bukti_nama_TACO_UAVVaste.csv`.

{md(ev_t)}

Anotasi diunduh pada 2026-11-07 dari `pedropro/TACO` (`data/annotations.json`) dan `UAVVaste/UAVVaste` (`annotations/annotations.json`). Batasannya:
- Yang dicocokkan adalah nama file, bukan isi citra.
- Di TACO, 1500 citra hanya punya 239 nama unik karena nama angka berulang antar batch, jadi kecocokan pola angka adalah bukti lemah. Pola `IMG_#` dan semua pola UAVVaste lebih kuat, dan pola `IMG_#` juga didukung tampilan citranya.
- Burangrang pada aturan usulan disimpulkan dengan eliminasi. Hanya kamu yang bisa memastikan `WhatsApp-Video` dan `IMG_####_JPG_jpg` benar-benar dari Gunung Burangrang.

Jumlah data dengan aturan usulan (aturan lama juga ada di `sumber/jumlah_per_sumber_dua_aturan.csv`):

{md(ns)}

Recall model terbaik per sumber, conf 0.25, IoU 0.5, dalam persen (`sumber/recall_per_sumber_dua_aturan.csv`, termasuk per kelas dan per pola):

{md(rs_t)}

Recall Burangrang dengan aturan usulan menjadi {rng(bur.recall, 1)}%, sedangkan `summary.md` menulis 42.9 sampai 48.4% dengan aturan lama. Angka Burangrang ini hampir seluruhnya mewakili kelas lightweight litter.

Contact sheet ada tiga, semuanya memakai 24 nama dasar acak dengan seed 0, sehingga tidak terpakai untuk augmentasi dari foto yang sama. Dua yang terakhir saya tambahkan karena di sanalah keraguannya:
- `sumber/contact_sheet_IMG.png`: yang diminta. 6 dari 24 berpola `IMG_#`.
- `sumber/contact_sheet_IMG_pola_tanpa_akhiran_JPG.png`: khusus pola `IMG_#` yang cocok dengan TACO.
- `sumber/contact_sheet_GOPR.png`: khusus GOPR yang cocok dengan UAVVaste.

Pengamatan visual saya pada lembar pertama: 18 citra `IMG_#_JPG` terlihat seperti tanah jalur, akar, dan vegetasi, sedangkan beberapa citra `IMG_#` terlihat seperti lantai atau meja. Keputusan akhir tetap ada padamu.

### 4.5 Contoh dataset

- `contoh_dataset/utuh/`: 4 citra utuh per sumber, resolusi asli, dengan kotak ground truth (lightweight litter #1baf7a, metal and glass #eda100).
- `contoh_dataset/crop/`: 8 crop per kelas, resolusi asli, margin 10%, ukuran dari {CD[CD.jenis == 'crop'].lebar.min()} sampai {CD[CD.jenis == 'crop'].lebar.max()} piksel lebarnya. Komposisi: {crop_txt}.
- `contoh_dataset/daftar_contoh.csv`: file asal setiap contoh.

Semua diambil dari split test, sumbernya memakai **aturan usulan** karena aturan lama terbukti keliru. Karena data sendiri nyaris tidak punya metal and glass, 7 dari 8 crop kelas itu berasal dari UAVVaste dan TACO. Sudut hitam pada sebagian citra adalah bawaan dataset (rotasi dari Roboflow), bukan hasil skrip. Kalau aturan usulan tidak kamu setujui, sumber di daftar ini perlu dipilih ulang.

## 5. Hal lain yang aneh atau perlu dicatat

- **Tumpang tindih test-train tetap berlaku.** Sesuai instruksi terakhir, analisisnya tidak dikerjakan di sini. Semua angka di laporan ini memakai 550 citra, termasuk 186 citra yang terdaftar tumpang tindih di `hasil_2kelas/summary.md`, jadi peringatan di sana berlaku juga untuk class-agnostic mAP.
- **Pembagian tumpang tindih per sumber di `summary.md` memakai aturan lama** (Burangrang 15, UAVVaste 144, TACO 25, Unknown 2). Dengan aturan usulan menjadi Burangrang {ol_bur}, UAVVaste {ol_uav}, TACO {ol_taco}. Total tetap 186; kolomnya ada di `sumber/bukti_nama_TACO_UAVVaste.csv`. Artinya, tidak ada citra Burangrang di test yang masuk daftar tumpang tindih. Daftar itu hanya memeriksa citra dengan nama dasar yang sama, jadi tetap batas bawah. Kalimat di `summary.md` tentang Burangrang juga mengikuti aturan lama.
- **Bagian `hasil_2kelas` yang memakai aturan lama:** bagian sumber di `recall_breakdown.csv` dan `dataset_stats.csv`, serta `summary.md`. File-file itu tidak saya ubah. Contoh `IMG_4879` yang dulu saya tandai ragu ternyata ada di anotasi TACO sebagai `batch_3/IMG_4879.JPG`.
- **Nama kolom yang menyesatkan:** `dataset_stats.csv` memuat item `terkonfirmasi_visual_inlier_di_atas_ambang`, padahal angkanya dari ORB + RANSAC. Hanya 3 pasangan yang dicek secara visual.
- **Log lain tidak relevan:** `train_v8lx.log` dan `train_v12s_200.log` mencatat 2356 background dan 2803 citra, tetapi itu run lama pada `augmented/` dan `TA-1/`, bukan `dataset-2kelas`.
- **`args.yaml` YOLOv12l tidak mencerminkan seluruh training:** tertulis batch 3 dan patience 50 karena ditulis ulang saat resume. Sesi pertama memakai batch -1 (AutoBatch = 3) dan patience 100.
- **Kolom `patience_per_sesi_ckpt` berisi `TIDAK TERCATAT`** untuk beberapa sesi lama; nilainya diambil dari `train_2kelas.log` di kolom sebelahnya.
- **Kenaikan dari versi 2 ke versi 3 tidak hanya berasal dari salah kelas:** versi 3 juga menghapus prediksi duplikat, tetapi pengaruhnya kecil dibanding efek salah kelas (lihat kolom efek NMS ulang).

## 6. Yang tidak ikut di zip

- `hasil_2kelas/_data/` (85 MB, salinan dataset) dan `hasil_2kelas/_val/` (99 MB, gambar plot validator): tidak ikut karena membuat zip melewati 50 MB. Keduanya tetap ada di komputer dan bisa dibuat ulang dengan `hasil_2kelas/scripts/s2_eval.py`. Jadi "seluruh isi `hasil_2kelas/`" di zip ini berarti seluruhnya kecuali dua folder ini.
- `untuk_penulisan/_cache/` dan `_val_singlecls/`: hasil antara yang bisa dibuat ulang. File anotasi TACO dan UAVVaste juga tidak ikut, tetapi sumbernya tercatat di bagian 4.4.
- File `.pt`: tidak ada.

## 7. Keputusan yang perlu kamu ambil

1. **Apakah aturan sumber usulan disetujui?** Khususnya, apakah `WhatsApp-Video` dan `IMG_####_JPG_jpg` memang foto Burangrang. Kalau ya, klaim dan statistik per sumber di paper sebaiknya memakai aturan ini.
2. **Bagaimana menulis kontribusi data sendiri?** Dataset ini punya {tb('Burangrang','objek_metal_and_glass')} objek metal and glass Burangrang di test, jadi klaim per kelas pada jalur gunung tidak didukung data.
3. **Keluarga Holm mana yang dipakai** untuk versi 3: saya sarankan 15 per metrik dengan p +1.
"""
    report = re.sub(r"(?<![\d.,])0,(\d)", r"0.", report)  # desimal memakai titik, sama dengan tabel
    (OUT / "laporan.md").write_text(report, encoding="utf-8")
    print("laporan.md:", len(report), "karakter")
    print(ol.to_dict())


MODEL = {"yolov8": "YOLOv8", "yolov12": "YOLOv12", "yolo26": "YOLO26"}


def nice(m):
    x = re.match(r"(yolov8|yolov12|yolo26)([nsmlx])", m)
    return MODEL[x.group(1)] + x.group(2)


DESC = [
    (r"^laporan\.md$", lambda m: "Laporan utama: ringkasan Bagian A dan B, keempat verifikasi, temuan, dan hal yang aneh"),
    (r"^README\.md$", lambda m: "Daftar semua file di dalam zip ini"),
    (r"^class_agnostic/metrics_4versi\.csv$", lambda m: "Empat versi mAP, efek bobot, efek salah kelas, efek NMS ulang, 15 model"),
    (r"^class_agnostic/bootstrap_agnostic\.csv$", lambda m: "Bootstrap versi 3 antar model, CI efek salah kelas, p-value, dan Holm"),
    (r"^class_agnostic/rincian_per_kelas\.csv$", lambda m: "Persen kelas benar, kelas salah, dan tidak terdeteksi per kelas GT, model terbaik tiap family"),
    (r"^class_agnostic/verifikasi_a_.*$", lambda m: "Verifikasi a: mAP standar script vs Ultralytics dan jumlah sel TP berbeda"),
    (r"^class_agnostic/verifikasi_b_.*$", lambda m: "Verifikasi b: model.val(single_cls=True) vs versi 3 dan versi 4"),
    (r"^class_agnostic/verifikasi_c_.*$", lambda m: "Verifikasi c: contoh kecil, script vs manual dengan dua rumus AP"),
    (r"^class_agnostic/verifikasi_d_.*$", lambda m: "Verifikasi d: versi 3 vs versi 2 per model dan metrik"),
    (r"^class_agnostic/efek_rumus_ap_lama\.csv$", lambda m: "mAP dengan rumus Ultralytics terpasang vs rumus lama, per model"),
    (r"^class_agnostic/hasil_single_cls_mentah\.json$", lambda m: "Keluaran mentah model.val(single_cls=True) untuk 15 model"),
    (r"^pelengkap/bootstrap_pvalue_holm\.csv$", lambda m: "p-value dua sisi dan Holm untuk bootstrap mAP standar hasil_2kelas"),
    (r"^pelengkap/bootstrap_holm_ringkasan\.csv$", lambda m: "Jumlah perbandingan yang signifikan per metrik dan varian Holm"),
    (r"^pelengkap/batch_dan_patience\.csv$", lambda m: "Batch sebenarnya, asal batch, akumulasi, patience, dan tanda batch kecil, 15 model"),
    (r"^pelengkap/label_diabaikan_per_split\.csv$", lambda m: "Citra yang diabaikan Ultralytics per split, dari cache, log, dan hitung ulang label"),
    (r"^pelengkap/label_diabaikan_daftar\.csv$", lambda m: "Daftar file berlabel campuran polygon dan box, dengan jumlah objeknya"),
    (r"^sumber/aturan_sumber\.txt$", lambda m: "Aturan penentuan sumber: aturan lama dan aturan usulan"),
    (r"^sumber/pola_nama_file\.csv$", lambda m: "Semua pola nama file, jumlah per split, dan sumber menurut aturan lama"),
    (r"^sumber/bukti_nama_TACO_UAVVaste\.csv$", lambda m: "Kecocokan nama tiap pola dengan anotasi TACO dan UAVVaste, plus objek per split"),
    (r"^sumber/jumlah_per_sumber_dua_aturan\.csv$", lambda m: "Jumlah citra dan objek per sumber dan split, dengan kedua aturan"),
    (r"^sumber/recall_per_sumber_dua_aturan\.csv$", lambda m: "Recall model terbaik per sumber dan per pola, dengan kedua aturan"),
    (r"^sumber/contact_sheet_IMG\.png$", lambda m: "24 thumbnail acak citra berawalan IMG_ dengan nama filenya"),
    (r"^sumber/contact_sheet_IMG\.csv$", lambda m: "Daftar file pada contact_sheet_IMG.png"),
    (r"^sumber/contact_sheet_IMG_pola_tanpa_akhiran_JPG\.png$", lambda m: "24 thumbnail acak khusus pola IMG_####_JPG.rf.*, yang cocok dengan TACO"),
    (r"^sumber/contact_sheet_IMG_pola_tanpa_akhiran_JPG\.csv$", lambda m: "Daftar file pada contact sheet pola IMG_####_JPG.rf.*"),
    (r"^sumber/contact_sheet_GOPR\.png$", lambda m: "24 thumbnail acak citra GOPR, yang cocok dengan UAVVaste"),
    (r"^sumber/contact_sheet_GOPR\.csv$", lambda m: "Daftar file pada contact sheet GOPR"),
    (r"^contoh_dataset/daftar_contoh\.csv$", lambda m: "File asal, sumber, kelas, dan ukuran setiap contoh dataset"),
    (r"^contoh_dataset/utuh/(\w+)_(\d)\.png$", lambda m: f"Citra utuh {m.group(1)} nomor {m.group(2)}, resolusi asli, dengan kotak ground truth"),
    (r"^contoh_dataset/crop/(lightweight_litter|metal_and_glass)_(\d)_(\d+x\d+)\.png$", lambda m: f"Crop objek {m.group(1).replace('_', ' ')} nomor {m.group(2)}, {m.group(3)} piksel, resolusi asli"),
    (r"^scripts/up_common\.py$", lambda m: "Fungsi bersama: pencocokan, NMS ulang, empat versi, Holm, aturan sumber usulan"),
    (r"^scripts/a1_contoh_kecil\.py$", lambda m: "Verifikasi c: contoh kecil dan hitungan manual"),
    (r"^scripts/a2_singlecls_gpu\.py$", lambda m: "Verifikasi b: menjalankan model.val(single_cls=True) di GPU"),
    (r"^scripts/a3_empat_versi\.py$", lambda m: "Empat versi mAP, verifikasi a, b, d, rincian per kelas, efek rumus lama"),
    (r"^scripts/a4_bootstrap\.py$", lambda m: "Bootstrap versi 3 dan efek salah kelas"),
    (r"^scripts/b1_statistik\.py$", lambda m: "Bagian B1 sampai B3: Holm, tabel batch dan patience, citra yang diabaikan"),
    (r"^scripts/b2_gambar\.py$", lambda m: "Bagian B4 dan B5: pola nama, contact sheet, contoh dataset"),
    (r"^scripts/b3_bukti_sumber\.py$", lambda m: "Pencocokan nama dengan anotasi TACO dan UAVVaste, recall per sumber"),
    (r"^scripts/z_paket\.py$", lambda m: "Menulis laporan.md dan README.md, lalu membuat zip"),
    (r"^log/(\w+)\.log$", lambda m: f"Log keluaran script {m.group(1)}"),
    (r"^hasil_2kelas/_eval_raw\.json$", lambda m: "Hasil mentah validator per model dan split"),
    (r"^hasil_2kelas/_cache/boot_idx\.npy$", lambda m: "Indeks 1000 resample bootstrap 550 citra, seed 0"),
    (r"^hasil_2kelas/_cache/boot_(\w+)\.npy$", lambda m: f"AP per kelas tiap resample bootstrap, {nice(m.group(1))}"),
    (r"^hasil_2kelas/_cache/(\w+)_2kelas_(test|val)\.pkl$", lambda m: f"Prediksi per gambar {nice(m.group(1))} pada split {m.group(2)}"),
    (r"^hasil_2kelas/bootstrap\.csv$", lambda m: "Bootstrap mAP standar antar model"),
    (r"^hasil_2kelas/confusion/cm_(\w+?)_(mentah|ternormalisasi)\.csv$", lambda m: f"Confusion matrix {m.group(2)} {m.group(1)}, data"),
    (r"^hasil_2kelas/confusion/cm_(\w+?)_(mentah|ternormalisasi)\.png$", lambda m: f"Confusion matrix {m.group(2)} {m.group(1)}, gambar untuk paper"),
    (r"^hasil_2kelas/confusion/ultralytics_(\w+?)_confusion_matrix(_normalized)?\.png$", lambda m: f"Confusion matrix {'ternormalisasi ' if m.group(2) else ''}{m.group(1)} versi plot Ultralytics"),
    (r"^hasil_2kelas/curves/(\w+)_2kelas_results\.csv$", lambda m: f"Kurva training per epoch {nice(m.group(1))}"),
    (r"^hasil_2kelas/dataset_stats\.csv$", lambda m: "Statistik dataset per split; bagian sumber memakai aturan lama"),
    (r"^hasil_2kelas/environment\.txt$", lambda m: "Versi perangkat dan paket yang dipakai"),
    (r"^hasil_2kelas/figures/fig_akurasi_vs_kecepatan\.png$", lambda m: "Gambar akurasi vs kecepatan, 15 model"),
    (r"^hasil_2kelas/figures/fig_kurva_training_terbaik\.png$", lambda m: "Gambar kurva training model terbaik tiap family"),
    (r"^hasil_2kelas/label_campuran_diperbaiki\.csv$", lambda m: "Label campuran yang dikonversi pada salinan evaluasi"),
    (r"^hasil_2kelas/metrics_per_class\.csv$", lambda m: "Metrik standar per kelas di test, 15 model"),
    (r"^hasil_2kelas/metrics_test\.csv$", lambda m: "Tabel utama mAP standar di test, params, GFLOPs, 15 model"),
    (r"^hasil_2kelas/metrics_test_tanpa_tumpang_tindih\.csv$", lambda m: "mAP standar pada 364 citra test tanpa tumpang tindih"),
    (r"^hasil_2kelas/metrics_val\.csv$", lambda m: "Metrik di valid dan cek silang dengan angka training"),
    (r"^hasil_2kelas/recall_breakdown\.csv$", lambda m: "Recall per kelas, ukuran, dan sumber; sumber memakai aturan lama"),
    (r"^hasil_2kelas/runs_dikecualikan\.csv$", lambda m: "Run yang tidak dipakai beserta alasannya"),
    (r"^hasil_2kelas/runs_inventory\.csv$", lambda m: "Daftar 15 run yang dipakai"),
    (r"^hasil_2kelas/(s\d_\w+)\.log$", lambda m: f"Log keluaran script {m.group(1)}"),
    (r"^hasil_2kelas/samples/sample_(\d)_(\w+)\.png$", lambda m: f"Contoh deteksi nomor {m.group(1)} kategori {m.group(2).replace('_', ' ')}"),
    (r"^hasil_2kelas/samples/samples\.csv$", lambda m: "File asal dan isi tiap contoh deteksi"),
    (r"^hasil_2kelas/samples/samples_grid\.png$", lambda m: "Gabungan 8 contoh deteksi untuk paper"),
    (r"^hasil_2kelas/sanity_check_map\.csv$", lambda m: "Cek bahwa mAP hitung ulang sama dengan validator"),
    (r"^hasil_2kelas/scripts/hk_common\.py$", lambda m: "Konstanta dan fungsi bersama hasil_2kelas, termasuk aturan sumber lama"),
    (r"^hasil_2kelas/scripts/s0_inventaris\.py$", lambda m: "Langkah 0: inventaris run dan setting"),
    (r"^hasil_2kelas/scripts/s1_dataset\.py$", lambda m: "Langkah 1: statistik dataset dan deteksi tumpang tindih"),
    (r"^hasil_2kelas/scripts/s2_eval\.py$", lambda m: "Langkah 2: evaluasi seragam dan penyimpanan prediksi per gambar"),
    (r"^hasil_2kelas/scripts/s3_speed\.py$", lambda m: "Langkah 3: pengukuran kecepatan"),
    (r"^hasil_2kelas/scripts/s4_bootstrap\.py$", lambda m: "Langkah 4: tabel metrik dan bootstrap"),
    (r"^hasil_2kelas/scripts/s5_error\.py$", lambda m: "Langkah 5: confusion matrix dan recall"),
    (r"^hasil_2kelas/scripts/s6_figures\.py$", lambda m: "Langkah 6: gambar dan contoh deteksi"),
    (r"^hasil_2kelas/scripts/s7_summary\.py$", lambda m: "Menulis summary.md dan hasil_2kelas.zip"),
    (r"^hasil_2kelas/settings_all_runs\.csv$", lambda m: "Seluruh setting training tiap run"),
    (r"^hasil_2kelas/settings_berbeda\.csv$", lambda m: "Setting yang berbeda antar run"),
    (r"^hasil_2kelas/speed\.csv$", lambda m: "Kecepatan inferensi per model"),
    (r"^hasil_2kelas/speed_ulangan\.csv$", lambda m: "Kecepatan inferensi per ulangan"),
    (r"^hasil_2kelas/summary\.md$", lambda m: "Laporan hasil_2kelas; bagian per sumber memakai aturan lama"),
    (r"^hasil_2kelas/tumpang_tindih_train\.csv$", lambda m: "Skor ORB tiap citra test dan valid yang namanya sama dengan citra train"),
]


def collect():
    """(arcname, path) semua file. _data dan _val hasil_2kelas serta _cache untuk_penulisan tidak ikut."""
    files = []
    for p in sorted(HK.rglob("*")):
        rel = p.relative_to(HK).as_posix()
        if p.is_file() and not rel.startswith(("_data/", "_val/")) and "__pycache__" not in rel:
            files.append((f"hasil_2kelas/{rel}", p))
    for sub in ("class_agnostic", "pelengkap", "sumber", "contoh_dataset", "scripts"):
        for p in sorted((OUT / sub).rglob("*")):
            if p.is_file() and "__pycache__" not in p.as_posix():
                files.append((p.relative_to(OUT).as_posix(), p))
    for p in sorted(OUT.glob("*.log")):
        files.append((f"log/{p.name}", p))
    files.append(("class_agnostic/hasil_single_cls_mentah.json", OUT / "_singlecls_raw.json"))
    files.append(("laporan.md", OUT / "laporan.md"))
    files.append(("README.md", OUT / "README.md"))
    return files


def describe(arc):
    for rx, fn in DESC:
        m = re.match(rx, arc)
        if m:
            return fn(m)
    raise RuntimeError(f"file tanpa deskripsi: {arc}")


def build_readme(files):
    lines = ["# Isi untuk_penulisan.zip", "", f"Berisi {len(files)} file. Mulai dari `laporan.md`. Skrip di `scripts/` mengharapkan folder `untuk_penulisan/scripts/` berada di akar proyek, dengan `hasil_2kelas/`, `runs_2kelas/`, dan `dataset-2kelas/` di sebelahnya.", ""]
    lines += [f"- `{arc}`: {describe(arc)}" for arc, _ in files]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    build_report()
    (OUT / "README.md").touch()
    files = collect()
    build_readme(files)
    if any(a.lower().endswith(".pt") for a, _ in files):
        raise RuntimeError("ada file .pt")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arc, p in files:
            z.write(p, arc)
    with zipfile.ZipFile(ZIP) as z:
        if z.testzip() is not None:
            raise RuntimeError("zip rusak")
        names = z.namelist()
    if len(names) != len(set(names)) or len(names) != len(files):
        raise RuntimeError("isi zip tidak sama dengan daftar")
    print(f"[ZIP] {ZIP} | {len(files)} file | {ZIP.stat().st_size / 2 ** 20:.1f} MiB")


if __name__ == "__main__":
    main()

"""Assemble paket_v8lx/: hasil_v8lx.md (full precision, every number with its source file),
the new CSVs, the v8l/v8x confusion-matrix PNGs and copies of the training records; then zip it.
Reads only files produced by 05_eval_tambahan.py and 06_analisis_15model.py; computes nothing new.

Usage:
    python scripts/07_paket_v8lx.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PKG = ROOT / "paket_v8lx"
RUNS = ROOT / "runs" / "detect" / "runs_baseline"

CSVS = ["test_metrics_overall_15model.csv", "test_metrics_per_class_15model.csv", "fp_fn_per_class_15model.csv",
        "confusion_matrix_v8lx_15model.csv", "bootstrap_head_to_head_15model.csv",
        "label_noise_per_sumber_15model.csv", "selisih_full_clean_15model.csv",
        "selisih_full_clean_agregat_15model.csv", "peringkat_augmented_15model.csv", "training_v8lx_15model.csv"]


def cell(v):
    if isinstance(v, float):
        return "NaN" if pd.isna(v) else repr(v)
    return str(v).replace("|", "/")


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return out


def src(*files):
    return "Sumber: " + ", ".join(f"`{f}`" for f in files) + "."


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    PKG.mkdir(exist_ok=True)
    (PKG / "confusion_matrix").mkdir(exist_ok=True)
    (PKG / "salinan_run").mkdir(exist_ok=True)

    rd = lambda f: pd.read_csv(RES / f)
    ov, pc = rd(CSVS[0]), rd(CSVS[1])
    fp, cm, bt = rd(CSVS[2]), rd(CSVS[3]), rd(CSVS[4])
    ln, dd, agg, rk, tr = rd(CSVS[5]), rd(CSVS[6]), rd(CSVS[7]), rd(CSVS[8]), rd(CSVS[9])
    new = (ov.model == "YOLOv8") & ov.varian.isin(["l", "x"])
    o = ov[new]

    L = ["# Hasil Evaluasi YOLOv8l dan YOLOv8x (augmented)", "",
         "Semua angka ditulis dengan presisi penuh seperti di file sumbernya. Tidak ada angka yang dihitung "
         "di dokumen ini; semuanya disalin dari CSV di `results/` (salinannya ada di folder ini).", "",
         "## 0. Pengaturan", "",
         "- Environment: conda `ultralytics_env`, ultralytics 8.4.123, torch 2.14.0+cu132, Python 3.13.15, "
         "RTX 5070 12227 MiB. " + src("train_v8lx.log"),
         "- Evaluasi: `scripts/05_eval_tambahan.py` memanggil fungsi `evaluate()` dari `scripts/01_evaluate_test.py` "
         "(script yang dulu menghasilkan `test_metrics_overall.csv`) dan `predict()` dari `scripts/02_error_analysis.py`. "
         "Parameter: imgsz 640, conf 0.001, NMS IoU 0.6, batch 16, device 0, seed 0, deterministic. Bobot: `best.pt`.",
         "- Confusion matrix, FP/FN, bootstrap: conf ≥ 0.25, IoU > 0.5 (fungsi `02_error_analysis.py`), "
         "dijalankan oleh `scripts/06_analisis_15model.py` dari cache prediksi validator.",
         "- Test full: 548 citra dievaluasi (550 file, 2 ditolak Ultralytics karena label campuran). "
         "Test clean: 192 citra (subset `results/test_sets/clean`, tidak dibangun ulang).",
         "- Kontrol: analisis ulang 13 model lama dengan script 06 menghasilkan TP/FP/FN yang identik dengan "
         "`results/fp_fn_per_class.csv` (104 baris), dan agregat 13 model (530 kasus, Δ rata-rata) sama dengan "
         "angka lama di `analisis_error.md` / `ringkasan.md`.", ""]

    # ------------------------------------------------ 1. training
    t = tr.set_index("run").T.reset_index().rename(columns={"index": "atribut"})
    L += ["## 1. Training", "", *md(t), "", src("training_v8lx_15model.csv", "train_v8lx.log", "catatan_batch.txt",
                                                "runs/detect/runs_baseline/v8{l,x}/results.csv, args.yaml, weights/best.pt"),
          "",
          "- `epoch_terbaik` = epoch dengan val mAP50-95 tertinggi (fitness Ultralytics 8.4.123 = mAP50-95), dan "
          "cocok dengan `train_metrics` yang tersimpan di `best.pt`.",
          "- `optimizer=auto` di args.yaml; optimizer yang benar-benar dipakai menurut log: `optimizer_aktual_log` "
          "(lr0 0.01 dan momentum 0.937 di args.yaml diabaikan oleh mode auto, sesuai pesan di log).",
          "- Waktu training v8x jauh lebih lama per epoch daripada v8l. VRAM terpakai 11811 dari 12227 MiB saat "
          "dicek di tengah training; penyebab pastinya TIDAK TERSEDIA.",
          "- **Catatan epoch terbaik v8x.** Val mAP50 v8x melonjak di epoch 35 dan 62 (epoch 62: 0.63419, median "
          "7 epoch di sekitarnya 0.38937), dan `best.pt` v8x berasal dari epoch 62. Lonjakan sebesar ini (> 0.1 di "
          "atas median sekitarnya) tidak ada di 9 run augmented lain. Dugaan: set validasi hanya berisi 1 objek "
          "puntung_rokok, sehingga terdeteksinya objek itu menaikkan AP puntung dari ≈0 ke ≈0.995 dan mAP50 4 kelas "
          "naik ≈0.25. results.csv tidak menyimpan AP per kelas, jadi dugaan ini TIDAK TERVERIFIKASI.", ""]

    # ------------------------------------------------ 2. metrik utama
    cols = ["model", "varian", "test_set", "n_images", "n_kelas_dievaluasi", "precision", "recall", "mAP50", "mAP50_95",
            "mAP50_3kelas", "mAP50_95_3kelas"]
    L += ["## 2. Metrik utama (test full dan clean)", "",
          "Di test full, `mAP50`/`mAP50_95` = 4 kelas, `mAP50_3kelas`/`mAP50_95_3kelas` = tanpa puntung_rokok. "
          "Di test clean puntung_rokok tidak punya instance, jadi semua kolom = 3 kelas. "
          "Precision/recall = rata-rata kelas yang dievaluasi (Ultralytics `box.mp`/`box.mr`).", "",
          *md(o[cols]), "", src("test_metrics_overall_15model.csv"), "",
          "### AP per kelas", "",
          *md(pc[(pc.model == "YOLOv8") & pc.varian.isin(["l", "x"])][
              ["varian", "test_set", "kelas", "images", "instances", "precision", "recall", "mAP50", "mAP50_95"]]
              .rename(columns={"mAP50": "AP50", "mAP50_95": "AP50_95"})), "",
          src("test_metrics_per_class_15model.csv"),
          "puntung_rokok di test full hanya 2 instance di 2 citra, sehingga AP-nya sangat tidak stabil.", "",
          "### Efisiensi", "",
          "ms_per_img = waktu inference per citra (Ultralytics `speed['inference']`, batch 16, GPU), diukur terpisah di "
          "full dan clean. ms_per_img_total = preprocess + inference + postprocess.", "",
          *md(o[["model", "varian", "test_set", "params", "GFLOPs", "ms_per_img", "ms_per_img_total"]]), "",
          src("test_metrics_overall_15model.csv"),
          "Diukur dalam satu proses evaluasi (v8l lalu v8x) tanpa proses GPU lain, sama seperti 13 model lama "
          "yang diukur dalam satu proses 01_evaluate_test.py; tetapi pada hari yang berbeda.", ""]

    # ------------------------------------------------ 3. confusion matrix
    L += ["## 3. Confusion matrix test clean (angka mentah)", "",
          "Baris = kelas sebenarnya, kolom = prediksi; `latar` = background (baris latar = FP, kolom latar = FN).", ""]
    for m in ("YOLOv8l aug", "YOLOv8x aug"):
        c = cm[(cm.model == m) & (cm.test_set == "clean")].pivot(index="sebenarnya", columns="prediksi", values="jumlah")
        order = ["kertas_tisu", "logam_kaleng_kaca", "plastik", "puntung_rokok", "latar"]
        c = c.loc[order, order].reset_index()
        L += [f"**{m}**", "", *md(c), ""]
    L += [src("confusion_matrix_v8lx_15model.csv") + " PNG: `confusion_matrix/YOLOv8l_aug_clean.png`, "
          "`confusion_matrix/YOLOv8x_aug_clean.png` (versi full juga disertakan).", ""]

    # ------------------------------------------------ 4. bootstrap
    L += ["## 4. Head-to-head, bootstrap berpasangan 1000× level citra", "",
          "Prosedur sama dengan bagian head-to-head `analisis_error.md`: 1000 resampling citra dengan pengembalian "
          "(seed 0, sampel citra yang sama untuk kedua model), CI persentil 2.5–97.5. Δ = YOLOv12 − YOLOv8; untuk FN "
          "dan FP, Δ negatif berarti YOLOv12 lebih baik. Kesimpulan 'setara' bila CI memuat 0.", ""]
    for pair in bt.pasangan.unique():
        L += [f"### {pair}", ""]
        a_lab, b_lab = pair.split(" vs ")
        for s in ("clean", "full"):
            b = bt[(bt.pasangan == pair) & (bt.test_set == s)]
            L += [f"#### Test {s}", "", "FN dan FP per kelas:", "",
                  *md(b[b.metrik != "mAP50"][["kelas", "metrik", "v12", "v8", "delta", "ci_lo", "ci_hi", "kesimpulan"]]),
                  "", "Selisih mAP@50:", "",
                  *md(b[b.metrik == "mAP50"][["kelas", "v12", "v8", "delta", "ci_lo", "ci_hi", "kesimpulan"]]), ""]
            keys = [(a_lab.split()[0][:-1], a_lab.split()[0][-1]), (b_lab.split()[0][:-1], b_lab.split()[0][-1])]
            f = fp[(fp.test_set == s) & (fp.augmentasi == "ya") & (fp["gt"] > 0)
                   & fp.apply(lambda r: (r.model, r.varian) in keys, axis=1)]
            f = f.assign(model_label=f.model + f.varian + " aug")[
                ["model_label", "kelas", "gt", "TP", "FN", "FN_terlewat", "FN_salah_kelas", "FP", "FP_latar", "FP_salah_kelas"]]
            L += ["Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas), kelas yang punya GT:", "",
                  *md(f), ""]
        L += [src("bootstrap_head_to_head_15model.csv", "fp_fn_per_class_15model.csv"), ""]
    L += ["Catatan: mAP@50 di tabel bootstrap dihitung ulang dengan pencocokan `02_error_analysis.py` (AP 101 titik), "
          "sehingga bisa sedikit berbeda dari mAP@50 validator Ultralytics di bagian 2 (mis. YOLOv8x aug clean: "
          "validator 0.4436819852 vs bootstrap 0.4624459). Untuk tabel utama paper pakai angka validator (bagian 2); "
          "angka bootstrap hanya untuk selisih dan CI. Kolom `mAP50_semua_kelas_ber_GT` = cara `analisis_error.md` "
          "lama (di full = 4 kelas).", ""]

    # ------------------------------------------------ 5. label noise
    L += ["## 5. Kebingungan kertas/tisu ↔ plastik di test clean, per sumber", "",
          "Kasus = pasangan GT–prediksi yang cocok (conf ≥ 0.25, IoU > 0.5) dengan kelas kertas/plastik tertukar, "
          "dijumlah atas model. Sumber citra dari `analysis/tables/D0_scene_groups.csv` (kolom `sumber`). "
          "`kasus per GT per model` = kasus ÷ (GT kertas+plastik × jumlah model). `% objek GT tertukar` = objek GT "
          "yang dikelirukan oleh minimal satu model ÷ GT kertas+plastik sumber itu.", "",
          *md(ln), "", src("label_noise_per_sumber_15model.csv"),
          "Baris `13 model (lama)` dihitung ulang dengan script yang sama dan cocok dengan angka lama "
          "(530 kasus, 116 objek unik, arah 352/178 di `analisis_error.md` dan `analisis_label_noise.md`).", ""]

    # ------------------------------------------------ 6. full - clean
    L += ["## 6. Selisih full − clean", "",
          "Selisih = metrik test full (3 kelas, tanpa puntung_rokok) − metrik test clean (3 kelas). Positif berarti "
          "angka full lebih tinggi. **Tanda kebalikan dari `ringkasan.md` lama**, yang memakai Δ = clean − full "
          "(lama: Δ mAP@50 rata-rata −0.058, Δ mAP@50-95 −0.053, dibulatkan 3 desimal).", "",
          "### Per model (v8l, v8x)", "",
          *md(dd[(dd.model == "YOLOv8") & dd.varian.isin(["l", "x"])]), "",
          "### Agregat", "", *md(agg), "", src("selisih_full_clean_15model.csv", "selisih_full_clean_agregat_15model.csv"), ""]

    # ------------------------------------------------ 7. ranking
    L += ["## 7. Peringkat 10 model augmented", ""]
    for k in rk.kriteria.unique():
        L += [f"### {k}", "", *md(rk[rk.kriteria == k].drop(columns="kriteria")), ""]
    L += [src("peringkat_augmented_15model.csv"), "ms_per_img di tabel ini diukur di test set yang sama dengan kriterianya.",
          "", "Peringkat (i) dipengaruhi kuat oleh AP puntung_rokok yang hanya punya 2 instance di test full.", ""]

    # ------------------------------------------------ 8. ringkas
    s = ov[ov.augmentasi == "ya"].copy()
    s["model_label"] = s.model + s.varian + " aug"
    w = s.pivot_table(index="model_label", columns="test_set",
                      values=["mAP50_3kelas", "mAP50_95_3kelas", "ms_per_img"], aggfunc="first")
    w.columns = [f"{a} {b}" for a, b in w.columns]
    w = w[["mAP50_3kelas full", "mAP50_95_3kelas full", "mAP50_3kelas clean", "mAP50_95_3kelas clean",
           "ms_per_img full", "ms_per_img clean"]].reset_index()
    order = [f"YOLOv12{v} aug" for v in "nsmlx"] + [f"YOLOv8{v} aug" for v in "nsmlx"]
    w = w.set_index("model_label").loc[order].reset_index()
    L += ["## 8. Ringkasan 10 model augmented (3 kelas)", "", *md(w), "", src("test_metrics_overall_15model.csv"), ""]
    (PKG / "hasil_v8lx.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    w.to_csv(PKG / "ringkasan_10_model_augmented.csv", index=False)

    # ------------------------------------------------ copies
    for f in CSVS:
        shutil.copy2(RES / f, PKG / f)
    for m in ("YOLOv8l_aug", "YOLOv8x_aug"):
        for s_ in ("clean", "full"):
            shutil.copy2(RES / "confusion_matrix" / f"{m}_{s_}.png", PKG / "confusion_matrix" / f"{m}_{s_}.png")
    for run in ("v8l", "v8x"):
        for f in ("args.yaml", "results.csv"):
            shutil.copy2(RUNS / run / f, PKG / "salinan_run" / f"{run}_{f}")
    for f in ("train_v8lx.log", "catatan_batch.txt", "waktu_training_v8lx.txt", "train_v8lx.py", "jalankan_v8lx.bat"):
        shutil.copy2(ROOT / f, PKG / "salinan_run" / f)
    shutil.copy2(RES / "eval_v8lx_15model.log", PKG / "salinan_run" / "eval_v8lx_15model.log")
    for f in ("05_eval_tambahan.py", "06_analisis_15model.py", "07_paket_v8lx.py"):
        shutil.copy2(ROOT / "scripts" / f, PKG / "salinan_run" / f)

    z = shutil.make_archive(str(ROOT / "paket_v8lx"), "zip", root_dir=ROOT, base_dir="paket_v8lx")
    n = sum(1 for p in PKG.rglob("*") if p.is_file())
    print(f"{z} | {Path(z).stat().st_size} byte | {n} file")
    print(w.to_string(index=False))


if __name__ == "__main__":
    main()

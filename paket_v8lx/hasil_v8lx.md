# Hasil Evaluasi YOLOv8l dan YOLOv8x (augmented)

Semua angka ditulis dengan presisi penuh seperti di file sumbernya. Tidak ada angka yang dihitung di dokumen ini; semuanya disalin dari CSV di `results/` (salinannya ada di folder ini).

## 0. Pengaturan

- Environment: conda `ultralytics_env`, ultralytics 8.4.123, torch 2.14.0+cu132, Python 3.13.15, RTX 5070 12227 MiB. Sumber: `train_v8lx.log`.
- Evaluasi: `scripts/05_eval_tambahan.py` memanggil fungsi `evaluate()` dari `scripts/01_evaluate_test.py` (script yang dulu menghasilkan `test_metrics_overall.csv`) dan `predict()` dari `scripts/02_error_analysis.py`. Parameter: imgsz 640, conf 0.001, NMS IoU 0.6, batch 16, device 0, seed 0, deterministic. Bobot: `best.pt`.
- Confusion matrix, FP/FN, bootstrap: conf ≥ 0.25, IoU > 0.5 (fungsi `02_error_analysis.py`), dijalankan oleh `scripts/06_analisis_15model.py` dari cache prediksi validator.
- Test full: 548 citra dievaluasi (550 file, 2 ditolak Ultralytics karena label campuran). Test clean: 192 citra (subset `results/test_sets/clean`, tidak dibangun ulang).
- Kontrol: analisis ulang 13 model lama dengan script 06 menghasilkan TP/FP/FN yang identik dengan `results/fp_fn_per_class.csv` (104 baris), dan agregat 13 model (530 kasus, Δ rata-rata) sama dengan angka lama di `analisis_error.md` / `ringkasan.md`.

## 1. Training

| atribut | v8l | v8x |
|---|---|---|
| epochs_selesai | 100 | 100 |
| lama_training_jam_log | 6.282 | 27.942 |
| waktu_results_csv_detik | 22614.2 | 100592.0 |
| epoch_terbaik | 42 | 62 |
| val_mAP50_95_epoch_terbaik | 0.27776 | 0.2838 |
| val_mAP50_epoch_terbaik | 0.45508 | 0.63419 |
| best_pt_train_metrics_mAP50_95 | 0.27776 | 0.2838 |
| batch_args_yaml | 16 | 16 |
| batch_train_args_ckpt | 16 | 16 |
| catatan_batch | yolov8l.pt / run=v8l / batch=16 / batch=16 berjalan tanpa CUDA OOM | yolov8x.pt / run=v8x / batch=16 / batch=16 berjalan tanpa CUDA OOM |
| optimizer_args_yaml | auto | auto |
| lr0_args_yaml | 0.01 | 0.01 |
| momentum_args_yaml | 0.937 | 0.937 |
| optimizer_aktual_log | MuSGD(lr=0.01, momentum=0.9) | MuSGD(lr=0.01, momentum=0.9) |

Sumber: `training_v8lx_15model.csv`, `train_v8lx.log`, `catatan_batch.txt`, `runs/detect/runs_baseline/v8{l,x}/results.csv, args.yaml, weights/best.pt`.

- `epoch_terbaik` = epoch dengan val mAP50-95 tertinggi (fitness Ultralytics 8.4.123 = mAP50-95), dan cocok dengan `train_metrics` yang tersimpan di `best.pt`.
- `optimizer=auto` di args.yaml; optimizer yang benar-benar dipakai menurut log: `optimizer_aktual_log` (lr0 0.01 dan momentum 0.937 di args.yaml diabaikan oleh mode auto, sesuai pesan di log).
- Waktu training v8x jauh lebih lama per epoch daripada v8l. VRAM terpakai 11811 dari 12227 MiB saat dicek di tengah training; penyebab pastinya TIDAK TERSEDIA.
- **Catatan epoch terbaik v8x.** Val mAP50 v8x melonjak di epoch 35 dan 62 (epoch 62: 0.63419, median 7 epoch di sekitarnya 0.38937), dan `best.pt` v8x berasal dari epoch 62. Lonjakan sebesar ini (> 0.1 di atas median sekitarnya) tidak ada di 9 run augmented lain. Dugaan: set validasi hanya berisi 1 objek puntung_rokok, sehingga terdeteksinya objek itu menaikkan AP puntung dari ≈0 ke ≈0.995 dan mAP50 4 kelas naik ≈0.25. results.csv tidak menyimpan AP per kelas, jadi dugaan ini TIDAK TERVERIFIKASI.

## 2. Metrik utama (test full dan clean)

Di test full, `mAP50`/`mAP50_95` = 4 kelas, `mAP50_3kelas`/`mAP50_95_3kelas` = tanpa puntung_rokok. Di test clean puntung_rokok tidak punya instance, jadi semua kolom = 3 kelas. Precision/recall = rata-rata kelas yang dievaluasi (Ultralytics `box.mp`/`box.mr`).

| model | varian | test_set | n_images | n_kelas_dievaluasi | precision | recall | mAP50 | mAP50_95 | mAP50_3kelas | mAP50_95_3kelas |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8 | l | full | 548 | 4 | 0.5133670157457033 | 0.5606507855679759 | 0.6039480527898576 | 0.3554099402419191 | 0.4735974037198102 | 0.2752132536558921 |
| YOLOv8 | l | clean | 192 | 3 | 0.44209534842941 | 0.5298131424077238 | 0.3964327510368735 | 0.202896223259155 | 0.3964327510368735 | 0.202896223259155 |
| YOLOv8 | x | full | 548 | 4 | 0.5385119406274707 | 0.5762384906226395 | 0.6177565315149329 | 0.3306145230339821 | 0.4920087086865772 | 0.2916526973786428 |
| YOLOv8 | x | clean | 192 | 3 | 0.5577448233556835 | 0.43888251046397 | 0.4436819610216208 | 0.2304425316927352 | 0.4436819610216208 | 0.2304425316927352 |

Sumber: `test_metrics_overall_15model.csv`.

### AP per kelas

| varian | test_set | kelas | images | instances | precision | recall | AP50 | AP50_95 |
|---|---|---|---|---|---|---|---|---|
| l | full | kertas_tisu | 108 | 233 | 0.4907727015614363 | 0.3090128755364807 | 0.3571593346826366 | 0.2012447232628522 |
| l | full | logam_kaleng_kaca | 81 | 129 | 0.5392910860575763 | 0.4418604651162791 | 0.4721327839105844 | 0.302858342180419 |
| l | full | plastik | 264 | 763 | 0.6856665162508506 | 0.4917298016191439 | 0.5915000925662096 | 0.3215366955244052 |
| l | full | puntung_rokok | 2 | 2 | 0.33773775911295 | 1.0 | 0.995 | 0.596 |
| l | clean | kertas_tisu | 37 | 75 | 0.4205408762714277 | 0.4666666666666667 | 0.3673960829830609 | 0.1710785813996505 |
| l | clean | logam_kaleng_kaca | 26 | 34 | 0.3883474883861559 | 0.5294117647058824 | 0.3314501100003208 | 0.200662136154032 |
| l | clean | plastik | 105 | 241 | 0.5173976806306465 | 0.5933609958506224 | 0.4904520601272388 | 0.2369479522237823 |
| l | clean | puntung_rokok | 0 | 0 | NaN | NaN | NaN | NaN |
| x | full | kertas_tisu | 108 | 233 | 0.4637596559745476 | 0.2618025751072961 | 0.326820516506999 | 0.1827480394379591 |
| x | full | logam_kaleng_kaca | 81 | 129 | 0.6560023189347532 | 0.5116279069767442 | 0.5349515885712214 | 0.344328408762079 |
| x | full | plastik | 264 | 763 | 0.7299982576075117 | 0.5315234804065179 | 0.6142540209815114 | 0.3478816439358903 |
| x | full | puntung_rokok | 2 | 2 | 0.3042875299930702 | 1.0 | 0.995 | 0.4474999999999999 |
| x | clean | kertas_tisu | 37 | 75 | 0.5090341684155328 | 0.3041743354319728 | 0.329583701531946 | 0.1662610712710325 |
| x | clean | logam_kaleng_kaca | 26 | 34 | 0.5486754792209065 | 0.4411764705882353 | 0.4545643189037401 | 0.2326348236644488 |
| x | clean | plastik | 105 | 241 | 0.6155248224306112 | 0.5712967253717018 | 0.5468978626291763 | 0.2924317001427243 |
| x | clean | puntung_rokok | 0 | 0 | NaN | NaN | NaN | NaN |

Sumber: `test_metrics_per_class_15model.csv`.
puntung_rokok di test full hanya 2 instance di 2 citra, sehingga AP-nya sangat tidak stabil.

### Efisiensi

ms_per_img = waktu inference per citra (Ultralytics `speed['inference']`, batch 16, GPU), diukur terpisah di full dan clean. ms_per_img_total = preprocess + inference + postprocess.

| model | varian | test_set | params | GFLOPs | ms_per_img | ms_per_img_total |
|---|---|---|---|---|---|---|
| YOLOv8 | l | full | 43632924 | 165.4077952 | 10.798456935453345 | 12.122603469395695 |
| YOLOv8 | l | clean | 43632924 | 165.4077952 | 10.80159687262494 | 12.316034374331744 |
| YOLOv8 | x | full | 68156460 | 258.1284352 | 17.423034488413837 | 18.693656020332128 |
| YOLOv8 | x | clean | 68156460 | 258.1284352 | 17.377001562029665 | 18.784596873350285 |

Sumber: `test_metrics_overall_15model.csv`.
Diukur dalam satu proses evaluasi (v8l lalu v8x) tanpa proses GPU lain, sama seperti 13 model lama yang diukur dalam satu proses 01_evaluate_test.py; tetapi pada hari yang berbeda.

## 3. Confusion matrix test clean (angka mentah)

Baris = kelas sebenarnya, kolom = prediksi; `latar` = background (baris latar = FP, kolom latar = FN).

**YOLOv8l aug**

| sebenarnya | kertas_tisu | logam_kaleng_kaca | plastik | puntung_rokok | latar |
|---|---|---|---|---|---|
| kertas_tisu | 27 | 1 | 23 | 0 | 24 |
| logam_kaleng_kaca | 2 | 14 | 14 | 0 | 4 |
| plastik | 20 | 13 | 117 | 0 | 91 |
| puntung_rokok | 0 | 0 | 0 | 0 | 0 |
| latar | 24 | 11 | 94 | 5 | 0 |

**YOLOv8x aug**

| sebenarnya | kertas_tisu | logam_kaleng_kaca | plastik | puntung_rokok | latar |
|---|---|---|---|---|---|
| kertas_tisu | 19 | 0 | 27 | 0 | 29 |
| logam_kaleng_kaca | 0 | 14 | 17 | 0 | 3 |
| plastik | 12 | 8 | 137 | 0 | 84 |
| puntung_rokok | 0 | 0 | 0 | 0 | 0 |
| latar | 19 | 7 | 60 | 4 | 0 |

Sumber: `confusion_matrix_v8lx_15model.csv`. PNG: `confusion_matrix/YOLOv8l_aug_clean.png`, `confusion_matrix/YOLOv8x_aug_clean.png` (versi full juga disertakan).

## 4. Head-to-head, bootstrap berpasangan 1000× level citra

Prosedur sama dengan bagian head-to-head `analisis_error.md`: 1000 resampling citra dengan pengembalian (seed 0, sampel citra yang sama untuk kedua model), CI persentil 2.5–97.5. Δ = YOLOv12 − YOLOv8; untuk FN dan FP, Δ negatif berarti YOLOv12 lebih baik. Kesimpulan 'setara' bila CI memuat 0.

### YOLOv12l aug vs YOLOv8l aug

#### Test clean

FN dan FP per kelas:

| kelas | metrik | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_tisu | FN | 55.0 | 48.0 | 7.0 | 0.0 | 14.0 | setara |
| kertas_tisu | FP | 42.0 | 46.0 | -4.0 | -17.025 | 9.0 | setara |
| logam_kaleng_kaca | FN | 21.0 | 20.0 | 1.0 | -2.0249999999999986 | 4.0 | setara |
| logam_kaleng_kaca | FP | 19.0 | 25.0 | -6.0 | -17.0 | 4.024999999999977 | setara |
| plastik | FN | 106.0 | 124.0 | -18.0 | -34.0 | -2.0 | YOLOv12 lebih baik |
| plastik | FP | 108.0 | 131.0 | -23.0 | -47.025 | 1.0 | setara |
| puntung_rokok | FP | 0.0 | 5.0 | -5.0 | -10.0 | -1.0 | YOLOv12 lebih baik |
| TOTAL | FN | 182.0 | 192.0 | -10.0 | -29.0 | 8.0 | setara |
| TOTAL | FP | 169.0 | 207.0 | -38.0 | -73.0 | -3.0 | YOLOv12 lebih baik |

Selisih mAP@50:

| kelas | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50_3kelas (3 kelas) | 0.418242555040083 | 0.4044157747809698 | 0.0138267802591132 | -0.0367944184989279 | 0.0539170512351647 | setara |
| mAP50_semua_kelas_ber_GT (3 kelas) | 0.418242555040083 | 0.4044157747809698 | 0.0138267802591132 | -0.0367944184989279 | 0.0539170512351647 | setara |

Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas), kelas yang punya GT:

| model_label | kelas | gt | TP | FN | FN_terlewat | FN_salah_kelas | FP | FP_latar | FP_salah_kelas |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv12l aug | kertas_tisu | 75 | 20 | 55 | 29 | 26 | 42 | 30 | 12 |
| YOLOv12l aug | logam_kaleng_kaca | 34 | 13 | 21 | 3 | 18 | 19 | 11 | 8 |
| YOLOv12l aug | plastik | 241 | 135 | 106 | 87 | 19 | 108 | 65 | 43 |
| YOLOv8l aug | kertas_tisu | 75 | 27 | 48 | 24 | 24 | 46 | 24 | 22 |
| YOLOv8l aug | logam_kaleng_kaca | 34 | 14 | 20 | 4 | 16 | 25 | 11 | 14 |
| YOLOv8l aug | plastik | 241 | 117 | 124 | 91 | 33 | 131 | 94 | 37 |

#### Test full

FN dan FP per kelas:

| kelas | metrik | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_tisu | FN | 161.0 | 145.0 | 16.0 | 5.0 | 28.024999999999977 | YOLOv12 lebih buruk |
| kertas_tisu | FP | 177.0 | 169.0 | 8.0 | -17.0 | 33.02499999999998 | setara |
| logam_kaleng_kaca | FN | 67.0 | 65.0 | 2.0 | -6.024999999999999 | 11.0 | setara |
| logam_kaleng_kaca | FP | 60.0 | 90.0 | -30.0 | -50.0 | -11.0 | YOLOv12 lebih baik |
| plastik | FN | 300.0 | 325.0 | -25.0 | -51.025 | -1.0 | YOLOv12 lebih baik |
| plastik | FP | 306.0 | 356.0 | -50.0 | -83.0 | -15.0 | YOLOv12 lebih baik |
| puntung_rokok | FN | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | setara |
| puntung_rokok | FP | 3.0 | 8.0 | -5.0 | -12.0 | 2.0 | setara |
| TOTAL | FN | 528.0 | 535.0 | -7.0 | -35.0 | 22.024999999999977 | setara |
| TOTAL | FP | 546.0 | 623.0 | -77.0 | -125.0 | -26.950000000000045 | YOLOv12 lebih baik |

Selisih mAP@50:

| kelas | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50_3kelas (3 kelas) | 0.5092589006812243 | 0.4812278412519755 | 0.0280310594292487 | 0.0002836304679626 | 0.0544828254545759 | YOLOv12 lebih baik |
| mAP50_semua_kelas_ber_GT (4 kelas) | 0.6306941755109182 | 0.6096708809389816 | 0.0210232945719366 | 0.000212722850972 | 0.0452488949826869 | YOLOv12 lebih baik |

Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas), kelas yang punya GT:

| model_label | kelas | gt | TP | FN | FN_terlewat | FN_salah_kelas | FP | FP_latar | FP_salah_kelas |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv12l aug | kertas_tisu | 233 | 72 | 161 | 66 | 95 | 177 | 92 | 85 |
| YOLOv12l aug | logam_kaleng_kaca | 129 | 62 | 67 | 26 | 41 | 60 | 38 | 22 |
| YOLOv12l aug | plastik | 763 | 463 | 300 | 198 | 102 | 306 | 176 | 130 |
| YOLOv12l aug | puntung_rokok | 2 | 2 | 0 | 0 | 0 | 3 | 2 | 1 |
| YOLOv8l aug | kertas_tisu | 233 | 88 | 145 | 54 | 91 | 169 | 89 | 80 |
| YOLOv8l aug | logam_kaleng_kaca | 129 | 64 | 65 | 24 | 41 | 90 | 43 | 47 |
| YOLOv8l aug | plastik | 763 | 438 | 325 | 205 | 120 | 356 | 231 | 125 |
| YOLOv8l aug | puntung_rokok | 2 | 2 | 0 | 0 | 0 | 8 | 8 | 0 |

Sumber: `bootstrap_head_to_head_15model.csv`, `fp_fn_per_class_15model.csv`.

### YOLOv12x aug vs YOLOv8x aug

#### Test clean

FN dan FP per kelas:

| kelas | metrik | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_tisu | FN | 57.0 | 56.0 | 1.0 | -4.0 | 6.0 | setara |
| kertas_tisu | FP | 44.0 | 31.0 | 13.0 | 0.0 | 28.0 | setara |
| logam_kaleng_kaca | FN | 22.0 | 20.0 | 2.0 | -2.0 | 6.0 | setara |
| logam_kaleng_kaca | FP | 14.0 | 15.0 | -1.0 | -8.0 | 6.024999999999977 | setara |
| plastik | FN | 139.0 | 104.0 | 35.0 | 17.0 | 53.0 | YOLOv12 lebih buruk |
| plastik | FP | 105.0 | 104.0 | 1.0 | -24.0 | 26.024999999999977 | setara |
| puntung_rokok | FP | 0.0 | 4.0 | -4.0 | -9.0 | 0.0 | setara |
| TOTAL | FN | 218.0 | 180.0 | 38.0 | 18.0 | 60.0 | YOLOv12 lebih buruk |
| TOTAL | FP | 163.0 | 154.0 | 9.0 | -23.0 | 43.0 | setara |

Selisih mAP@50:

| kelas | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50_3kelas (3 kelas) | 0.4025915799873275 | 0.4624464203708438 | -0.0598548403835163 | -0.1266722208354063 | 0.0075101953143201 | setara |
| mAP50_semua_kelas_ber_GT (3 kelas) | 0.4025915799873275 | 0.4624464203708438 | -0.0598548403835163 | -0.1266722208354063 | 0.0075101953143201 | setara |

Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas), kelas yang punya GT:

| model_label | kelas | gt | TP | FN | FN_terlewat | FN_salah_kelas | FP | FP_latar | FP_salah_kelas |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv12x aug | kertas_tisu | 75 | 18 | 57 | 36 | 21 | 44 | 20 | 24 |
| YOLOv12x aug | logam_kaleng_kaca | 34 | 12 | 22 | 7 | 15 | 14 | 7 | 7 |
| YOLOv12x aug | plastik | 241 | 102 | 139 | 110 | 29 | 105 | 71 | 34 |
| YOLOv8x aug | kertas_tisu | 75 | 19 | 56 | 29 | 27 | 31 | 19 | 12 |
| YOLOv8x aug | logam_kaleng_kaca | 34 | 14 | 20 | 3 | 17 | 15 | 7 | 8 |
| YOLOv8x aug | plastik | 241 | 137 | 104 | 84 | 20 | 104 | 60 | 44 |

#### Test full

FN dan FP per kelas:

| kelas | metrik | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_tisu | FN | 164.0 | 159.0 | 5.0 | -9.025 | 20.0 | setara |
| kertas_tisu | FP | 192.0 | 141.0 | 51.0 | 26.0 | 79.0 | YOLOv12 lebih buruk |
| logam_kaleng_kaca | FN | 75.0 | 59.0 | 16.0 | 6.0 | 27.0 | YOLOv12 lebih buruk |
| logam_kaleng_kaca | FP | 61.0 | 66.0 | -5.0 | -25.0 | 16.0 | setara |
| plastik | FN | 374.0 | 292.0 | 82.0 | 53.0 | 112.0 | YOLOv12 lebih buruk |
| plastik | FP | 317.0 | 296.0 | 21.0 | -18.025 | 63.04999999999996 | setara |
| puntung_rokok | FN | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | setara |
| puntung_rokok | FP | 1.0 | 7.0 | -6.0 | -12.0 | -1.0 | YOLOv12 lebih baik |
| TOTAL | FN | 613.0 | 510.0 | 103.0 | 68.0 | 140.0 | YOLOv12 lebih buruk |
| TOTAL | FP | 571.0 | 510.0 | 61.0 | 2.9750000000000014 | 125.02499999999998 | YOLOv12 lebih buruk |

Selisih mAP@50:

| kelas | v12 | v8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50_3kelas (3 kelas) | 0.4866334851876979 | 0.5077783193878399 | -0.021144834200142 | -0.0559597916977862 | 0.0119980819720155 | setara |
| mAP50_semua_kelas_ber_GT (4 kelas) | 0.6137251138907734 | 0.6295837395408799 | -0.0158586256501064 | -0.0440350893847873 | 0.0096320994170236 | setara |

Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas), kelas yang punya GT:

| model_label | kelas | gt | TP | FN | FN_terlewat | FN_salah_kelas | FP | FP_latar | FP_salah_kelas |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv12x aug | kertas_tisu | 233 | 69 | 164 | 87 | 77 | 192 | 90 | 102 |
| YOLOv12x aug | logam_kaleng_kaca | 129 | 54 | 75 | 35 | 40 | 61 | 36 | 25 |
| YOLOv12x aug | plastik | 763 | 389 | 374 | 254 | 120 | 317 | 207 | 110 |
| YOLOv12x aug | puntung_rokok | 2 | 2 | 0 | 0 | 0 | 1 | 1 | 0 |
| YOLOv8x aug | kertas_tisu | 233 | 74 | 159 | 62 | 97 | 141 | 62 | 79 |
| YOLOv8x aug | logam_kaleng_kaca | 129 | 70 | 59 | 20 | 39 | 66 | 25 | 41 |
| YOLOv8x aug | plastik | 763 | 471 | 292 | 175 | 117 | 296 | 164 | 132 |
| YOLOv8x aug | puntung_rokok | 2 | 2 | 0 | 0 | 0 | 7 | 6 | 1 |

Sumber: `bootstrap_head_to_head_15model.csv`, `fp_fn_per_class_15model.csv`.

Catatan: mAP@50 di tabel bootstrap dihitung ulang dengan pencocokan `02_error_analysis.py` (AP 101 titik), sehingga bisa sedikit berbeda dari mAP@50 validator Ultralytics di bagian 2 (mis. YOLOv8x aug clean: validator 0.4436819852 vs bootstrap 0.4624459). Untuk tabel utama paper pakai angka validator (bagian 2); angka bootstrap hanya untuk selisih dan CI. Kolom `mAP50_semua_kelas_ber_GT` = cara `analisis_error.md` lama (di full = 4 kelas).

## 5. Kebingungan kertas/tisu ↔ plastik di test clean, per sumber

Kasus = pasangan GT–prediksi yang cocok (conf ≥ 0.25, IoU > 0.5) dengan kelas kertas/plastik tertukar, dijumlah atas model. Sumber citra dari `analysis/tables/D0_scene_groups.csv` (kolom `sumber`). `kasus per GT per model` = kasus ÷ (GT kertas+plastik × jumlah model). `% objek GT tertukar` = objek GT yang dikelirukan oleh minimal satu model ÷ GT kertas+plastik sumber itu.

| model | sumber | GT kertas+plastik | % GT | kasus | % kasus | kasus per GT per model | objek GT unik tertukar (min. 1 model) | % objek GT tertukar (min. 1 model) | Kertas/Tisu → Plastik | Plastik → Kertas/Tisu |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8l aug | UAVVaste | 152 | 0.4810126582278481 | 30 | 0.6976744186046512 | 0.1973684210526315 | 30 | 0.1973684210526315 | 14 | 16 |
| YOLOv8l aug | Burangrang | 139 | 0.439873417721519 | 12 | 0.2790697674418604 | 0.0863309352517985 | 12 | 0.0863309352517985 | 8 | 4 |
| YOLOv8l aug | TACO | 22 | 0.0696202531645569 | 1 | 0.0232558139534883 | 0.0454545454545454 | 1 | 0.0454545454545454 | 1 | 0 |
| YOLOv8l aug | TOTAL | 316 | 1.0 | 43 | 1.0 | 0.1360759493670886 | 43 | 0.1360759493670886 | 23 | 20 |
| YOLOv8x aug | UAVVaste | 152 | 0.4810126582278481 | 32 | 0.8205128205128205 | 0.2105263157894736 | 32 | 0.2105263157894736 | 22 | 10 |
| YOLOv8x aug | Burangrang | 139 | 0.439873417721519 | 7 | 0.1794871794871795 | 0.0503597122302158 | 7 | 0.0503597122302158 | 5 | 2 |
| YOLOv8x aug | TACO | 22 | 0.0696202531645569 | 0 | 0.0 | 0.0 | 0 | 0.0 | 0 | 0 |
| YOLOv8x aug | TOTAL | 316 | 1.0 | 39 | 1.0 | 0.1234177215189873 | 39 | 0.1234177215189873 | 27 | 12 |
| 13 model (lama) | UAVVaste | 152 | 0.4810126582278481 | 388 | 0.7320754716981132 | 0.1963562753036437 | 78 | 0.5131578947368421 | 248 | 140 |
| 13 model (lama) | Burangrang | 139 | 0.439873417721519 | 134 | 0.2528301886792453 | 0.0741560597675705 | 34 | 0.2446043165467626 | 100 | 34 |
| 13 model (lama) | TACO | 22 | 0.0696202531645569 | 8 | 0.0150943396226415 | 0.0279720279720279 | 4 | 0.1818181818181818 | 4 | 4 |
| 13 model (lama) | TOTAL | 316 | 1.0 | 530 | 1.0 | 0.1290165530671859 | 116 | 0.3670886075949367 | 352 | 178 |
| 15 model | UAVVaste | 152 | 0.4810126582278481 | 450 | 0.7352941176470589 | 0.1973684210526315 | 81 | 0.5328947368421053 | 284 | 166 |
| 15 model | Burangrang | 139 | 0.439873417721519 | 153 | 0.25 | 0.0733812949640287 | 35 | 0.2517985611510791 | 113 | 40 |
| 15 model | TACO | 22 | 0.0696202531645569 | 9 | 0.0147058823529411 | 0.0272727272727272 | 4 | 0.1818181818181818 | 5 | 4 |
| 15 model | TOTAL | 316 | 1.0 | 612 | 1.0 | 0.1291139240506329 | 120 | 0.379746835443038 | 402 | 210 |

Sumber: `label_noise_per_sumber_15model.csv`.
Baris `13 model (lama)` dihitung ulang dengan script yang sama dan cocok dengan angka lama (530 kasus, 116 objek unik, arah 352/178 di `analisis_error.md` dan `analisis_label_noise.md`).

## 6. Selisih full − clean

Selisih = metrik test full (3 kelas, tanpa puntung_rokok) − metrik test clean (3 kelas). Positif berarti angka full lebih tinggi. **Tanda kebalikan dari `ringkasan.md` lama**, yang memakai Δ = clean − full (lama: Δ mAP@50 rata-rata −0.058, Δ mAP@50-95 −0.053, dibulatkan 3 desimal).

### Per model (v8l, v8x)

| model | varian | augmentasi | mAP50_full_3kelas | mAP50_clean | mAP50_95_full_3kelas | mAP50_95_clean | selisih_full_minus_clean_mAP50 | selisih_full_minus_clean_mAP50_95 |
|---|---|---|---|---|---|---|---|---|
| YOLOv8 | l | ya | 0.4735974037198102 | 0.3964327510368735 | 0.2752132536558921 | 0.202896223259155 | 0.0771646526829367 | 0.0723170303967371 |
| YOLOv8 | x | ya | 0.4920087086865772 | 0.4436819610216208 | 0.2916526973786428 | 0.2304425316927352 | 0.0483267476649564 | 0.0612101656859075 |

### Agregat

| kelompok | n | metrik | rata_rata | minimum | model_min | maksimum | model_maks |
|---|---|---|---|---|---|---|---|
| 13 model (lama) | 13 | selisih_full_minus_clean_mAP50 | 0.0583168412220604 | 0.0237277176571666 | YOLOv12x | 0.0930099093145747 | YOLOv12l aug |
| 13 model (lama) | 13 | selisih_full_minus_clean_mAP50_95 | 0.0526560372854674 | 0.015338655153561 | YOLOv12x | 0.0834208604444576 | YOLOv12l aug |
| 15 model | 15 | selisih_full_minus_clean_mAP50 | 0.0589073557489785 | 0.0237277176571666 | YOLOv12x | 0.0930099093145747 | YOLOv12l aug |
| 15 model | 15 | selisih_full_minus_clean_mAP50_95 | 0.054537045386248 | 0.015338655153561 | YOLOv12x | 0.0834208604444576 | YOLOv12l aug |
| 5 model YOLOv8 | 5 | selisih_full_minus_clean_mAP50 | 0.0563256802111243 | 0.0381386722140893 | YOLOv8s aug | 0.0771646526829367 | YOLOv8l aug |
| 5 model YOLOv8 | 5 | selisih_full_minus_clean_mAP50_95 | 0.0593317381500524 | 0.0497483252793001 | YOLOv8s aug | 0.0723170303967371 | YOLOv8l aug |
| 3 model YOLOv8 lama (n/s/m) | 3 | selisih_full_minus_clean_mAP50 | 0.0520456669025762 | 0.0381386722140893 | YOLOv8s aug | 0.0656682992415896 | YOLOv8n aug |
| 3 model YOLOv8 lama (n/s/m) | 3 | selisih_full_minus_clean_mAP50_95 | 0.0543771648892058 | 0.0497483252793001 | YOLOv8s aug | 0.0608324965392238 | YOLOv8m aug |

Sumber: `selisih_full_clean_15model.csv`, `selisih_full_clean_agregat_15model.csv`.

## 7. Peringkat 10 model augmented

### mAP50 4 kelas, test full

| peringkat | model_label | mAP50 | mAP50_95 | mAP50_3kelas | mAP50_95_3kelas | ms_per_img |
|---|---|---|---|---|---|---|
| 1 | YOLOv12m aug | 0.6329757949385129 | 0.4007072905727694 | 0.5123010599180171 | 0.3022763874303593 | 8.681996715727541 |
| 2 | YOLOv12l aug | 0.6250028061183419 | 0.268947080030496 | 0.5016704081577892 | 0.2924294400406614 | 12.65993631312311 |
| 3 | YOLOv12n aug | 0.6186702855118745 | 0.2652396096878621 | 0.493227047349166 | 0.2874861462504828 | 2.201430110369612 |
| 4 | YOLOv8n aug | 0.6184326879737074 | 0.310084392025944 | 0.4929102506316098 | 0.280945856034592 | 1.0726346730349765 |
| 5 | YOLOv8x aug | 0.6177565315149329 | 0.3306145230339821 | 0.4920087086865772 | 0.2916526973786428 | 17.423034488413837 |
| 6 | YOLOv12x aug | 0.6079000775198025 | 0.3275275288754762 | 0.4788667700264032 | 0.271370038500635 | 21.94561733631417 |
| 7 | YOLOv8l aug | 0.6039480527898576 | 0.3554099402419191 | 0.4735974037198102 | 0.2752132536558921 | 10.798456935453345 |
| 8 | YOLOv8s aug | 0.5475155458267651 | 0.3003092918121569 | 0.5089096166579091 | 0.3010790557495424 | 2.7754910585103407 |
| 9 | YOLOv12s aug | 0.5202765916969821 | 0.2791566227034511 | 0.5248707046381218 | 0.3046762632214598 | 3.841591238836166 |
| 10 | YOLOv8m aug | 0.5074040258271659 | 0.2960228222152722 | 0.5107053677695546 | 0.303530429620363 | 6.191171717619265 |

### mAP50 3 kelas, test clean

| peringkat | model_label | mAP50 | mAP50_95 | mAP50_3kelas | mAP50_95_3kelas | ms_per_img |
|---|---|---|---|---|---|---|
| 1 | YOLOv8s aug | 0.4707709444438198 | 0.2513307304702423 | 0.4707709444438198 | 0.2513307304702423 | 3.4850968744043107 |
| 2 | YOLOv12s aug | 0.4648252185615815 | 0.2514466902126503 | 0.4648252185615815 | 0.2514466902126502 | 4.124677605432225 |
| 3 | YOLOv8m aug | 0.4583753385175047 | 0.2426979330811391 | 0.4583753385175047 | 0.2426979330811391 | 6.473719272131954 |
| 4 | YOLOv8x aug | 0.4436819610216208 | 0.2304425316927352 | 0.4436819610216208 | 0.2304425316927352 | 17.377001562029665 |
| 5 | YOLOv12n aug | 0.4393407264428872 | 0.2384189871597918 | 0.4393407264428872 | 0.2384189871597918 | 2.31466823121688 |
| 6 | YOLOv8n aug | 0.4272419513900202 | 0.2283951831854985 | 0.4272419513900202 | 0.2283951831854984 | 1.878312500290728 |
| 7 | YOLOv12m aug | 0.4270044832804343 | 0.2253796861110016 | 0.4270044832804343 | 0.2253796861110016 | 8.686503646458732 |
| 8 | YOLOv12l aug | 0.4086604988432145 | 0.2090085795962038 | 0.4086604988432145 | 0.2090085795962038 | 12.723902604193428 |
| 9 | YOLOv8l aug | 0.3964327510368735 | 0.202896223259155 | 0.3964327510368735 | 0.202896223259155 | 10.80159687262494 |
| 10 | YOLOv12x aug | 0.3936856638395928 | 0.2061622643363625 | 0.3936856638395928 | 0.2061622643363625 | 21.958091147098457 |

Sumber: `peringkat_augmented_15model.csv`.
ms_per_img di tabel ini diukur di test set yang sama dengan kriterianya.

Peringkat (i) dipengaruhi kuat oleh AP puntung_rokok yang hanya punya 2 instance di test full.

## 8. Ringkasan 10 model augmented (3 kelas)

| model_label | mAP50_3kelas full | mAP50_95_3kelas full | mAP50_3kelas clean | mAP50_95_3kelas clean | ms_per_img full | ms_per_img clean |
|---|---|---|---|---|---|---|
| YOLOv12n aug | 0.493227047349166 | 0.2874861462504828 | 0.4393407264428872 | 0.2384189871597918 | 2.201430110369612 | 2.31466823121688 |
| YOLOv12s aug | 0.5248707046381218 | 0.3046762632214598 | 0.4648252185615815 | 0.2514466902126502 | 3.841591238836166 | 4.124677605432225 |
| YOLOv12m aug | 0.5123010599180171 | 0.3022763874303593 | 0.4270044832804343 | 0.2253796861110016 | 8.681996715727541 | 8.686503646458732 |
| YOLOv12l aug | 0.5016704081577892 | 0.2924294400406614 | 0.4086604988432145 | 0.2090085795962038 | 12.65993631312311 | 12.723902604193428 |
| YOLOv12x aug | 0.4788667700264032 | 0.271370038500635 | 0.3936856638395928 | 0.2061622643363625 | 21.94561733631417 | 21.958091147098457 |
| YOLOv8n aug | 0.4929102506316098 | 0.280945856034592 | 0.4272419513900202 | 0.2283951831854984 | 1.0726346730349765 | 1.878312500290728 |
| YOLOv8s aug | 0.5089096166579091 | 0.3010790557495424 | 0.4707709444438198 | 0.2513307304702423 | 2.7754910585103407 | 3.4850968744043107 |
| YOLOv8m aug | 0.5107053677695546 | 0.303530429620363 | 0.4583753385175047 | 0.2426979330811391 | 6.191171717619265 | 6.473719272131954 |
| YOLOv8l aug | 0.4735974037198102 | 0.2752132536558921 | 0.3964327510368735 | 0.202896223259155 | 10.798456935453345 | 10.80159687262494 |
| YOLOv8x aug | 0.4920087086865772 | 0.2916526973786428 | 0.4436819610216208 | 0.2304425316927352 | 17.423034488413837 | 17.377001562029665 |

Sumber: `test_metrics_overall_15model.csv`.


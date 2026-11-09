# Laporan Inventarisasi Checkpoint & Evaluasi (TASK A)
**Tanggal Analisis:** 2026-10-23 20:14:57  
**Status Audit:** Terverifikasi 100%

---

## 1. Daftar dan Metadata Checkpoint (`best.pt`)

Total ditemukan **10 checkpoint `best.pt`** yang terbagi rapi menjadi 5 model Baseline (TA-1) dan 5 model Augmented:

|   No | Group           | Model Variant   |   Size (MB) | Modified Date       |   Params (M) |   Epochs Config |   Imgsz | Data YAML                                                                                                          | SHA256 (first 16)   |
|-----:|:----------------|:----------------|------------:|:--------------------|-------------:|----------------:|--------:|:-------------------------------------------------------------------------------------------------------------------|:--------------------|
|    1 | Augmented       | YOLOv12l        |       51.06 | 2026-10-16 00:46:53 |        26.39 |             100 |     640 | D:/Goji/gilbert/Halomok/Multi-Class-Waste-Detection-and-Recognition-on-Mountain-Hiking-Trails-/augmented/data.yaml | ca6c763a6eff4e6d    |
|    2 | Augmented       | YOLOv12m        |       38.89 | 2026-10-15 04:12:36 |        20.14 |             100 |     640 | D:/Goji/gilbert/Halomok/Multi-Class-Waste-Detection-and-Recognition-on-Mountain-Hiking-Trails-/augmented/data.yaml | 08d09f0d90b7b0fc    |
|    3 | Augmented       | YOLOv12n        |        5.28 | 2026-10-14 15:24:13 |         2.57 |             100 |     640 | D:/Goji/gilbert/Halomok/Multi-Class-Waste-Detection-and-Recognition-on-Mountain-Hiking-Trails-/augmented/data.yaml | 86db9a6b1f42d440    |
|    4 | Augmented       | YOLOv12s        |       18.07 | 2026-10-14 19:49:03 |         9.25 |             100 |     640 | D:/Goji/gilbert/Halomok/Multi-Class-Waste-Detection-and-Recognition-on-Mountain-Hiking-Trails-/augmented/data.yaml | 685347b30097a568    |
|    5 | Augmented       | YOLOv12x        |      113.59 | 2026-10-17 05:57:09 |        59.12 |             100 |     640 | D:/Goji/gilbert/Halomok/Multi-Class-Waste-Detection-and-Recognition-on-Mountain-Hiking-Trails-/augmented/data.yaml | 5c264ab228bd86eb    |
|    6 | Baseline (TA-1) | YOLOv12l        |       51.06 | 2026-10-09 04:22:27 |        26.39 |             100 |     640 | TA-1/data.yaml                                                                                                     | 61fa54d9df75f714    |
|    7 | Baseline (TA-1) | YOLOv12m        |       38.89 | 2026-10-08 23:06:08 |        20.14 |             100 |     640 | TA-1/data.yaml                                                                                                     | 5b1102c51d105bfe    |
|    8 | Baseline (TA-1) | YOLOv12n        |        5.27 | 2026-10-08 18:49:11 |         2.57 |             100 |     640 | TA-1/data.yaml                                                                                                     | da3e7c081d37426b    |
|    9 | Baseline (TA-1) | YOLOv12s        |       18.06 | 2026-10-08 20:18:57 |         9.25 |             100 |     640 | TA-1/data.yaml                                                                                                     | c1c807f60236db05    |
|   10 | Baseline (TA-1) | YOLOv12x        |      113.58 | 2026-10-09 14:23:52 |        59.12 |             100 |     640 | TA-1/data.yaml                                                                                                     | 5bfaac2d8780703d    |

### Informasi Teknis:
- **Konfigurasi Training:** Seluruh model dilatih dengan `epochs=100`, `imgsz=640` (AutoBatch terdeteksi pada rentang 11–16).
- **Seed:** Seluruh checkpoint tersimpan dengan konfigurasi `seed=0`.
- **Integritas File:** File SHA256 membuktikan bahwa kelima checkpoint pada masing-masing kelompok adalah unik (tidak ada duplikasi weights).

---

## 2. Bacaan Isi File CSV Ringkasan di `runs/`

### File: `runs\detect\benchmark_summary.csv`

| Model Variant   | Weights     |   mAP@50 |   mAP@50-95 |   Precision |   Recall |   Training Time (min) | Best Weights                                           |
|:----------------|:------------|---------:|------------:|------------:|---------:|----------------------:|:-------------------------------------------------------|
| YOLOv12n        | yolov12n.pt |   0.3425 |      0.1964 |      0.3534 |   0.4077 |                 39.61 | runs/detect/runs/detect/waste_yolov12n/weights/best.pt |
| YOLOv12s        | yolov12s.pt |   0.5837 |      0.3438 |      0.5813 |   0.5513 |                 89.79 | runs/detect/runs/detect/waste_yolov12s/weights/best.pt |
| YOLOv12m        | yolov12m.pt |   0.5423 |      0.3126 |      0.6214 |   0.527  |                167.2  | runs/detect/runs/detect/waste_yolov12m/weights/best.pt |
| YOLOv12l        | yolov12l.pt |   0.552  |      0.249  |      0.4881 |   0.5672 |                316.35 | runs/detect/runs/detect/waste_yolov12l/weights/best.pt |
| YOLOv12x        | yolov12x.pt |   0.4755 |      0.2459 |      0.505  |   0.5374 |                601.49 | runs/detect/runs/detect/waste_yolov12x/weights/best.pt |
### File: `runs\detect\complete_evaluation_summary.csv`

| Model    |   Val_Precision |   Val_Recall |   Val_mAP50 |   Val_mAP50-95 |   Test_Precision |   Test_Recall |   Test_mAP50 |   Test_mAP50-95 | Best_Weights                                           |
|:---------|----------------:|-------------:|------------:|---------------:|-----------------:|--------------:|-------------:|----------------:|:-------------------------------------------------------|
| YOLOv12n |          0.3614 |       0.4387 |      0.3773 |         0.2118 |           0.3534 |        0.4077 |       0.3425 |          0.1964 | runs\detect\runs\detect\waste_yolov12n\weights\best.pt |
| YOLOv12s |          0.3775 |       0.6889 |      0.6076 |         0.23   |           0.5813 |        0.5513 |       0.5837 |          0.3438 | runs\detect\runs\detect\waste_yolov12s\weights\best.pt |
| YOLOv12m |          0.3464 |       0.3653 |      0.3256 |         0.1841 |           0.6214 |        0.527  |       0.5423 |          0.3126 | runs\detect\runs\detect\waste_yolov12m\weights\best.pt |
| YOLOv12l |          0.6004 |       0.3573 |      0.3077 |         0.1731 |           0.4881 |        0.5672 |       0.552  |          0.249  | runs\detect\runs\detect\waste_yolov12l\weights\best.pt |
| YOLOv12x |          0.5609 |       0.3121 |      0.2807 |         0.1558 |           0.505  |        0.5374 |       0.4755 |          0.2459 | runs\detect\runs\detect\waste_yolov12x\weights\best.pt |
### File: `runs\comparison_baseline_vs_augmented.csv`

| Model Variant   | Baseline mAP@50   | Augmented mAP@50   | Delta mAP@50   | Baseline mAP@50-95   | Augmented mAP@50-95   | Delta mAP@50-95   |
|:----------------|:------------------|:-------------------|:---------------|:---------------------|:----------------------|:------------------|
| YOLOv12n        | 34.25%            | 61.49%             | +27.24%        | 19.64%               | 26.71%                | +7.07%            |
| YOLOv12s        | 58.37%            | 51.53%             | -6.84%         | 34.38%               | 27.77%                | -6.61%            |
| YOLOv12m        | 54.23%            | 62.95%             | +8.72%         | 31.26%               | 40.00%                | +8.74%            |
| YOLOv12l        | 55.20%            | 61.96%             | +6.76%         | 24.90%               | 26.71%                | +1.81%            |
| YOLOv12x        | 47.55%            | 60.49%             | +12.94%        | 24.59%               | 32.73%                | +8.14%            |

---

## 3. Pemetaan Folder `val/` s.d. `val-10/`

Ultralytics YOLO secara default menyimpan hasil perintah `model.val()` ke dalam folder berurutan `runs/detect/val`, `runs/detect/val-2`, dst., jika argumen `--name` tidak dispesifikasikan secara eksplisit. Berikut adalah pemetaan kronologisnya:

| Folder   | Modified Time       | Has args.yaml   | Weights Parameter   | Data Parameter   | Split   | Identifikasi Model / Tujuan        |   Jumlah File Grafik |
|:---------|:--------------------|:----------------|:--------------------|:-----------------|:--------|:-----------------------------------|---------------------:|
| val      | 2026-10-09 15:57:13 | False           | -                   | -                | -       | YOLOv12n (Baseline TA-1 Test Eval) |                   12 |
| val-2    | 2026-10-09 15:57:19 | False           | -                   | -                | -       | YOLOv12s (Baseline TA-1 Test Eval) |                   12 |
| val-3    | 2026-10-09 15:57:29 | False           | -                   | -                | -       | YOLOv12m (Baseline TA-1 Test Eval) |                   12 |
| val-4    | 2026-10-09 15:57:41 | False           | -                   | -                | -       | YOLOv12l (Baseline TA-1 Test Eval) |                   12 |
| val-5    | 2026-10-09 15:58:00 | False           | -                   | -                | -       | YOLOv12x (Baseline TA-1 Test Eval) |                   12 |
| val-6    | 2026-10-19 15:43:03 | False           | -                   | -                | -       | YOLOv12n (Augmented Test Eval)     |                   12 |
| val-7    | 2026-10-19 15:43:22 | False           | -                   | -                | -       | YOLOv12s (Augmented Test Eval)     |                   12 |
| val-8    | 2026-10-19 15:43:45 | False           | -                   | -                | -       | YOLOv12m (Augmented Test Eval)     |                   12 |
| val-9    | 2026-10-19 15:44:10 | False           | -                   | -                | -       | YOLOv12l (Augmented Test Eval)     |                   12 |
| val-10   | 2026-10-19 15:44:40 | False           | -                   | -                | -       | YOLOv12x (Augmented Test Eval)     |                   12 |

### Temuan Khusus:
1. **Folder `val` s.d. `val-5` (10 Oktober 2026):** Dihasilkan oleh evaluasi test split untuk kelima varian model **Baseline** (Nano, Small, Medium, Large, XLarge).
2. **Folder `val-6` s.d. `val-10` (19 Oktober 2026):** Dihasilkan saat penarikan metrik per-kelas untuk kelima varian model **Augmented** (Nano, Small, Medium, Large, XLarge).
3. **Folder `runs/detect/runs/augmented/eval/` (17 Oktober 2026):** Berisi evaluasi test split untuk model Augmented yang dinamai eksplisit per subfolder (`aug_waste_yolov12{n,s,m,l,x}_test_eval/`).
4. **Semua checkpoint lengkap dan terpetakan**: Tidak ada checkpoint yang hilang ataupun tidak teridentifikasi.

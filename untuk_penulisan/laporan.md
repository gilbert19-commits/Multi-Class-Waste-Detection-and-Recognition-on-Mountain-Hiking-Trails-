# Laporan bahan penulisan paper (dataset 2 kelas)

Semua angka berasal dari skrip di `scripts/` dengan setting yang sama dengan `hasil_2kelas`: split test 550 citra, imgsz 512, conf 0.001, iou 0.7, best.pt, 15 model. Tidak ada training ulang, dan folder runs serta dataset tidak berubah. Semua langkah Bagian A dan B berhasil dijalankan. Prediksi per gambar dipakai ulang dari `hasil_2kelas/_cache`; inferensi baru hanya untuk `single_cls=True` di verifikasi b. Satuan selisih dan CI adalah poin persen.

## 1. Temuan penting

**a. Aturan sumber di `hasil_2kelas` salah untuk sebagian file, dan ini mengubah cerita "data Burangrang".** Nama file saya cocokkan dengan anotasi resmi TACO dan UAVVaste. Semua 558 file berpola `IMG_####_JPG.rf.*` ada di TACO (`batch_3`), dan semua 209 file `GOPR` ada di UAVVaste, padahal keduanya dihitung Burangrang. Pola Unknown (`camera_img_`, `photo_`, `DJI_`) juga semuanya ada di UAVVaste. Tampilan contact sheet mendukung ini: lantai, aspal, meja, permukaan air, dan daun gugur mirip maple. Yang tidak ada di kedua daftar hanya `WhatsApp-Video` dan `IMG_####_JPG_jpg.rf.*`, jadi hanya keduanya yang masih mungkin data sendiri, dan itu perlu kamu pastikan. Dengan aturan usulan, Burangrang di test tinggal 300 citra, 232 di antaranya tanpa objek, dengan **207 objek lightweight litter dan hanya 1 objek metal and glass**. Di train hanya ada 10 objek metal and glass, dan di valid 0. Dari 1131 objek test, hanya 208 yang berasal dari Burangrang. Jadi performa kelas metal and glass pada data sendiri tidak bisa diukur dari dataset ini. Saya tidak mengubah file di `hasil_2kelas/`. Lihat bagian 4.4.

**b. Dengan class-agnostic mAP, YOLOv8 tampak lebih rendah di beberapa skala, tetapi kekuatan kesimpulannya bergantung pada cara koreksi.** Di mAP standar tidak ada perbandingan per skala yang lolos Holm. Di versi 3, 12 dari 30 perbandingan punya CI yang tidak melewati nol, semuanya menguntungkan model yang lebih baru. Dengan Holm per metrik dan p konservatif (+1), yang bertahan: YOLO26n vs YOLOv12n (mAP@50-95, +1.68); YOLO26n vs YOLOv8n (mAP@50, +2.96); YOLO26n vs YOLOv8n (mAP@50-95, +3.39); YOLOv12n vs YOLOv8n (mAP@50-95, +1.71); YOLOv12m vs YOLOv8m (mAP@50-95, +1.54); YOLO26l vs YOLOv8l (mAP@50-95, +1.88). Lihat bagian 2.2.

**c. Efek salah kelas nyata di semua model.** 45 dari 45 CI (15 model, 3 metrik) berada di atas nol. Pada mAP@50 besarnya 6.24 sampai 8.37 poin. Pada conf 0.25 dan IoU 0.5, 26.4 sampai 27.1% objek metal and glass terdeteksi dengan kelas salah, sedangkan lightweight litter hanya 1.5 sampai 3.2%.

**d. Angka harapan 0.9 dan 0.8 di verifikasi c tidak sesuai dengan Ultralytics yang terpasang.** Script dan hitungan manual sama-sama menghasilkan 0.475 dan 0.650 dengan rumus Ultralytics 8.4.123. Angka 0.9 dan 0.772 hanya muncul dengan rumus titik penutup yang lebih lama. Akibatnya, mAP pada Ultralytics 8.4.123 lebih rendah daripada rumus lama untuk prediksi yang sama. Lihat bagian 3.

**e. `single_cls=True` di YOLO26 tidak setara dengan versi 3.** Selisihnya sampai 0.0116, melewati batas 0.05, karena validator tidak menjalankan NMS untuk model end-to-end. Hasilnya tepat sama dengan versi 4. Lihat bagian 3.

## 2. Bagian A: class-agnostic mAP

Definisi versi: (1) standar, (2) pooled class-aware, (3) class-agnostic dengan NMS ulang IoU 0.7, (4) class-agnostic tanpa NMS ulang. File: `class_agnostic/metrics_4versi.csv`.

### 2.1 Empat versi (dalam persen)

**mAP@50**

| Model | 1 Standar | 2 Pooled | 3 Agnostic | 4 Tanpa NMS ulang | Efek bobot (2-1) | Efek salah kelas (3-2) | Efek NMS ulang (3-4) |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 62.07 | 68.91 | 75.15 | 74.42 | 6.84 | 6.24 | 0.72 |
| YOLOv8s | 64.82 | 72.60 | 79.59 | 79.06 | 7.79 | 6.99 | 0.53 |
| YOLOv8m | 64.38 | 71.37 | 78.10 | 77.53 | 6.99 | 6.73 | 0.57 |
| YOLOv8l | 66.04 | 72.25 | 78.51 | 77.88 | 6.21 | 6.26 | 0.63 |
| YOLOv8x | 64.48 | 70.05 | 77.02 | 76.30 | 5.57 | 6.97 | 0.72 |
| YOLOv12n | 61.39 | 69.43 | 75.68 | 75.20 | 8.04 | 6.24 | 0.48 |
| YOLOv12s | 64.95 | 72.38 | 79.33 | 78.74 | 7.44 | 6.94 | 0.58 |
| YOLOv12m | 64.61 | 72.24 | 79.67 | 79.25 | 7.63 | 7.43 | 0.42 |
| YOLOv12l | 66.58 | 72.29 | 78.94 | 78.14 | 5.71 | 6.65 | 0.80 |
| YOLOv12x | 66.23 | 71.81 | 78.30 | 77.28 | 5.58 | 6.49 | 1.02 |
| YOLO26n | 62.40 | 70.46 | 78.11 | 76.95 | 8.06 | 7.65 | 1.16 |
| YOLO26s | 63.65 | 71.04 | 79.40 | 78.51 | 7.39 | 8.36 | 0.89 |
| YOLO26m | 64.58 | 72.79 | 80.42 | 79.47 | 8.22 | 7.63 | 0.95 |
| YOLO26l | 64.54 | 71.21 | 79.58 | 78.61 | 6.67 | 8.37 | 0.97 |
| YOLO26x | 64.09 | 70.77 | 78.97 | 77.89 | 6.68 | 8.21 | 1.08 |

**mAP@75**

| Model | 1 Standar | 2 Pooled | 3 Agnostic | 4 Tanpa NMS ulang | Efek bobot (2-1) | Efek salah kelas (3-2) | Efek NMS ulang (3-4) |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 34.83 | 36.23 | 39.50 | 39.45 | 1.40 | 3.26 | 0.05 |
| YOLOv8s | 38.99 | 41.60 | 45.59 | 45.57 | 2.61 | 3.99 | 0.02 |
| YOLOv8m | 39.92 | 41.49 | 45.25 | 45.39 | 1.57 | 3.76 | -0.14 |
| YOLOv8l | 42.12 | 40.96 | 44.77 | 45.07 | -1.16 | 3.81 | -0.30 |
| YOLOv8x | 41.10 | 41.35 | 45.81 | 45.75 | 0.25 | 4.46 | 0.06 |
| YOLOv12n | 36.09 | 39.11 | 42.69 | 42.76 | 3.02 | 3.58 | -0.07 |
| YOLOv12s | 39.42 | 41.83 | 45.65 | 45.98 | 2.41 | 3.82 | -0.34 |
| YOLOv12m | 42.08 | 44.30 | 48.64 | 48.74 | 2.22 | 4.34 | -0.10 |
| YOLOv12l | 44.67 | 43.79 | 47.66 | 47.77 | -0.88 | 3.87 | -0.11 |
| YOLOv12x | 40.58 | 42.14 | 45.85 | 45.84 | 1.56 | 3.71 | 0.02 |
| YOLO26n | 36.64 | 40.24 | 43.38 | 43.75 | 3.60 | 3.14 | -0.37 |
| YOLO26s | 40.94 | 43.03 | 47.05 | 47.44 | 2.09 | 4.02 | -0.39 |
| YOLO26m | 40.70 | 43.12 | 46.25 | 46.80 | 2.42 | 3.13 | -0.55 |
| YOLO26l | 42.64 | 45.00 | 48.96 | 49.47 | 2.36 | 3.96 | -0.51 |
| YOLO26x | 41.12 | 42.54 | 46.14 | 46.53 | 1.42 | 3.60 | -0.39 |

**mAP@50-95**

| Model | 1 Standar | 2 Pooled | 3 Agnostic | 4 Tanpa NMS ulang | Efek bobot (2-1) | Efek salah kelas (3-2) | Efek NMS ulang (3-4) |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 34.89 | 37.75 | 41.09 | 40.93 | 2.86 | 3.34 | 0.16 |
| YOLOv8s | 38.93 | 41.75 | 45.75 | 45.63 | 2.83 | 3.99 | 0.12 |
| YOLOv8m | 39.29 | 41.26 | 45.01 | 44.88 | 1.98 | 3.75 | 0.12 |
| YOLOv8l | 39.88 | 41.39 | 44.97 | 44.85 | 1.51 | 3.57 | 0.12 |
| YOLOv8x | 38.62 | 40.14 | 44.30 | 44.08 | 1.52 | 4.15 | 0.21 |
| YOLOv12n | 35.95 | 39.26 | 42.80 | 42.69 | 3.31 | 3.54 | 0.11 |
| YOLOv12s | 39.43 | 41.93 | 45.88 | 45.78 | 2.49 | 3.95 | 0.10 |
| YOLOv12m | 39.34 | 42.19 | 46.55 | 46.44 | 2.85 | 4.36 | 0.11 |
| YOLOv12l | 40.67 | 42.14 | 45.91 | 45.72 | 1.46 | 3.77 | 0.18 |
| YOLOv12x | 40.01 | 41.78 | 45.54 | 45.25 | 1.77 | 3.77 | 0.29 |
| YOLO26n | 37.23 | 40.60 | 44.48 | 44.34 | 3.37 | 3.88 | 0.14 |
| YOLO26s | 38.91 | 41.70 | 46.16 | 46.09 | 2.79 | 4.46 | 0.07 |
| YOLO26m | 39.31 | 42.41 | 46.50 | 46.34 | 3.09 | 4.09 | 0.15 |
| YOLO26l | 39.81 | 42.40 | 46.84 | 46.78 | 2.59 | 4.44 | 0.06 |
| YOLO26x | 39.66 | 41.79 | 46.00 | 45.87 | 2.13 | 4.21 | 0.14 |

Nilai versi 3 tertinggi: mAP@50 YOLO26m (80.42), mAP@75 YOLO26l (48.96), mAP@50-95 YOLO26l (46.84).

Efek bobot positif pada semua model di mAP@50 (5.57 sampai 8.22 poin) dan mAP@50-95 (1.46 sampai 3.37 poin), karena kelas lightweight litter yang AP-nya lebih tinggi mendominasi objek (1002 banding 129). Di mAP@75 efeknya negatif pada YOLOv8l dan YOLOv12l (-1.16 dan -0.88 poin): keduanya satu-satunya model yang AP@75 metal and glass-nya lebih tinggi dari lightweight litter (`hasil_2kelas/metrics_per_class.csv`). Efek ini terutama soal pembobotan, bukan peningkatan kemampuan model.

### 2.2 Bootstrap versi 3

Protokol sama dengan `hasil_2kelas`: paired percentile, level gambar, 1000 resample, seed 0, dan indeks resample yang sama persis. File: `class_agnostic/bootstrap_agnostic.csv`.

**Per skala.** CI tidak melewati nol pada 12 dari 30 perbandingan (mAP@50 dan mAP@50-95): YOLO26n vs YOLOv12n (mAP@50, +2.43); YOLO26n vs YOLOv12n (mAP@50-95, +1.68); YOLO26n vs YOLOv8n (mAP@50, +2.96); YOLO26n vs YOLOv8n (mAP@50-95, +3.39); YOLOv12n vs YOLOv8n (mAP@50-95, +1.71); YOLO26m vs YOLOv8m (mAP@50, +2.32); YOLO26m vs YOLOv8m (mAP@50-95, +1.49); YOLOv12m vs YOLOv8m (mAP@50-95, +1.54); YOLO26l vs YOLOv8l (mAP@50-95, +1.88); YOLO26x vs YOLOv8x (mAP@50, +1.96); YOLO26x vs YOLOv8x (mAP@50-95, +1.71); YOLOv12x vs YOLOv8x (mAP@50-95, +1.24). Selisih dalam poin persen, tanda positif menguntungkan model pertama.

Jumlah yang tetap signifikan setelah Holm:

| Keluarga Holm | p mentah: mAP@50 | p mentah: mAP@50-95 | p +1: mAP@50 | p +1: mAP@50-95 |
|---|---|---|---|---|
| 15 per metrik | 3 | 5 | 1 | 5 |
| 30 gabungan | 1 | 3 | 0 | 0 |

Beberapa perbandingan punya p mentah 0, artinya tidak ada satu pun dari 1000 resample yang melewati nol. Itu bukan berarti p benar-benar 0. Dengan koreksi +1, p terkecil yang mungkin adalah 2/1001 = 0.02. Akibatnya Holm 30 gabungan paling kecil 0.0599, jadi dengan 1000 resample keluarga 30 gabungan (+1) **tidak mungkin** mencapai 0.5. Angka 0 di sel itu adalah batas resolusi, bukan bukti tidak ada perbedaan. Keluarga 15 per metrik dengan p +1 adalah pilihan yang masih bisa mencapai signifikansi dan tidak melebih-lebihkan p. Pilihan akhirnya ada padamu.

**Antar model terbaik tiap family.** Terbaik menurut mAP@50-95 standar: YOLOv8l, YOLOv12l, YOLO26l. Terbaik menurut versi 3: YOLOv8s, YOLOv12m, YOLO26l. Pemilihannya dilakukan pada test yang sama, jadi selisihnya cenderung sedikit optimistis. Tidak ada koreksi berganda di tabel ini.

| Pilihan | A | B | Metrik | A (%) | B (%) | Selisih | CI bawah | CI atas | Hasil | p (+1) |
|---|---|---|---|---|---|---|---|---|---|---|
| standar | YOLO26l | YOLOv12l | mAP50 | 79.58 | 78.94 | 0.64 | -1.12 | 2.52 | setara | 0.470 |
| standar | YOLO26l | YOLOv12l | mAP75 | 48.96 | 47.66 | 1.30 | -1.22 | 3.62 | setara | 0.300 |
| standar | YOLO26l | YOLOv12l | mAP50_95 | 46.84 | 45.91 | 0.94 | -0.20 | 2.05 | setara | 0.104 |
| standar | YOLO26l | YOLOv8l | mAP50 | 79.58 | 78.51 | 1.07 | -0.67 | 2.83 | setara | 0.240 |
| standar | YOLO26l | YOLOv8l | mAP75 | 48.96 | 44.77 | 4.19 | 1.60 | 6.38 | YOLO26l | 0.002 |
| standar | YOLO26l | YOLOv8l | mAP50_95 | 46.84 | 44.97 | 1.88 | 0.78 | 2.98 | YOLO26l | 0.002 |
| standar | YOLOv12l | YOLOv8l | mAP50 | 78.94 | 78.51 | 0.43 | -1.14 | 2.04 | setara | 0.551 |
| standar | YOLOv12l | YOLOv8l | mAP75 | 47.66 | 44.77 | 2.89 | 0.16 | 5.48 | YOLOv12l | 0.036 |
| standar | YOLOv12l | YOLOv8l | mAP50_95 | 45.91 | 44.97 | 0.94 | -0.04 | 1.98 | setara | 0.058 |
| versi3 | YOLO26l | YOLOv12m | mAP50 | 79.58 | 79.67 | -0.09 | -1.67 | 1.58 | setara | 0.957 |
| versi3 | YOLO26l | YOLOv12m | mAP75 | 48.96 | 48.64 | 0.32 | -2.13 | 2.44 | setara | 0.837 |
| versi3 | YOLO26l | YOLOv12m | mAP50_95 | 46.84 | 46.55 | 0.30 | -0.61 | 1.26 | setara | 0.498 |
| versi3 | YOLO26l | YOLOv8s | mAP50 | 79.58 | 79.59 | -0.01 | -1.76 | 1.47 | setara | 0.913 |
| versi3 | YOLO26l | YOLOv8s | mAP75 | 48.96 | 45.59 | 3.37 | 0.59 | 5.94 | YOLO26l | 0.022 |
| versi3 | YOLO26l | YOLOv8s | mAP50_95 | 46.84 | 45.75 | 1.10 | 0.04 | 2.14 | YOLO26l | 0.048 |
| versi3 | YOLOv12m | YOLOv8s | mAP50 | 79.67 | 79.59 | 0.08 | -1.48 | 1.51 | setara | 0.969 |
| versi3 | YOLOv12m | YOLOv8s | mAP75 | 48.64 | 45.59 | 3.05 | 0.69 | 5.50 | YOLOv12m | 0.012 |
| versi3 | YOLOv12m | YOLOv8s | mAP50_95 | 46.55 | 45.75 | 0.80 | -0.09 | 1.64 | setara | 0.084 |

**Efek salah kelas (versi 3 dikurangi versi 2).** CI di atas nol pada 45 dari 45 kombinasi model dan metrik. Rentang batas bawah CI di mAP@50: 4.33 sampai 6.15. Rincian per model ada di file yang sama (`jenis = efek_salah_kelas`).

### 2.3 Rincian per kelas ground truth

Conf 0.25, IoU 0.5, satu prediksi hanya untuk satu ground truth; pencocokan dengan kelas benar didahulukan. Angkanya cocok dengan `hasil_2kelas/recall_breakdown.csv` (dicek otomatis). File: `class_agnostic/rincian_per_kelas.csv`.

| Model | Kelas GT | Objek | Kelas benar (%) | Kelas salah (%) | Tidak terdeteksi (%) |
|---|---|---|---|---|---|
| YOLOv8l | lightweight litter | 1002 | 71.96 | 2.10 | 25.95 |
| YOLOv8l | metal and glass | 129 | 56.59 | 26.36 | 17.05 |
| YOLOv12l | lightweight litter | 1002 | 71.46 | 1.50 | 27.05 |
| YOLOv12l | metal and glass | 129 | 55.81 | 26.36 | 17.83 |
| YOLO26l | lightweight litter | 1002 | 72.85 | 3.19 | 23.95 |
| YOLO26l | metal and glass | 129 | 58.91 | 27.13 | 13.95 |

Masalah kedua kelas berbeda: metal and glass lebih sering salah kelas, lightweight litter lebih sering tidak terdeteksi (24.0 sampai 27.0%, sedangkan metal and glass 14.0 sampai 17.8%).

## 3. Hasil keempat verifikasi

**a. Standar vs Ultralytics: lulus.** Selisih maksimum 1.67e-16 (batas 0.01). Fungsi pencocokan script juga menghasilkan matriks TP yang persis sama dengan validator: 0 sel berbeda dari 1131690 sel. File: `verifikasi_a_standar_vs_ultralytics.csv`.

**b. Cek silang `single_cls=True`: 10 model NMS lulus, 5 model YOLO26 melewati batas dan penyebabnya sudah ditemukan.**

| Model | Validator | Pred single_cls | Pred v3 | Pred v4 | mAP@50 | mAP@50-95 | Selisih maks vs v3 | Selisih maks vs v4 |
|---|---|---|---|---|---|---|---|---|
| YOLOv8n | NMS agnostic | 12947 | 12790 | 15313 | 0.751415 | 0.410924 | 0.000036 | 0.007189 |
| YOLOv8s | NMS agnostic | 6675 | 6674 | 8139 | 0.795897 | 0.457463 | 0.000004 | 0.005343 |
| YOLOv8m | NMS agnostic | 6204 | 6203 | 7574 | 0.780997 | 0.450081 | 0.000000 | 0.005746 |
| YOLOv8l | NMS agnostic | 4394 | 4394 | 5492 | 0.785086 | 0.449674 | 0.000002 | 0.006298 |
| YOLOv8x | NMS agnostic | 4580 | 4579 | 5673 | 0.770152 | 0.442965 | 0.000015 | 0.007144 |
| YOLOv12n | NMS agnostic | 9760 | 9720 | 11585 | 0.756750 | 0.428031 | 0.000019 | 0.004754 |
| YOLOv12s | NMS agnostic | 5840 | 5840 | 7109 | 0.793325 | 0.458719 | 0.000078 | 0.005894 |
| YOLOv12m | NMS agnostic | 5243 | 5239 | 6284 | 0.796811 | 0.465517 | 0.000123 | 0.004351 |
| YOLOv12l | NMS agnostic | 6164 | 6136 | 7442 | 0.789376 | 0.459055 | 0.000015 | 0.007972 |
| YOLOv12x | NMS agnostic | 5740 | 5672 | 6971 | 0.782996 | 0.455406 | 0.000033 | 0.010186 |
| YOLO26n | end-to-end, tanpa NMS | 9035 | 6637 | 9035 | 0.769546 | 0.443398 | 0.011550 | 0.000000 |
| YOLO26s | end-to-end, tanpa NMS | 6130 | 4503 | 6130 | 0.785119 | 0.460898 | 0.008909 | 0.000000 |
| YOLO26m | end-to-end, tanpa NMS | 6181 | 4508 | 6181 | 0.794652 | 0.463426 | 0.009546 | 0.000000 |
| YOLO26l | end-to-end, tanpa NMS | 4888 | 3484 | 4888 | 0.786073 | 0.467827 | 0.009725 | 0.000000 |
| YOLO26x | end-to-end, tanpa NMS | 5353 | 3833 | 5353 | 0.778900 | 0.458675 | 0.010826 | 0.000000 |

- YOLOv8 dan YOLOv12: selisih terhadap versi 3 maksimum 0.000123. Jumlah prediksi bisa sedikit berbeda karena versi 3 menjalankan NMS tanpa kelas pada keluaran yang sudah melewati NMS per kelas, sedangkan validator langsung NMS tanpa kelas dari kandidat awal. Satu kotak yang sudah hilang di tahap pertama bisa berbeda efeknya pada rantai penekanan.
- YOLO26: selisih terhadap versi 3 sampai 0.011550, melewati 0.05. Penyebabnya, `non_max_suppression` langsung mengembalikan keluaran kalau `end2end=True`, tanpa NMS, sehingga `single_cls=True` hanya mengganti label kelas. Buktinya, selisih terhadap versi 4 tepat 0.000000 dan jumlah prediksinya sama persis. Jadi versi 3 pada YOLO26 tidak bisa direproduksi hanya dengan `single_cls=True`.
- Penulisan di paper: versi 3 berarti NMS tanpa kelas dijalankan setelah prediksi disatukan, untuk semua model. Kalau hanya memakai `single_cls=True`, YOLO26 akan dinilai tanpa NMS, sedangkan model lain dengan NMS, sehingga tidak sebanding.

Apa yang dihapus NMS ulang:

| Model | Prediksi | Setelah NMS ulang | Dihapus, ada penekan kelas beda | Dihapus, hanya kelas sama |
|---|---|---|---|---|
| YOLOv8n | 15313 | 12790 | 2523 | 0 |
| YOLOv8s | 8139 | 6674 | 1465 | 0 |
| YOLOv8m | 7574 | 6203 | 1371 | 0 |
| YOLOv8l | 5492 | 4394 | 1098 | 0 |
| YOLOv8x | 5673 | 4579 | 1094 | 0 |
| YOLOv12n | 11585 | 9720 | 1865 | 0 |
| YOLOv12s | 7109 | 5840 | 1269 | 0 |
| YOLOv12m | 6284 | 5239 | 1045 | 0 |
| YOLOv12l | 7442 | 6136 | 1306 | 0 |
| YOLOv12x | 6971 | 5672 | 1299 | 0 |
| YOLO26n | 9035 | 6637 | 1258 | 1140 |
| YOLO26s | 6130 | 4503 | 896 | 731 |
| YOLO26m | 6181 | 4508 | 937 | 736 |
| YOLO26l | 4888 | 3484 | 773 | 631 |
| YOLO26x | 5353 | 3833 | 853 | 667 |

Pada YOLOv8 dan YOLOv12 semua yang dihapus punya penekan dengan kelas berbeda, karena NMS per kelas sudah menghapus duplikat sekelas. Pada YOLO26 sekitar separuhnya hanya berdampingan dengan kotak sekelas, artinya model end-to-end ini memang masih mengeluarkan kotak hampir sama dengan IoU di atas 0.. Efek NMS ulang di mAP@50: YOLOv8 0.53 sampai 0.72, YOLOv12 0.42 sampai 1.02, YOLO26 0.89 sampai 1.16 poin. Di mAP@50-95 efeknya 0.06 sampai 0.29 poin. Di mAP@75, NMS ulang justru menurunkan nilai pada 11 dari 15 model (terendah -0.55 poin). Kemungkinan penyebabnya, kotak dengan confidence lebih tinggi menekan kotak yang lokalisasinya lebih pas, dan ini paling terasa pada ambang IoU tinggi; penyebab ini belum saya uji terpisah.

**c. Contoh kecil: perilaku dan hitungan manual lulus, tetapi angka harapan di permintaan tidak cocok dengan rumus terpasang.**

| Pengecekan | Manual/harapan | Script | Cocok |
|---|---|---|---|
| P1 di standar (IoU 0.5) | TP | TP | True |
| P2 di standar (IoU 0.5) | FP | FP | True |
| P2 di class-agnostic (IoU 0.5) | TP | TP | True |
| P3 di class-agnostic (IoU 0.5) | FP | FP | True |
| NMS ulang mempertahankan ketiga prediksi | 3 | 3 | True |
| mAP@50 standar (v1): script vs manual rumus terpasang | 0.2475 | 0.2475 | True |
| mAP@50 pooled class-aware (v2): script vs manual rumus terpasang | 0.335 | 0.335 | True |
| mAP@50 class-agnostic (v3): script vs manual rumus terpasang | 0.665 | 0.665 | True |
| mAP@50 class-agnostic tanpa NMS (v4): script vs manual rumus terpasang | 0.665 | 0.665 | True |

P2 dihitung FP di standar dan TP di class-agnostic, sesuai harapan. Hitungan manual tidak memakai fungsi script:
- Rumus terpasang (`compute_ap` Ultralytics 8.4.123, titik penutup `mrec=[0, r..., r_akhir, 1]`, `mpre=[1, p..., 0, 0]`). Precision jatuh ke 0 tepat setelah recall maksimum.
  - Class-agnostic: recall maksimum 2/3. Nilai 1 pada x = 0 sampai 0.6, lalu 0 mulai x = 0.7. Trapezoid: 0.6 + 0.05 = **0.65**.
  - Standar: kelas ringan (2 GT) recall maksimum 0., AP = 0.95. Kelas logam tanpa prediksi, AP = 0. Rata-rata = **0.475**.
- Rumus lama (`mrec=[0, r..., 1]`, `mpre=[1, p..., 0]`), dengan garis miring dari titik terakhir ke (1, 0): standar **0.29** dan class-agnostic **0.7772**. Inilah sumber angka 0.9 dan 0.8.

Saya tetap memakai rumus terpasang karena verifikasi a mensyaratkan hasil sama dengan Ultralytics, dan seluruh `hasil_2kelas` memakainya. Pengaruh rumus lama pada ke-15 model (`efek_rumus_ap_lama.csv`): rumus lama lebih tinggi 0.39 sampai 1.98 poin di mAP@50, 0.77 sampai 3.89 poin di mAP@75, dan 0.52 sampai 2.76 poin di mAP@50-95. Ini penting saat membandingkan dengan paper lain yang memakai Ultralytics versi lama. Versi berapa rumus ini berubah tidak saya pastikan.

**d. Versi 3 lebih besar atau sama dengan versi 2: lulus.** 0 dari 45 kombinasi model dan metrik lebih rendah. Selisih terkecil 3.13 poin. File: `verifikasi_d_v3_vs_v2.csv`.

## 4. Bagian B: pelengkap

### 4.1 p-value dan Holm untuk bootstrap `hasil_2kelas` (mAP standar)

p dua sisi dari distribusi bootstrap yang sudah ada (CI-nya dicek sama dengan `bootstrap.csv`). File: `pelengkap/bootstrap_pvalue_holm.csv` dan `pelengkap/bootstrap_holm_ringkasan.csv`.

| Metrik | Perbandingan | p<0.05 tanpa koreksi | Holm keluarga 15 | Holm keluarga 30 | Holm 15 (+1) | Holm 30 (+1) | p terkecil | Holm 15 terkecil | Holm 30 terkecil |
|---|---|---|---|---|---|---|---|---|---|
| mAP50 | 15 | 0 | 0 | 0 | 0 | 0 | 0.120 | 1.000 | 1.000 |
| mAP50_95 | 15 | 1 | 0 | 0 | 0 | 0 | 0.032 | 0.480 | 0.960 |

**Tidak ada perbandingan per skala yang tetap signifikan setelah koreksi Holm**, pada semua varian. Yang terkecil adalah YOLO26n vs YOLOv8n di mAP@50-95 (+2.35 poin, p = 0.032, Holm 15 = 0.48, Holm 30 = 0.96). Permintaan "30 perbandingan, terpisah untuk mAP@50 dan mAP@50-95" bisa dibaca dua cara, jadi saya menghitung keduanya: keluarga 15 per metrik dan keluarga 30 gabungan. Hasilnya sama.

### 4.2 Batch dan patience

Batas "batch kecil" yang saya pakai adalah batch 4 atau kurang; ini batas saya untuk penandaan, bukan standar. File: `pelengkap/batch_dan_patience.csv`.

| Model | Batch | Asal batch | Akumulasi | Patience args.yaml | Patience sesi pertama (log) | Tanda |
|---|---|---|---|---|---|---|
| YOLOv8n | 16 | ditetapkan manual | 4 | 50 | 50 |  |
| YOLOv8s | 16 | ditetapkan manual | 4 | 50 | 50 |  |
| YOLOv8m | 11 | AutoBatch | 6 | 50 | 50 |  |
| YOLOv8l | 5 | AutoBatch | 13 | 50 | 50 |  |
| YOLOv8x | 3 | AutoBatch | 21 | 50 | 50 | skala x, batch kecil |
| YOLOv12n | 23 | AutoBatch | 3 | 100 | 100 |  |
| YOLOv12s | 12 | AutoBatch | 5 | 100 | 100 |  |
| YOLOv12m | 6 | AutoBatch | 11 | 100 | 100 |  |
| YOLOv12l | 3 | AutoBatch | 21 | 50 | 100 | batch kecil |
| YOLOv12x | 2 | AutoBatch | 32 | 50 | 50 | skala x, batch kecil |
| YOLO26n | 27 | AutoBatch | 2 | 50 | 50 |  |
| YOLO26s | 14 | AutoBatch | 5 | 50 | 50 |  |
| YOLO26m | 7 | AutoBatch | 9 | 50 | 50 |  |
| YOLO26l | 4 | AutoBatch | 16 | 50 | 50 | batch kecil |
| YOLO26x | 2 | AutoBatch | 32 | 50 | 50 | skala x, batch kecil |

Semua model skala x mendapat batch kecil: YOLOv8x 3, YOLOv12x 2, YOLO26x 2. YOLOv12l (3) dan YOLO26l (4) juga kecil, dan keduanya model terbaik di family masing-masing. Akumulasi gradien menjaga ukuran batch nominal sekitar 64, tetapi batch normalization tetap melihat batch kecil. Data ini tidak bisa membuktikan bahwa batch kecil memengaruhi hasil. YOLOv8n dan YOLOv8s memakai batch 16 yang ditetapkan manual, sisanya AutoBatch. YOLOv12n/s/m memakai patience 100, dan YOLOv12l memakai 100 sampai epoch 46 lalu 50 setelah resume.

### 4.3 Citra yang diabaikan Ultralytics

Dari `labels.cache` di `dataset-2kelas` dan `train_2kelas.log`, keduanya cocok, dan dihitung ulang dari file label. File: `pelengkap/label_diabaikan_per_split.csv` dan `pelengkap/label_diabaikan_daftar.csv`.

| Split | Citra | Diabaikan | Dipakai | Objek yang ikut hilang | Alasan |
|---|---|---|---|---|---|
| train | 8409 | **54** | 8355 | 168 lightweight, 39 metal | polygon dan box bercampur |
| valid | 595 | **3** | 592 | 14 lightweight, 2 metal | polygon dan box bercampur |
| test | 550 | 2 kalau memakai `yolo val` biasa; 0 di evaluasi ini | tidak dipakai training | 4 lightweight, 0 metal | polygon dan box bercampur; dikonversi di salinan |

Tidak ada alasan lain: tidak ada label hilang, gambar rusak, atau pesan duplikat di cache. Per sumber (aturan lama): test: TACO 2; train: Burangrang 18, TACO 30, UAVVaste 6; valid: TACO 3. Selain itu, 4964 label train, 469 valid, dan 242 test seluruhnya polygon; label ini tidak diabaikan, melainkan diubah Ultralytics menjadi kotak pembungkus.

### 4.4 Pola nama file dan sumber

Aturan lengkap: `sumber/aturan_sumber.txt`. Pola per split dan jumlah objek: `sumber/pola_nama_file.csv` dan `sumber/bukti_nama_TACO_UAVVaste.csv`.

| Pola | Aturan lama | Aturan usulan | Train | Valid | Test | Cocok TACO | Cocok UAVVaste | Tidak cocok |
|---|---|---|---|---|---|---|---|---|
| WhatsApp-Video-#-#-#-at-#-#-#_frame_# | Burangrang | Burangrang | 2052 | 68 | 200 | 0 | 0 | 2320 |
| IMG_#_JPG | Burangrang | Burangrang | 1116 | 29 | 100 | 0 | 0 | 1245 |
| batch_s#_img_# | UAVVaste | UAVVaste | 948 | 101 | 54 | 0 | 1103 | 0 |
| # | TACO | TACO | 963 | 84 | 33 | 1080 | 0 | 0 |
| BATCH_d#_img_# | UAVVaste | UAVVaste | 891 | 86 | 55 | 0 | 1032 | 0 |
| batch_#_img_# | UAVVaste | UAVVaste | 786 | 82 | 42 | 0 | 910 | 0 |
| BATCH_s#_img_# | UAVVaste | UAVVaste | 735 | 60 | 36 | 0 | 831 | 0 |
| IMG_# | Burangrang | TACO | 504 | 41 | 13 | 558 | 0 | 0 |
| batch_#_frame_# | UAVVaste | UAVVaste | 189 | 19 | 8 | 0 | 216 | 0 |
| GOPR# | Burangrang | UAVVaste | 183 | 20 | 6 | 0 | 209 | 0 |
| camera_img_# | Unknown | UAVVaste | 18 | 1 | 3 | 0 | 22 | 0 |
| photo_#-#-#_#-#-# | Unknown | UAVVaste | 18 | 2 | 0 | 0 | 20 | 0 |
| DJI_# | Unknown | UAVVaste | 6 | 2 | 0 | 0 | 8 | 0 |

Anotasi diunduh pada 2026-11-07 dari `pedropro/TACO` (`data/annotations.json`) dan `UAVVaste/UAVVaste` (`annotations/annotations.json`). Batasannya:
- Yang dicocokkan adalah nama file, bukan isi citra.
- Di TACO, 1500 citra hanya punya 239 nama unik karena nama angka berulang antar batch, jadi kecocokan pola angka adalah bukti lemah. Pola `IMG_#` dan semua pola UAVVaste lebih kuat, dan pola `IMG_#` juga didukung tampilan citranya.
- Burangrang pada aturan usulan disimpulkan dengan eliminasi. Hanya kamu yang bisa memastikan `WhatsApp-Video` dan `IMG_####_JPG_jpg` benar-benar dari Gunung Burangrang.

Jumlah data dengan aturan usulan (aturan lama juga ada di `sumber/jumlah_per_sumber_dua_aturan.csv`):

| Sumber | Split | Citra | Tanpa objek | Lightweight | Metal |
|---|---|---|---|---|---|
| Burangrang | test | 300 | 232 | 207 | 1 |
| Burangrang | train | 3168 | 2223 | 2635 | 10 |
| Burangrang | valid | 97 | 77 | 69 | 0 |
| TACO | test | 46 | 1 | 80 | 17 |
| TACO | train | 1467 | 39 | 2756 | 746 |
| TACO | valid | 125 | 10 | 253 | 53 |
| UAVVaste | test | 204 | 5 | 715 | 111 |
| UAVVaste | train | 3774 | 139 | 12223 | 2431 |
| UAVVaste | valid | 373 | 11 | 1419 | 229 |

Recall model terbaik per sumber, conf 0.25, IoU 0.5, dalam persen (`sumber/recall_per_sumber_dua_aturan.csv`, termasuk per kelas dan per pola):

| Aturan | Sumber | YOLOv8l | YOLOv12l | YOLO26l |
|---|---|---|---|---|
| aturan_lama | Burangrang | 42.86 | 48.41 | 44.44 |
| aturan_lama | TACO | 69.86 | 72.60 | 71.23 |
| aturan_lama | UAVVaste | 78.95 | 76.21 | 79.70 |
| aturan_lama | Unknown | 33.33 | 33.33 | 66.67 |
| aturan_usulan | Burangrang | 40.87 | 45.19 | 41.35 |
| aturan_usulan | TACO | 69.07 | 72.16 | 71.13 |
| aturan_usulan | UAVVaste | 77.72 | 75.54 | 78.81 |

Recall Burangrang dengan aturan usulan menjadi 40.9 sampai 45.2%, sedangkan `summary.md` menulis 42.9 sampai 48.4% dengan aturan lama. Angka Burangrang ini hampir seluruhnya mewakili kelas lightweight litter.

Contact sheet ada tiga, semuanya memakai 24 nama dasar acak dengan seed 0, sehingga tidak terpakai untuk augmentasi dari foto yang sama. Dua yang terakhir saya tambahkan karena di sanalah keraguannya:
- `sumber/contact_sheet_IMG.png`: yang diminta. 6 dari 24 berpola `IMG_#`.
- `sumber/contact_sheet_IMG_pola_tanpa_akhiran_JPG.png`: khusus pola `IMG_#` yang cocok dengan TACO.
- `sumber/contact_sheet_GOPR.png`: khusus GOPR yang cocok dengan UAVVaste.

Pengamatan visual saya pada lembar pertama: 18 citra `IMG_#_JPG` terlihat seperti tanah jalur, akar, dan vegetasi, sedangkan beberapa citra `IMG_#` terlihat seperti lantai atau meja. Keputusan akhir tetap ada padamu.

### 4.5 Contoh dataset

- `contoh_dataset/utuh/`: 4 citra utuh per sumber, resolusi asli, dengan kotak ground truth (lightweight litter #1baf7a, metal and glass #eda100).
- `contoh_dataset/crop/`: 8 crop per kelas, resolusi asli, margin 10%, ukuran dari 16 sampai 501 piksel lebarnya. Komposisi: lightweight litter: Burangrang 5, TACO 3; metal and glass: Burangrang 1, TACO 4, UAVVaste 3.
- `contoh_dataset/daftar_contoh.csv`: file asal setiap contoh.

Semua diambil dari split test, sumbernya memakai **aturan usulan** karena aturan lama terbukti keliru. Karena data sendiri nyaris tidak punya metal and glass, 7 dari 8 crop kelas itu berasal dari UAVVaste dan TACO. Sudut hitam pada sebagian citra adalah bawaan dataset (rotasi dari Roboflow), bukan hasil skrip. Kalau aturan usulan tidak kamu setujui, sumber di daftar ini perlu dipilih ulang.

## 5. Hal lain yang aneh atau perlu dicatat

- **Tumpang tindih test-train tetap berlaku.** Sesuai instruksi terakhir, analisisnya tidak dikerjakan di sini. Semua angka di laporan ini memakai 550 citra, termasuk 186 citra yang terdaftar tumpang tindih di `hasil_2kelas/summary.md`, jadi peringatan di sana berlaku juga untuk class-agnostic mAP.
- **Pembagian tumpang tindih per sumber di `summary.md` memakai aturan lama** (Burangrang 15, UAVVaste 144, TACO 25, Unknown 2). Dengan aturan usulan menjadi Burangrang 0, UAVVaste 151, TACO 35. Total tetap 186; kolomnya ada di `sumber/bukti_nama_TACO_UAVVaste.csv`. Artinya, tidak ada citra Burangrang di test yang masuk daftar tumpang tindih. Daftar itu hanya memeriksa citra dengan nama dasar yang sama, jadi tetap batas bawah. Kalimat di `summary.md` tentang Burangrang juga mengikuti aturan lama.
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
2. **Bagaimana menulis kontribusi data sendiri?** Dataset ini punya 1 objek metal and glass Burangrang di test, jadi klaim per kelas pada jalur gunung tidak didukung data.
3. **Keluarga Holm mana yang dipakai** untuk versi 3: saya sarankan 15 per metrik dengan p +1.

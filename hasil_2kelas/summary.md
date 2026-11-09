# Hasil evaluasi dataset 2 kelas: YOLOv8, YOLOv12, YOLO26

Semua angka berasal dari script di `scripts/` yang benar-benar dijalankan, dan tabel di dokumen ini dibuat otomatis dari CSV di folder yang sama. Kelas: `kertas_plastik_puntungRokok` = *lightweight litter*, `logam_kaleng_kaca` = *metal and glass*.

## 1. Baca ini dulu: empat hal yang mempengaruhi cara menulis paper

**a. Test set tumpang tindih dengan train (temuan paling serius).** 186 dari 550 gambar test (33.8%) terbukti berasal dari foto sumber yang sama dengan gambar di train, hanya beda augmentasi Roboflow. Di valid 392 dari 595 (65.9%), padahal valid dipakai memilih `best.pt` dan early stopping. Buktinya pencocokan fitur ORB + RANSAC antar gambar dengan nama dasar sama (median inlier 1206; kontrol: nilai tertinggi dari 8 pasangan acak per gambar, bermedian 5), lalu saya cek langsung tiga pasangan secara visual. Ambangnya konservatif (lebih dari 397 inlier, nilai kontrol tertinggi), dan gambar dengan nama berbeda tidak dicek, jadi 186 adalah batas bawah. Per sumber di test: UAVVaste 144/195, TACO 25/33, Burangrang 15/319, Unknown 2/3. Daftar lengkap: `tumpang_tindih_train.csv`.

Tanpa 186 gambar itu, mAP@50-95 turun 7.4 sampai 11.3 poin persen dan mAP@50 turun 6.3 sampai 10.7 poin persen (`metrics_test_tanpa_tumpang_tindih.csv`, angka pelengkap, angka utama tetap 550 gambar sesuai permintaan). Penurunan ini tidak seluruhnya akibat kebocoran: sisa 364 gambar didominasi Burangrang (304 gambar) yang memang lebih sulit. Jadi angka ini menunjukkan besarnya kepekaan, bukan performa yang sudah terkoreksi.

Klaim performa dari 550 gambar berisiko dibaca terlalu optimistis kalau hal ini tidak disebut. Yang bisa dilakukan: (1) melaporkan kedua angka dan menyebut tumpang tindihnya di paper, tanpa training ulang; atau (2) menyusun ulang split berdasarkan foto sumber lalu melatih ulang, yang paling bersih tetapi butuh sekitar 152 jam GPU kalau 15 model diulang semua. Ini keputusan kamu.

**b. Protokol training belum sepenuhnya sama.** Epochs 200, imgsz 512, optimizer, augmentasi, dan seed seragam. Yang berbeda: patience (YOLOv12n/s/m memakai 100) dan batch (2 sampai 27). Simulasi dari `results.csv`: patience 100 tidak mengubah apa pun untuk YOLOv12n dan YOLOv12s, tetapi **YOLOv12m seharusnya berhenti di epoch 166 kalau memakai patience 50**, sedangkan `best.pt`-nya dari epoch 199. Jadi kalimat "protokol yang sama" belum akurat untuk YOLOv12m tanpa catatan. Rincian di bagian 6.

**c. 550 gambar bukan angka default Ultralytics.** Ultralytics membuang 2 gambar test (dan 3 valid, 54 train) karena labelnya mencampur poligon dan box. Supaya 550 gambar benar-benar dinilai sesuai permintaan, label kedua gambar test itu dikonversi ke box di salinan `_data/` (dataset asli tidak diubah), persis seperti cara Ultralytics memperlakukan label poligon. Pengaruhnya sangat kecil, tetapi `yolo val` biasa akan menghasilkan angka 548 gambar. Kedua versi ada di `metrics_test.csv` (kolom `*_548`). Training sendiri berjalan dengan 8355 gambar train dan 592 valid, bukan 8409 dan 595.

**d. Aturan sumber dari nama file perlu dikonfirmasi.** Aturannya (dari `scripts/fase1_scene_groups.py`): `batch_*` = UAVVaste, nama angka = TACO, `IMG_*`/`GOPR*`/`WhatsApp-Video*` = Burangrang. Tetapi `IMG_4879` (contoh (b) di `samples/`) adalah foto wadah plastik di dalam ruangan, jadi jelas bukan foto jalur pendakian. Angka per sumber hanya seakurat aturan ini.

## 2. Tabel utama (test, 550 gambar)

imgsz 512, batch 16, conf 0.001, IoU 0.7. Latensi: batch 1, rata-rata 3 ulangan. Params dan GFLOPs dihitung pada model setelah fuse (model yang benar-benar dipakai inferensi). Nilai tertinggi ditebalkan. Sumber: `metrics_test.csv`, `speed.csv`.

| Model | Params (M) | GFLOPs | mAP@50 (%) | mAP@75 (%) | mAP@50-95 (%) | P (%) | R (%) | ms/gambar | FPS |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 3.01 | 5.17 | 62.07 | 34.83 | 34.89 | 68.50 | 60.14 | 6.79 | **147.4** |
| YOLOv8s | 11.13 | 18.20 | 64.82 | 38.99 | 38.93 | 73.47 | 60.75 | 6.84 | 146.3 |
| YOLOv8m | 25.84 | 50.36 | 64.38 | 39.92 | 39.29 | 66.23 | 66.20 | 8.17 | 122.4 |
| YOLOv8l | 43.61 | 105.48 | 66.04 | 42.12 | 39.88 | 79.16 | 58.73 | 11.34 | 88.2 |
| YOLOv8x | 68.13 | 164.73 | 64.48 | 41.10 | 38.62 | 73.62 | 62.78 | 16.11 | 62.1 |
| YOLOv12n | 2.56 | 4.45 | 61.39 | 36.09 | 35.95 | 71.72 | 59.43 | 11.51 | 86.9 |
| YOLOv12s | 9.23 | 14.38 | 64.95 | 39.42 | 39.43 | 72.10 | 62.61 | 11.74 | 85.2 |
| YOLOv12m | 20.11 | 44.29 | 64.61 | 42.08 | 39.34 | 70.69 | 62.85 | 12.88 | 77.7 |
| YOLOv12l | 26.34 | 59.35 | **66.58** | **44.67** | **40.67** | 74.05 | 63.40 | 20.80 | 48.1 |
| YOLOv12x | 59.05 | 131.08 | 66.23 | 40.58 | 40.01 | 76.00 | 62.32 | 20.85 | 48.0 |
| YOLO26n | 2.38 | 3.37 | 62.40 | 36.64 | 37.23 | 73.14 | 58.78 | 9.04 | 110.7 |
| YOLO26s | 9.47 | 13.23 | 63.65 | 40.94 | 38.91 | 70.54 | 61.42 | 9.45 | 105.8 |
| YOLO26m | 20.35 | 43.52 | 64.58 | 40.70 | 39.31 | 72.06 | 61.36 | 10.64 | 94.0 |
| YOLO26l | 24.75 | 55.25 | 64.54 | 42.64 | 39.81 | 72.13 | 63.12 | 14.75 | 67.8 |
| YOLO26x | 55.64 | 123.98 | 64.09 | 41.12 | 39.66 | 74.15 | 62.55 | 16.05 | 62.3 |

### Rincian latensi (`speed.csv`)

| Model | Pre | Inference | Post | Total | FPS | Total, conf 0.25 | Post, conf 0.25 |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 0.55 | 5.50 | 0.74 | 6.79 | 147.4 | 6.72 | 0.63 |
| YOLOv8s | 0.56 | 5.57 | 0.71 | 6.84 | 146.3 | 6.94 | 0.66 |
| YOLOv8m | 0.56 | 6.92 | 0.70 | 8.17 | 122.4 | 8.22 | 0.65 |
| YOLOv8l | 0.60 | 10.04 | 0.70 | 11.34 | 88.2 | 11.26 | 0.65 |
| YOLOv8x | 0.60 | 14.78 | 0.73 | 16.11 | 62.1 | 16.21 | 0.70 |
| YOLOv12n | 0.58 | 10.20 | 0.73 | 11.51 | 86.9 | 11.42 | 0.65 |
| YOLOv12s | 0.58 | 10.47 | 0.69 | 11.74 | 85.2 | 11.88 | 0.65 |
| YOLOv12m | 0.58 | 11.59 | 0.70 | 12.88 | 77.7 | 12.94 | 0.65 |
| YOLOv12l | 0.59 | 19.49 | 0.72 | 20.80 | 48.1 | 20.67 | 0.65 |
| YOLOv12x | 0.61 | 19.54 | 0.71 | 20.85 | 48.0 | 20.79 | 0.66 |
| YOLO26n | 0.57 | 8.30 | 0.17 | 9.04 | 110.7 | 9.48 | 0.17 |
| YOLO26s | 0.57 | 8.71 | 0.17 | 9.45 | 105.8 | 9.66 | 0.17 |
| YOLO26m | 0.58 | 9.88 | 0.17 | 10.64 | 94.0 | 10.53 | 0.17 |
| YOLO26l | 0.59 | 13.99 | 0.17 | 14.75 | 67.8 | 14.66 | 0.17 |
| YOLO26x | 0.59 | 15.27 | 0.18 | 16.05 | 62.3 | 16.03 | 0.18 |

## 3. Per kelas (test, `metrics_per_class.csv`)

| Model | Kelas | Gambar | Objek | P (%) | R (%) | AP@50 (%) | AP@75 (%) | AP@50-95 (%) |
|---|---|---|---|---|---|---|---|---|
| YOLOv8n | lightweight litter | 295 | 1002 | 76.25 | 67.56 | 70.77 | 36.51 | 38.49 |
| YOLOv8n | metal and glass | 81 | 129 | 60.76 | 52.71 | 53.37 | 33.16 | 31.28 |
| YOLOv8s | lightweight litter | 295 | 1002 | 83.08 | 69.56 | 74.84 | 42.22 | 42.52 |
| YOLOv8s | metal and glass | 81 | 129 | 63.86 | 51.94 | 54.79 | 35.75 | 35.34 |
| YOLOv8m | lightweight litter | 295 | 1002 | 78.95 | 71.16 | 73.46 | 41.83 | 41.82 |
| YOLOv8m | metal and glass | 81 | 129 | 53.51 | 61.24 | 55.31 | 38.01 | 36.76 |
| YOLOv8l | lightweight litter | 295 | 1002 | 86.58 | 66.30 | 73.65 | 40.68 | 41.71 |
| YOLOv8l | metal and glass | 81 | 129 | 71.75 | 51.16 | 58.42 | 43.56 | 38.06 |
| YOLOv8x | lightweight litter | 295 | 1002 | 81.83 | 67.43 | 71.70 | 41.43 | 40.56 |
| YOLOv8x | metal and glass | 81 | 129 | 65.40 | 58.14 | 57.25 | 40.77 | 36.68 |
| YOLOv12n | lightweight litter | 295 | 1002 | 81.20 | 67.69 | 71.54 | 39.77 | 40.05 |
| YOLOv12n | metal and glass | 81 | 129 | 62.24 | 51.16 | 51.24 | 32.41 | 31.84 |
| YOLOv12s | lightweight litter | 295 | 1002 | 83.62 | 70.96 | 74.51 | 42.43 | 42.64 |
| YOLOv12s | metal and glass | 81 | 129 | 60.57 | 54.26 | 55.38 | 36.40 | 36.23 |
| YOLOv12m | lightweight litter | 295 | 1002 | 82.36 | 72.21 | 74.27 | 44.69 | 42.93 |
| YOLOv12m | metal and glass | 81 | 129 | 59.01 | 53.49 | 54.94 | 39.47 | 35.75 |
| YOLOv12l | lightweight litter | 295 | 1002 | 81.82 | 70.99 | 73.76 | 43.50 | 42.47 |
| YOLOv12l | metal and glass | 81 | 129 | 66.28 | 55.81 | 59.39 | 45.84 | 38.88 |
| YOLOv12x | lightweight litter | 295 | 1002 | 83.88 | 68.05 | 73.57 | 42.39 | 42.23 |
| YOLOv12x | metal and glass | 81 | 129 | 68.12 | 56.59 | 58.89 | 38.77 | 37.79 |
| YOLO26n | lightweight litter | 295 | 1002 | 83.99 | 67.17 | 72.74 | 41.13 | 41.48 |
| YOLO26n | metal and glass | 81 | 129 | 62.29 | 50.39 | 52.05 | 32.15 | 32.99 |
| YOLO26s | lightweight litter | 295 | 1002 | 83.20 | 70.65 | 73.15 | 43.54 | 42.52 |
| YOLO26s | metal and glass | 81 | 129 | 57.88 | 52.19 | 54.14 | 38.35 | 35.31 |
| YOLO26m | lightweight litter | 295 | 1002 | 83.52 | 68.46 | 75.46 | 43.86 | 43.39 |
| YOLO26m | metal and glass | 81 | 129 | 60.60 | 54.26 | 53.70 | 37.55 | 35.24 |
| YOLO26l | lightweight litter | 295 | 1002 | 83.67 | 69.66 | 73.24 | 45.56 | 43.23 |
| YOLO26l | metal and glass | 81 | 129 | 60.59 | 56.59 | 55.84 | 39.72 | 36.39 |
| YOLO26x | lightweight litter | 295 | 1002 | 84.65 | 69.46 | 72.79 | 42.90 | 42.51 |
| YOLO26x | metal and glass | 81 | 129 | 63.65 | 55.64 | 55.38 | 39.34 | 36.82 |

## 4. Uji signifikansi (`bootstrap.csv`)

Paired percentile bootstrap level gambar, 1000 resample, seed 0, 550 gambar per resample, gambar yang sama untuk kedua model. mAP dihitung ulang dengan `ap_per_class` Ultralytics; sanity check terhadap validator: selisih maksimum 1.7e-16 (`sanity_check_map.csv`). Delta = A minus B, dalam poin persen.

| Jenis | A | B | Metrik | Delta | CI 95% bawah | CI 95% atas | Kesimpulan |
|---|---|---|---|---|---|---|---|
| skala | YOLO26n | YOLOv12n | mAP@50 | +1.01 | -1.68 | +3.30 | setara |
| skala | YOLO26n | YOLOv12n | mAP@50-95 | +1.29 | -0.27 | +2.72 | setara |
| skala | YOLO26n | YOLOv8n | mAP@50 | +0.33 | -3.16 | +3.44 | setara |
| skala | YOLO26n | YOLOv8n | mAP@50-95 | +2.35 | +0.24 | +4.60 | YOLO26n |
| skala | YOLOv12n | YOLOv8n | mAP@50 | -0.68 | -3.62 | +2.07 | setara |
| skala | YOLOv12n | YOLOv8n | mAP@50-95 | +1.06 | -0.82 | +2.88 | setara |
| skala | YOLO26s | YOLOv12s | mAP@50 | -1.30 | -3.81 | +1.25 | setara |
| skala | YOLO26s | YOLOv12s | mAP@50-95 | -0.52 | -2.47 | +1.41 | setara |
| skala | YOLO26s | YOLOv8s | mAP@50 | -1.17 | -4.29 | +1.37 | setara |
| skala | YOLO26s | YOLOv8s | mAP@50-95 | -0.02 | -2.22 | +1.97 | setara |
| skala | YOLOv12s | YOLOv8s | mAP@50 | +0.13 | -2.32 | +2.29 | setara |
| skala | YOLOv12s | YOLOv8s | mAP@50-95 | +0.51 | -1.01 | +1.95 | setara |
| skala | YOLO26m | YOLOv12m | mAP@50 | -0.03 | -2.92 | +2.66 | setara |
| skala | YOLO26m | YOLOv12m | mAP@50-95 | -0.02 | -1.90 | +2.13 | setara |
| skala | YOLO26m | YOLOv8m | mAP@50 | +0.19 | -2.90 | +2.73 | setara |
| skala | YOLO26m | YOLOv8m | mAP@50-95 | +0.03 | -2.04 | +1.81 | setara |
| skala | YOLOv12m | YOLOv8m | mAP@50 | +0.22 | -2.88 | +3.08 | setara |
| skala | YOLOv12m | YOLOv8m | mAP@50-95 | +0.05 | -2.09 | +1.71 | setara |
| skala | YOLO26l | YOLOv12l | mAP@50 | -2.03 | -4.48 | +0.36 | setara |
| skala | YOLO26l | YOLOv12l | mAP@50-95 | -0.87 | -2.47 | +0.87 | setara |
| skala | YOLO26l | YOLOv8l | mAP@50 | -1.49 | -3.99 | +0.99 | setara |
| skala | YOLO26l | YOLOv8l | mAP@50-95 | -0.08 | -1.77 | +1.86 | setara |
| skala | YOLOv12l | YOLOv8l | mAP@50 | +0.54 | -1.57 | +2.82 | setara |
| skala | YOLOv12l | YOLOv8l | mAP@50-95 | +0.79 | -0.53 | +2.22 | setara |
| skala | YOLO26x | YOLOv12x | mAP@50 | -2.15 | -5.19 | +1.08 | setara |
| skala | YOLO26x | YOLOv12x | mAP@50-95 | -0.34 | -2.40 | +1.68 | setara |
| skala | YOLO26x | YOLOv8x | mAP@50 | -0.39 | -3.56 | +2.48 | setara |
| skala | YOLO26x | YOLOv8x | mAP@50-95 | +1.04 | -1.08 | +3.11 | setara |
| skala | YOLOv12x | YOLOv8x | mAP@50 | +1.76 | -1.35 | +4.57 | setara |
| skala | YOLOv12x | YOLOv8x | mAP@50-95 | +1.38 | -0.67 | +3.17 | setara |
| terbaik | YOLO26l | YOLOv12l | mAP@50 | -2.03 | -4.48 | +0.36 | setara |
| terbaik | YOLO26l | YOLOv12l | mAP@50-95 | -0.87 | -2.47 | +0.87 | setara |
| terbaik | YOLO26l | YOLOv8l | mAP@50 | -1.49 | -3.99 | +0.99 | setara |
| terbaik | YOLO26l | YOLOv8l | mAP@50-95 | -0.08 | -1.77 | +1.86 | setara |
| terbaik | YOLOv12l | YOLOv8l | mAP@50 | +0.54 | -1.57 | +2.82 | setara |
| terbaik | YOLOv12l | YOLOv8l | mAP@50-95 | +0.79 | -0.53 | +2.22 | setara |

## 5. Temuan penting

- **Tidak ada model yang terbukti lebih unggul secara umum.** Dari 30 perbandingan per skala, hanya 1 yang CI-nya tidak melewati nol: YOLO26n vs YOLOv8n pada mAP@50-95 (+2.35 poin, CI +0.24 sampai +4.60). CI ini tidak dikoreksi untuk perbandingan berganda; pada taraf 5% sekitar 1.5 hasil seperti ini bisa muncul secara kebetulan, jadi jangan dijadikan klaim utama.
- Angka tertinggi di test: YOLOv12l (mAP@50 66.58%, mAP@75 44.67%, mAP@50-95 40.67%), tetapi selisihnya dengan model terbaik family lain tidak signifikan. Model terbaik tiap family: YOLOv8l, YOLOv12l, YOLO26l, semuanya skala l, sehingga perbandingan antar model terbaik sama dengan perbandingan skala l.
- Urutan tidak stabil antar split: di valid yang tertinggi YOLO26l (mAP@50-95 40.85%), sedangkan YOLOv12l di valid hanya 38.58%. Ini mendukung kesimpulan bahwa perbedaannya sebesar variasi sampel.
- Kecepatan (total, batch 1): YOLOv12 paling lambat di skala n, s, m, l, x. YOLOv8 lebih cepat dari YOLO26 di skala n, s, m, l; di skala x keduanya hampir sama (YOLOv8x 16.11 ms, YOLO26x 16.05 ms). YOLO26 berjalan tanpa NMS sehingga postprocess-nya sekitar 0.17 ms, dibanding sekitar 0.71 ms untuk YOLOv8/YOLOv12, tetapi selisih itu kecil dibanding waktu inferensi. Latensi antara skala n dan s nyaris sama, kemungkinan karena overhead per panggilan pada batch 1.
- Kelas *metal and glass* lebih lemah di semua model: AP@50 51.2 sampai 59.4%, dibanding *lightweight litter* 70.8 sampai 75.5%. Objeknya hanya 129 di test.
- **Burangrang paling sulit:** recall 42.9 sampai 48.4%, sedangkan UAVVaste 76.2 sampai 79.7%. Perlu dicatat, hanya 15 dari 319 gambar Burangrang yang tumpang tindih dengan train, dibanding 144 dari 195 untuk UAVVaste. Jadi selisih ini sebagian bisa berasal dari kebocoran, bukan murni tingkat kesulitan.
- Sebagian false positive tampaknya sampah yang tidak dilabeli: contoh (g) di `samples/` adalah gambar tanpa label, tetapi objek yang dideteksi ketiga model secara visual terlihat seperti sampah. Ini pengamatan visual dari satu contoh, bukan hasil audit label.

### Analisis error model terbaik (`confusion/`, `recall_breakdown.csv`)

Confusion matrix: conf 0.25, IoU 0.45 (default Ultralytics). Baris = prediksi, kolom = ground truth. Recall: conf 0.25, IoU 0.5, kelas harus benar.

**YOLOv8l**

|  | GT lightweight litter | GT metal and glass | GT background |
|---|---|---|---|
| pred lightweight litter | 718 | 35 | 140 |
| pred metal and glass | 36 | 74 | 24 |
| pred background | 248 | 20 | 0 |

**YOLOv12l**

|  | GT lightweight litter | GT metal and glass | GT background |
|---|---|---|---|
| pred lightweight litter | 719 | 40 | 122 |
| pred metal and glass | 24 | 67 | 22 |
| pred background | 259 | 22 | 0 |

**YOLO26l**

|  | GT lightweight litter | GT metal and glass | GT background |
|---|---|---|---|
| pred lightweight litter | 734 | 38 | 146 |
| pred metal and glass | 40 | 74 | 21 |
| pred background | 228 | 17 | 0 |

| Model | Dimensi | Kategori | Gambar | GT | TP | Terlewat | Salah kelas | Recall (%) |
|---|---|---|---|---|---|---|---|---|
| YOLOv8l | semua | semua |  | 1131 | 794 | 282 | 55 | 70.20 |
| YOLOv8l | ukuran | small |  | 909 | 636 | 226 | 47 | 69.97 |
| YOLOv8l | ukuran | medium |  | 196 | 135 | 56 | 5 | 68.88 |
| YOLOv8l | ukuran | large |  | 26 | 23 | 0 | 3 | 88.46 |
| YOLOv8l | sumber | UAVVaste | 195.0 | 803 | 634 | 120 | 49 | 78.95 |
| YOLOv8l | sumber | TACO | 33.0 | 73 | 51 | 19 | 3 | 69.86 |
| YOLOv8l | sumber | Burangrang | 319.0 | 252 | 108 | 141 | 3 | 42.86 |
| YOLOv8l | sumber | Unknown | 3.0 | 3 | 1 | 2 | 0 | 33.33 |
| YOLOv12l | semua | semua |  | 1131 | 788 | 294 | 49 | 69.67 |
| YOLOv12l | ukuran | small |  | 909 | 622 | 245 | 42 | 68.43 |
| YOLOv12l | ukuran | medium |  | 196 | 144 | 48 | 4 | 73.47 |
| YOLOv12l | ukuran | large |  | 26 | 22 | 1 | 3 | 84.62 |
| YOLOv12l | sumber | UAVVaste | 195.0 | 803 | 612 | 147 | 44 | 76.21 |
| YOLOv12l | sumber | TACO | 33.0 | 73 | 53 | 18 | 2 | 72.60 |
| YOLOv12l | sumber | Burangrang | 319.0 | 252 | 122 | 127 | 3 | 48.41 |
| YOLOv12l | sumber | Unknown | 3.0 | 3 | 1 | 2 | 0 | 33.33 |
| YOLO26l | semua | semua |  | 1131 | 806 | 258 | 67 | 71.26 |
| YOLO26l | ukuran | small |  | 909 | 646 | 206 | 57 | 71.07 |
| YOLO26l | ukuran | medium |  | 196 | 137 | 52 | 7 | 69.90 |
| YOLO26l | ukuran | large |  | 26 | 23 | 0 | 3 | 88.46 |
| YOLO26l | sumber | UAVVaste | 195.0 | 803 | 640 | 109 | 54 | 79.70 |
| YOLO26l | sumber | TACO | 33.0 | 73 | 52 | 15 | 6 | 71.23 |
| YOLO26l | sumber | Burangrang | 319.0 | 252 | 112 | 134 | 6 | 44.44 |
| YOLO26l | sumber | Unknown | 3.0 | 3 | 2 | 0 | 1 | 66.67 |

## 6. Inventaris run dan setting

| Model | Path | Epoch | Best | Early stop | Jam | Sesi | Simulasi stop patience 50 | Batch | Akumulasi | Patience |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | runs_2kelas/yolov8n_2kelas | 200 | 200 | tidak | 2.79 | 3 | - | 16 | 4 | 50 |
| YOLOv8s | runs_2kelas/yolov8s_2kelas | 200 | 166 | tidak | 4.61 | 1 | - | 16 | 4 | 50 |
| YOLOv8m | runs_2kelas/yolov8m_2kelas | 171 | 121 | ya | 3.80 | 1 | 171 | 11 | 6 | 50 |
| YOLOv8l | runs_2kelas/yolov8l_2kelas | 200 | 174 | tidak | 7.98 | 1 | - | 5 | 13 | 50 |
| YOLOv8x | runs_2kelas/yolov8x_2kelas | 200 | 187 | tidak | 12.97 | 1 | - | 3 | 21 | 50 |
| YOLOv12n | runs_2kelas/yolov12n_2kelas | 200 | 195 | tidak | 2.17 | 1 | - | 23 | 3 | 100 |
| YOLOv12s | runs_2kelas/yolov12s_2kelas | 200 | 190 | tidak | 3.70 | 1 | - | 12 | 5 | 100 |
| YOLOv12m | runs_2kelas/yolov12m_2kelas | 200 | 199 | tidak | 7.40 | 1 | 166 | 6 | 11 | 100 |
| YOLOv12l | runs_2kelas/yolov12l_2kelas | 200 | 196 | tidak | 19.59 | 2 | - | 3 | 21 | 50 |
| YOLOv12x | runs_2kelas/yolov12x_2kelas | 200 | 193 | tidak | 29.45 | 1 | - | 2 | 32 | 50 |
| YOLO26n | runs_2kelas/yolo26n_2kelas | 200 | 200 | tidak | 2.54 | 1 | - | 27 | 2 | 50 |
| YOLO26s | runs_2kelas/yolo26s_2kelas | 200 | 154 | tidak | 4.10 | 1 | - | 14 | 5 | 50 |
| YOLO26m | runs_2kelas/yolo26m_2kelas | 159 | 109 | ya | 6.06 | 1 | 159 | 7 | 9 | 50 |
| YOLO26l | runs_2kelas/yolo26l_2kelas | 200 | 161 | tidak | 15.67 | 1 | - | 4 | 16 | 50 |
| YOLO26x | runs_2kelas/yolo26x_2kelas | 200 | 163 | tidak | 29.07 | 1 | - | 2 | 32 | 50 |

Semua 15 run selesai dan memakai `dataset-2kelas/data.yaml` (hasil ekstrak `dataset_2class.zip`, jumlah file per split sama). Rincian lengkap: `runs_inventory.csv`, `settings_all_runs.csv`, `settings_berbeda.csv`.

**Sama di semua run:** epochs 200, imgsz 512, optimizer=auto yang di log tercatat `MuSGD(lr=0.01, momentum=0.9)`, lrf 0.01, weight_decay 0.0005 (nominal), warmup_epochs 3, mosaic 1.0, close_mosaic 10, mixup 0, hsv 0.015/0.7/0.4, fliplr 0.5, flipud 0, scale 0.5, translate 0.1, degrees/shear/perspective/copy_paste/cutmix 0, seed 0, deterministic, AMP, pretrained COCO, Ultralytics 8.4.123.

**Berbeda:**

- **patience:** YOLOv12n/s/m 100; YOLOv12l 100 sampai epoch 46 lalu 50 (simulasi: tidak berpengaruh); lainnya 50.
- **batch:** YOLOv8n/s tetap 16; sisanya AutoBatch dan hasilnya berbeda per model (2 sampai 27). Akumulasi gradien menyesuaikan ke nbs 64, tetapi efektifnya tidak persis sama, sehingga weight decay efektif 0.000422 sampai 0.000547. Model besar juga memperbarui statistik BatchNorm dengan batch kecil.
- **workers:** YOLOv8s 0, YOLOv8n 0 sampai epoch 67 lalu 4, lainnya 4. Ini pengaturan pemuatan data, bukan hyperparameter.
- **Training terputus:** YOLOv8n 3 sesi (lanjut di epoch 68 dan 75), YOLOv12l 2 sesi (lanjut di epoch 47).
- **Simulasi patience 50:** YOLOv8m dan YOLO26m memang berhenti sesuai simulasi (epoch 171 dan 159), jadi simulasinya tervalidasi. YOLOv12m akan berhenti di epoch 166 dengan epoch terbaik saat itu 116. Bobot epoch itu tidak tersimpan, jadi angka test-nya **TIDAK TERSEDIA** tanpa training ulang. Di valid, mAP@50-95 terbaik sampai epoch 166 adalah 39.23% vs 39.40% pada `best.pt`.
- **warmup_bias_lr** tertulis 0.1 di sebagian besar args.yaml dan 0.0 di YOLOv8n/YOLOv12l. Ini hanya artefak penulisan file: mode optimizer=auto mengubahnya menjadi 0.0 setelah args.yaml ditulis, dan `train_args` di semua `best.pt` memang 0.0. Jadi nilainya seragam.
- **batch di args.yaml:** tertulis -1 untuk AutoBatch; angka sebenarnya diambil dari `train_args` di `best.pt` dan sudah cocok dengan log.

**Run yang dikecualikan:** 32 folder berisi args.yaml, alasannya per folder ada di `runs_dikecualikan.csv`. Tidak ada run yang meragukan: run `New goji preapprove/runs/detect/train*` memakai TA-4 (4 kelas), dan `runs_2kelas/_dibatalkan_e100/yolov12s_2kelas` adalah percobaan 100 epoch yang dibatalkan di epoch 16 dan sudah digantikan.

```
ultralytics: 8.4.123
torch: 2.14.0+cu132
CUDA (torch.version.cuda): 13.2
cudnn: 92400
GPU: NVIDIA GeForce RTX 5070
nvidia-smi: NVIDIA GeForce RTX 5070, 610.88, 12227 MiB
python: 3.13.15 (C:\Users\User\.conda\envs\ultralytics_env\python.exe)
OS: Windows-11-10.0.26200-SP0
versi ultralytics di checkpoint best.pt: ['8.4.123']
```

## 7. Statistik dataset (`dataset_stats.csv`)

| Split | Gambar | Tanpa_objek | Lightweight | Metal | Label_campuran | UAVVaste | TACO | Burangrang | Unknown |
|---|---|---|---|---|---|---|---|---|---|
| train | 8409 | 2401 | 17614 | 3187 | 54 | 3549 | 963 | 3855 | 42 |
| valid | 595 | 98 | 1741 | 282 | 3 | 348 | 84 | 158 | 5 |
| test | 550 | 238 | 1002 | 129 | 2 | 195 | 33 | 319 | 3 |

Jumlah gambar sesuai target (8.409 / 595 / 550). Ukuran objek di test pada 512x512: small 909, medium 196, large 26; rincian per kelas ada di CSV. Menurut README Roboflow, seluruh gambar di-resize ke 514x514 dengan black edges dan auto-contrast. Train memakai 3 versi augmentasi per gambar sumber (flip, rotasi 90 derajat, crop 0-20%, rotasi +-15 derajat, shear +-10 derajat, brightness, exposure, noise), lalu ditambah augmentasi Ultralytics saat training.

## 8. Catatan cara ukur dan penyimpangan dari permintaan

Tidak ada langkah yang gagal. Hal yang perlu diketahui:

- **YOLO26 berjalan end-to-end tanpa NMS** pada mode default: `end2end=True` di validator, dan keluaran model berbentuk `[16, 300, 6]`, sedangkan YOLOv8/YOLOv12 memakai NMS (`[16, 6, 6069]`). Argumen iou=0.7 hanya berpengaruh pada YOLOv8/YOLOv12.
- **Params dan GFLOPs:** tabel utama memakai model setelah fuse. Untuk YOLO26 ini penting, karena sebelum fuse head one-to-many yang tidak dipakai inferensi ikut terhitung. Nilai sebelum fuse ada di kolom `*_tanpa_fuse`.
- **Jangan campur dengan `eval_2kelas/`.** Evaluasi lama memakai IoU NMS 0.6, 548 gambar, dan params sebelum fuse, jadi angkanya sedikit berbeda.
- **Confusion matrix:** Ultralytics 8.4 menurunkan threshold confusion matrix ke 0.001 kalau conf=0.001 diberikan eksplisit. Nilainya saya set ke 0.25 dan dicek otomatis terhadap jumlah prediksi dan GT.
- **Kecepatan:** permintaan tidak menyebut conf, jadi diukur pada conf 0.001 (utama, sama dengan Langkah 2) dan 0.25 (pelengkap); bedanya kecil. Waktu dari validator; preprocess hanya mencakup pindah ke GPU dan normalisasi, baca JPEG dan letterbox tidak terukur. GPU tidak eksklusif: aplikasi desktop (browser, VS Code, Wallpaper Engine) ikut memakai GPU yang sama. Selisih terbesar antar ulangan 0.70 ms (`speed_ulangan.csv`), jadi selisih kecil antar model jangan dibaca berlebihan.
- **Cross-check valid vs training (`metrics_val.csv`):** kolom `*_592` memakai 592 gambar seperti saat training. Selisih terhadap `results.csv` maksimal 0.31 poin (mAP@50) dan 0.12 poin (mAP@50-95), terbesar pada model n.
- **Contoh prediksi (`samples/samples.csv`):** dipilih otomatis dengan kriteria tertulis di `s6_figures.py`, mengutamakan gambar yang sumbernya tidak ada di train. Gambar di-crop di sekitar objek supaya terbaca di lebar satu kolom (kolom `crop`). Contoh (a) memuat kedua kelas; (b) hanya *lightweight litter*.
- **Gambar:** PNG 300 dpi, lebar 8,8 cm, font serif 7,5-8 pt, kecuali `ultralytics_*` di `confusion/` yang merupakan keluaran asli Ultralytics.

## 9. Isi folder

- `summary.md`, `environment.txt`
- Inventaris: `runs_inventory.csv`, `settings_all_runs.csv`, `settings_berbeda.csv`, `runs_dikecualikan.csv`
- Dataset: `dataset_stats.csv`, `label_campuran_diperbaiki.csv`, `tumpang_tindih_train.csv`
- Metrik: `metrics_test.csv`, `metrics_per_class.csv`, `metrics_val.csv`, `metrics_test_tanpa_tumpang_tindih.csv`, `sanity_check_map.csv`
- `speed.csv`, `speed_ulangan.csv`, `bootstrap.csv`, `recall_breakdown.csv`
- `figures/`, `curves/`, `confusion/`, `samples/`
- `scripts/` dan log `s*.log`. Urutan menjalankan dari folder `scripts/` dengan environment `ultralytics_env`: `s0_inventaris`, `s1_dataset`, `s2_eval`, `s3_speed`, `s4_bootstrap`, `s5_error`, `s6_figures`, `s7_summary`.
- Di disk saja, tidak ikut zip: `_cache/` (prediksi per gambar), `_data/` (salinan hardlink split), `_val/` (keluaran validator).

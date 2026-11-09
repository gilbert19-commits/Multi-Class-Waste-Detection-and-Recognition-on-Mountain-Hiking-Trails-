# Laporan Audit Dataset Sumber (TASK D0)
**Tanggal:** 2026-10-27 10:02  
**Cakupan:** `TA-1/` (non-augmented) train + valid + test, 3948 citra. Read-only, tidak ada file dataset yang diubah.

---

## 1. Instance per Kelas (Seluruh TA-1)

| kelas             |   instance |   citra_mengandung |   inst_train |   inst_valid |   inst_test |
|:------------------|-----------:|-------------------:|-------------:|-------------:|------------:|
| kertas_tisu       |       2172 |               1017 |         1506 |          433 |         233 |
| logam_kaleng_kaca |       1516 |                846 |         1105 |          282 |         129 |
| plastik           |       6750 |               2364 |         4676 |         1307 |         767 |
| puntung_rokok     |         89 |                 36 |           86 |            1 |           2 |

Format label: **9287** dari 10527 baris adalah poligon segmentasi (dikonversi ke bbox min/max seperti Ultralytics), sisanya bbox `xywh`.

Citra tanpa anotasi (background): **1119**. Total instance puntung_rokok di seluruh dataset: **89** (train 86, valid 1, test 2).

## 2. Asal Data

> Penentuan sumber berbasis **pola nama file** (heuristik, belum ada tagging manual — `manual_source_tags.csv` belum pernah diekspor): `batch_*`/`BATCH_*` = UAVVaste, nama numerik (`000005`) = TACO, `WhatsApp-Video*`/`IMG_*`/`GOPR*` = Burangrang. Unknown = DJI_#, camera_img_#, photo_#-#-#_#-#-#.

| sumber     |   citra |   nama_sumber_unik |   kertas_tisu |   logam_kaleng_kaca |   plastik |   puntung_rokok |
|:-----------|--------:|-------------------:|--------------:|--------------------:|----------:|----------------:|
| UAVVaste   |    1726 |                722 |          1712 |                1148 |      4441 |               0 |
| TACO       |     438 |                117 |           137 |                 253 |       705 |               2 |
| Burangrang |    1762 |                879 |           301 |                  92 |      1443 |              86 |
| Unknown    |      22 |                 12 |            22 |                  23 |       161 |               1 |

Catatan: angka citra adalah jumlah **file** (termasuk varian rotasi/duplikat Roboflow), bukan citra mentah unik.

## 3. Scene Group (SIFT + RANSAC)

- Kandidat pasangan: nama sumber sama + 3 frame/foto berurutan + 10 tetangga embedding ResNet50 + 5 tetangga dHash (flip-aware) → **52,370 pasangan diuji**.
- Kalibrasi 200 pasangan acak: inlier maks **8**, p95 **4**, median **0**.
- Pasangan adegan sama (inlier ≥ 100): **3,613**. Zona abu-abu (25–99): **1,979** pasangan, **1017** di antaranya menghubungkan group yang berbeda.
- Indikator recall: **709** dari 1613 nama sumber non-numerik masih terpecah ke >1 group (idealnya 0; sisa ini adalah varian yang tidak tertangkap SIFT).

**Total group: 1919** (ambang ≥100); 1525 bila zona abu-abu ikut digabung (konservatif).

| ukuran group   |   jumlah group (≥100) |   citra (≥100) |   jumlah group (≥25, konservatif) |
|:---------------|----------------------:|---------------:|----------------------------------:|
| 1              |                  1037 |           1037 |                               766 |
| 2-5            |                   831 |           2030 |                               687 |
| 6-10           |                    30 |            220 |                                43 |
| >10            |                    21 |            661 |                                29 |

### 10 Group Terbesar

|   group_id |   citra | split                                  | sumber             | contoh                                             |   kertas_tisu |   logam_kaleng_kaca |   plastik |   puntung_rokok |
|-----------:|--------:|:---------------------------------------|:-------------------|:---------------------------------------------------|--------------:|--------------------:|----------:|----------------:|
|       1367 |      91 | {'train': 53, 'valid': 14, 'test': 24} | {'Burangrang': 91} | WhatsApp-Video-2026-04-20-at-13-01-12_frame_000046 |             0 |                   0 |         8 |               0 |
|       1500 |      77 | {'train': 55, 'valid': 6, 'test': 16}  | {'Burangrang': 77} | WhatsApp-Video-2026-04-20-at-13-01-17_frame_000284 |             0 |                   0 |         0 |               0 |
|       1400 |      76 | {'train': 50, 'valid': 5, 'test': 21}  | {'Burangrang': 76} | WhatsApp-Video-2026-04-20-at-13-01-17_frame_000015 |             0 |                   0 |         0 |               0 |
|       1412 |      52 | {'train': 33, 'valid': 2, 'test': 17}  | {'Burangrang': 52} | WhatsApp-Video-2026-04-20-at-13-01-17_frame_000067 |             0 |                   0 |         0 |               0 |
|       1385 |      47 | {'train': 28, 'valid': 6, 'test': 13}  | {'Burangrang': 47} | WhatsApp-Video-2026-04-20-at-13-01-12_frame_000128 |             0 |                   0 |         0 |               0 |
|       1430 |      40 | {'train': 24, 'valid': 3, 'test': 13}  | {'Burangrang': 40} | WhatsApp-Video-2026-04-20-at-13-01-17_frame_000141 |             0 |                   0 |         0 |               0 |
|       1493 |      34 | {'train': 20, 'valid': 2, 'test': 12}  | {'Burangrang': 34} | WhatsApp-Video-2026-04-20-at-13-01-17_frame_000247 |             0 |                   0 |         0 |               0 |
|       1516 |      34 | {'train': 19, 'valid': 4, 'test': 11}  | {'Burangrang': 34} | WhatsApp-Video-2026-04-20-at-13-01-17_frame_000340 |             0 |                   0 |         0 |               0 |
|       1360 |      26 | {'train': 18, 'valid': 4, 'test': 4}   | {'Burangrang': 26} | WhatsApp-Video-2026-04-20-at-13-01-12_frame_000021 |             0 |                   0 |         0 |               0 |
|        444 |      25 | {'train': 13, 'valid': 12}             | {'UAVVaste': 25}   | BATCH_d07_img_1010                                 |            29 |                   3 |        60 |               0 |

### Leakage berbasis group (split Roboflow saat ini)

- Group yang tersebar di lebih dari satu split: **395**
- Citra di dalam group tersebut: train 1016, valid 384, test **358** dari 550

### Sebaran puntung_rokok per group

|   group_id |   instance puntung |   citra dgn puntung |   ukuran group | contoh nama                              |
|-----------:|-------------------:|--------------------:|---------------:|:-----------------------------------------|
|       1321 |                 16 |                   4 |              4 | IMG_5456_JPG, IMG_5457_JPG               |
|       1323 |                 11 |                   4 |              4 | IMG_5459_JPG, IMG_5460_JPG               |
|       1324 |                 10 |                   2 |              2 | IMG_5460_JPG                             |
|       1322 |                  9 |                   2 |              2 | IMG_5456_JPG, IMG_5457_JPG               |
|       1325 |                  8 |                   3 |              3 | IMG_5461_JPG                             |
|       1328 |                  5 |                   2 |              2 | IMG_5467_JPG                             |
|       1286 |                  4 |                   4 |              4 | IMG_5332_JPG, IMG_5333_JPG, IMG_5334_JPG |
|       1330 |                  4 |                   2 |              3 | IMG_5503_JPG                             |
|       1333 |                  4 |                   2 |              2 | IMG_5504_JPG                             |
|       1075 |                  3 |                   1 |              1 | IMG_4948                                 |
|       1327 |                  3 |                   1 |              1 | IMG_5467_JPG                             |
|       1336 |                  3 |                   3 |              3 | IMG_5508_JPG                             |
|         82 |                  2 |                   1 |              1 | 000036                                   |
|       1331 |                  2 |                   1 |              1 | IMG_5503_JPG                             |
|       1332 |                  2 |                   1 |              1 | IMG_5504_JPG                             |
|       1285 |                  1 |                   1 |              1 | IMG_5333_JPG                             |
|       1341 |                  1 |                   1 |              1 | photo_2020-10-23_14-02-04                |
|       1725 |                  1 |                   1 |              2 | IMG_5463_JPG                             |

Puntung rokok tersebar di **18 group** (ambang ≥100) / **16 group** (konservatif).

## 4. Simulasi Split Ulang 70:15:15 Berbasis Group

Semua citra dalam satu group masuk split yang sama. Greedy terstratifikasi (grup dengan kelas paling langka ditempatkan dulu), 300 percobaan acak, dipilih galat relatif terkecil terhadap target per kelas.

### a) Group ambang ≥100

|       |   citra |   kertas_tisu |   logam_kaleng_kaca |   plastik |   puntung_rokok |   citra_% |
|:------|--------:|--------------:|--------------------:|----------:|----------------:|----------:|
| train |    2763 |          1519 |                1063 |      4687 |              61 |        70 |
| valid |     593 |           324 |                 227 |      1023 |              14 |        15 |
| test  |     592 |           329 |                 226 |      1040 |              14 |        15 |

### b) Group konservatif (≥25, zona abu-abu ikut digabung)

|       |   citra |   kertas_tisu |   logam_kaleng_kaca |   plastik |   puntung_rokok |   citra_% |
|:------|--------:|--------------:|--------------------:|----------:|----------------:|----------:|
| train |    2725 |          1528 |                1056 |      4683 |              59 |      69   |
| valid |     604 |           318 |                 228 |      1009 |              13 |      15.3 |
| test  |     619 |           326 |                 232 |      1058 |              17 |      15.7 |

### c) Group aman (konservatif + nama sumber sama, kecuali nama numerik TACO) — 1011 group

|       |   citra |   kertas_tisu |   logam_kaleng_kaca |   plastik |   puntung_rokok |   citra_% |
|:------|--------:|--------------:|--------------------:|----------:|----------------:|----------:|
| train |    2713 |          1548 |                1060 |      4766 |              55 |      68.7 |
| valid |     653 |           317 |                 241 |      1017 |              22 |      16.5 |
| test  |     582 |           307 |                 215 |       967 |              12 |      14.7 |

puntung_rokok tersebar di 11 group aman.


**puntung_rokok di test:** 14 instance (≥100), 17 instance (konservatif), 12 instance (aman).

## 5. Kandidat Anotasi Tidak Konsisten

Untuk setiap pasangan adegan sama (inlier ≥ 100), box di citra A dipetakan ke citra B lewat transformasi RANSAC (dan sebaliknya). Box yang tetap ≥70% di dalam frame dicek:
- **konflik kelas**: box terpetakan tumpang tindih (IoU ≥ 0.3) dengan box berkelas lain
- **box hilang**: tidak ada box apa pun di lokasi itu (IoU < 0.1) — objek dianotasi di satu citra tapi tidak di pasangannya

- Pasangan adegan sama yang diperiksa: **3,613**
- Pasangan dengan **konflik kelas**: **866**
- Pasangan dengan **box hilang**: **391**
- Citra unik yang terlibat: **1244**

Per sumber (jumlah pasangan):

| sumber     |   konflik |   hilang |   pasangan_adegan_sama |
|:-----------|----------:|---------:|-----------------------:|
| Burangrang |        57 |       94 |                   2139 |
| TACO       |        33 |       37 |                    193 |
| UAVVaste   |       774 |      256 |                   1274 |
| Unknown    |         2 |        4 |                      7 |

Jenis konflik kelas (A→B):

| konflik                        |   jumlah box |
|:-------------------------------|-------------:|
| kertas_tisu->plastik           |         1158 |
| plastik->kertas_tisu           |         1153 |
| plastik->logam_kaleng_kaca     |          479 |
| logam_kaleng_kaca->plastik     |          467 |
| kertas_tisu->logam_kaleng_kaca |           46 |
| logam_kaleng_kaca->kertas_tisu |           44 |
| plastik->puntung_rokok         |            6 |
| puntung_rokok->plastik         |            6 |
| puntung_rokok->kertas_tisu     |            1 |

Kelas pada box hilang:

| kelas             |   jumlah box |
|:------------------|-------------:|
| plastik           |          730 |
| kertas_tisu       |          255 |
| logam_kaleng_kaca |          111 |
| puntung_rokok     |           10 |

> Catatan: "box hilang" pada frame video berurutan bisa sah (objek bergerak/terhalang); konflik kelas pada lokasi yang sama hampir pasti kesalahan anotasi. Daftar lengkap: `analysis/tables/D0_label_inconsistency.csv`.

## File Output

- `analysis/tables/D0_class_totals.csv`
- `analysis/tables/D0_scene_groups.csv`
- `analysis/tables/D0_scene_edges.csv`
- `analysis/tables/D0_split_simulation.csv`
- `analysis/tables/D0_split_simulation_konservatif.csv`
- `analysis/tables/D0_split_simulation_aman.csv`
- `analysis/tables/D0_split_simulation_assignment.csv`
- `analysis/tables/D0_label_inconsistency.csv`

_Waktu komputasi: 5.1 menit._
---

## Catatan Verifikasi Manual (ditulis tangan, bukan output script)

- **Konflik kelas UAVVaste diverifikasi visual**: pada adegan yang sama (varian rotasi Roboflow), objek kecil
  yang sama diberi kelas berbeda antar salinan (mis. `plastik` di satu salinan, `logam_kaleng_kaca`/`kertas_tisu`
  di salinan lain). Contoh: `train/batch_s02_img_0` (2 salinan), `train/batch_04_img_3180` (2 salinan),
  `batch_04_img_3600` (train vs valid). Indikasi: tiap salinan dianotasi terpisah, bukan diwariskan.
- **Format label**: mayoritas baris adalah poligon segmentasi. Ultralytics mengonversinya otomatis ke bbox,
  jadi training tidak terganggu oleh format ini.
- **Koreksi terhadap `C0_verify.md`**: kesimpulan "185 overlap nama = tabrakan nama, bukan leakage" tidak tepat.
  SSIM gagal pada varian yang dirotasi/di-crop; SIFT+RANSAC menunjukkan sebagian besar overlap adalah adegan sama.
  Tabrakan nama sungguhan hanya terkonfirmasi pada sebagian nama numerik TACO (mis. `000025`, `000026`).
- **Kalibrasi ambang**: 200 pasangan acak berbeda adegan menghasilkan inlier maks 8 (affine penuh), sehingga
  ambang 100 sangat konservatif terhadap false positive. Kelemahan justru di recall: 709 nama sumber non-numerik
  masih terpecah ke >1 group SIFT, karena itu simulasi (c) "group aman" ikut menggabungkan nama sumber yang sama.
- **Sumber data** ditentukan dari pola nama file (heuristik); tagging manual belum pernah diselesaikan.

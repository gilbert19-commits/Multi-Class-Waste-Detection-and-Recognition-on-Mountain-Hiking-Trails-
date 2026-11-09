# Fase 1.4 — Scene group dan test clean (dataset-2kelas)

- Citra: 9554 (semua split), ukuran 514×514 (dicek dari file)
- Kandidat pasangan diuji: **134,973** (sumber kandidat pertama yang menambahkan pasangan: {'nama_sumber_sama': 32644, 'berurutan': 8819, 'resnet50': 56938, 'dhash': 36572})
- Kalibrasi 200 pasangan acak: inlier maks **11**, p95 **5**, median **0**
- Pasangan adegan sama (inlier ≥ 100): **18,491**; zona abu-abu 25–99: **3,446** (tidak dipakai untuk group, sama seperti group_id lama)
- Total group: **2220**; ukuran group: {'2-5': 1288, '1': 546, '6-10': 301, '>10': 85}
- Group yang memuat test DAN train/valid: **216**

## Test clean

| | citra | background | kertas_plastik_puntungRokok | logam_kaleng_kaca | total instance |
|---|---|---|---|---|---|
| test full | 550 | 238 | 1002 | 129 | 1131 |
| test clean | 141 | 60 | 222 | 10 | 232 |
| dibuang | 409 | 178 | 780 | 119 | 899 |

## Sumber data (pola nama file, sama seperti D0)

| sumber     |   train |   valid |   test |   test_clean |
|:-----------|--------:|--------:|-------:|-------------:|
| UAVVaste   |    3549 |     348 |    195 |           20 |
| TACO       |     963 |      84 |     33 |            8 |
| Burangrang |    3855 |     158 |    319 |          112 |
| Unknown    |      42 |       5 |      3 |            1 |

Pola nama Unknown: DJI_#, camera_img_#, photo_#-#-#_#-#-#

_Waktu komputasi: 20.6 menit._
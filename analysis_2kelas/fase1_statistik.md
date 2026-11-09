# Fase 1 — Statistik dataset-2kelas

- data.yaml: `nc: 2`, `names: ['kertas_plastik_puntungRokok', 'logam_kaleng_kaca']` (indeks 0 = kertas_plastik_puntungRokok, 1 = logam_kaleng_kaca)
- Ukuran citra (dibaca dari file): test 514×514: 550, train 514×514: 8409, valid 514×514: 595

## Citra per split

| split   |   citra |   background |   file_campuran_poligon_box |   file_hanya_poligon |   baris_box |   baris_poligon |
|:--------|--------:|-------------:|----------------------------:|---------------------:|------------:|----------------:|
| train   |    8409 |         2401 |                          54 |                 4964 |        2747 |           18054 |
| valid   |     595 |           98 |                           3 |                  469 |          77 |            1946 |
| test    |     550 |          238 |                           2 |                  242 |         210 |             921 |
| total   |    9554 |         2737 |                          59 |                 5675 |        3034 |           20921 |

## Instance per kelas per split

| split   |   kertas_plastik_puntungRokok |   logam_kaleng_kaca |   total |
|:--------|------------------------------:|--------------------:|--------:|
| train   |                         17614 |                3187 |   20801 |
| valid   |                          1741 |                 282 |    2023 |
| test    |                          1002 |                 129 |    1131 |
| total   |                         20357 |                3598 |   23955 |

## Ukuran objek (piksel citra, bbox dari poligon bila poligon)

Bin luas COCO: kecil < 32² px, sedang 32²–96², besar > 96².

| split   |      objek |   % kecil (<32²) |   % lebar&tinggi <32px |   median luas (px²) |
|:--------|-----------:|-----------------:|-----------------------:|--------------------:|
| train   | 20801.0000 |          76.8905 |                63.7950 |            326.5577 |
| valid   |  2023.0000 |          84.9728 |                72.8127 |            237.5508 |
| test    |  1131.0000 |          80.1945 |                68.7003 |            275.0977 |
| total   | 23955.0000 |          77.7291 |                64.7881 |            314.8652 |

|                                          |   kecil (<32²) |   sedang (32²–96²) |   besar (>96²) |      total |   % kecil (<32²) |   % lebar&tinggi <32px |
|:-----------------------------------------|---------------:|-------------------:|---------------:|-----------:|-----------------:|-----------------------:|
| ('train', 'kertas_plastik_puntungRokok') |     13620.0000 |          2955.0000 |      1039.0000 | 17614.0000 |          77.3249 |                66.0952 |
| ('train', 'logam_kaleng_kaca')           |      2374.0000 |           500.0000 |       313.0000 |  3187.0000 |          74.4901 |                51.0825 |
| ('valid', 'kertas_plastik_puntungRokok') |      1491.0000 |           174.0000 |        76.0000 |  1741.0000 |          85.6404 |                74.9569 |
| ('valid', 'logam_kaleng_kaca')           |       228.0000 |            36.0000 |        18.0000 |   282.0000 |          80.8511 |                59.5745 |
| ('test', 'kertas_plastik_puntungRokok')  |       802.0000 |           180.0000 |        20.0000 |  1002.0000 |          80.0399 |                69.9601 |
| ('test', 'logam_kaleng_kaca')            |       105.0000 |            18.0000 |         6.0000 |   129.0000 |          81.3953 |                58.9147 |

## Masalah label

Jumlah: 0

# Analisis Error: YOLOv12 vs YOLOv8

Prediksi pada imgsz=640, NMS IoU=0.6. Confusion matrix dan TP/FP/FN memakai conf ≥ 0.25 dan IoU > 0.5 (pencocokan seperti Ultralytics ConfusionMatrix). AP50 memakai conf ≥ 0.001.

- Test full: 548 citra (citra berlabel campuran dibuang, sama seperti Ultralytics).
- Test clean: 192 citra. puntung_rokok tidak punya instance di test clean.
- FN = objek yang tidak terdeteksi (terlewat) + objek yang terdeteksi dengan kelas salah.
- FP = deteksi pada latar + deteksi dengan kelas salah.

## 1. Head-to-head YOLOv12s aug vs YOLOv8s aug

Δ = YOLOv12 − YOLOv8. Untuk FN dan FP, Δ negatif berarti YOLOv12 lebih baik.

### Test full (548 citra, bootstrap 1000× berpasangan pada level citra)

| Kelas | GT | FN v12 | FN v8 | ΔFN | ΔFN % | 95% CI ΔFN | FP v12 | FP v8 | ΔFP | 95% CI ΔFP | Kesimpulan FN | Kesimpulan FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kertas_tisu | 233 | 146 | 155 | -9 | -5.8% | [-25, +5] | 145 | 160 | -15 | [-44, +11] | setara | setara |
| logam_kaleng_kaca | 129 | 65 | 63 | +2 | +3.2% | [-6, +11] | 57 | 79 | -22 | [-41, -2] | setara | YOLOv12 lebih baik |
| plastik | 763 | 268 | 276 | -8 | -2.9% | [-34, +15] | 370 | 349 | +21 | [-21, +64] | setara | setara |
| puntung_rokok | 2 | 1 | 0 | +1 | – | [+0, +3] | 10 | 9 | +1 | [-7, +8] | setara | setara |
| TOTAL | 1127 | 480 | 494 | -14 | -2.8% | [-43, +13] | 582 | 597 | -15 | [-72, +41] | setara | setara |

mAP50 (4 kelas): YOLOv12s aug 0.523, YOLOv8s aug 0.554, Δ -0.030, 95% CI [-0.239, +0.176].

Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas):

| kelas | FN_salah_kelas YOLOv12 | FN_salah_kelas YOLOv8 | FN_terlewat YOLOv12 | FN_terlewat YOLOv8 | FP_latar YOLOv12 | FP_latar YOLOv8 | FP_salah_kelas YOLOv12 | FP_salah_kelas YOLOv8 |
|---|---|---|---|---|---|---|---|---|
| kertas_tisu | 94 | 98 | 52 | 57 | 84 | 84 | 61 | 76 |
| logam_kaleng_kaca | 43 | 41 | 22 | 22 | 31 | 42 | 26 | 37 |
| plastik | 86 | 112 | 182 | 164 | 236 | 212 | 134 | 137 |
| puntung_rokok | 0 | 0 | 1 | 0 | 8 | 8 | 2 | 1 |

### Test clean (192 citra, bootstrap 1000× berpasangan pada level citra)

| Kelas | GT | FN v12 | FN v8 | ΔFN | ΔFN % | 95% CI ΔFN | FP v12 | FP v8 | ΔFP | 95% CI ΔFP | Kesimpulan FN | Kesimpulan FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kertas_tisu | 75 | 51 | 53 | -2 | -3.8% | [-10, +5] | 35 | 33 | +2 | [-13, +16] | setara | setara |
| logam_kaleng_kaca | 34 | 23 | 18 | +5 | +27.8% | [-1, +11] | 15 | 25 | -10 | [-20, +0] | setara | setara |
| plastik | 241 | 98 | 101 | -3 | -3.0% | [-18, +11] | 137 | 117 | +20 | [-5, +46] | setara | setara |
| puntung_rokok | 0 | – | – | – | – | – | 7 | 6 | 1 | – | tidak ada GT | – |
| TOTAL | 350 | 172 | 172 | +0 | +0.0% | [-19, +17] | 194 | 181 | +13 | [-22, +47] | setara | setara |

mAP50 (3 kelas): YOLOv12s aug 0.469, YOLOv8s aug 0.480, Δ -0.011, 95% CI [-0.068, +0.045].

Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas):

| kelas | FN_salah_kelas YOLOv12 | FN_salah_kelas YOLOv8 | FN_terlewat YOLOv12 | FN_terlewat YOLOv8 | FP_latar YOLOv12 | FP_latar YOLOv8 | FP_salah_kelas YOLOv12 | FP_salah_kelas YOLOv8 |
|---|---|---|---|---|---|---|---|---|
| kertas_tisu | 29 | 28 | 22 | 25 | 20 | 19 | 15 | 14 |
| logam_kaleng_kaca | 19 | 16 | 4 | 2 | 9 | 15 | 6 | 10 |
| plastik | 21 | 24 | 77 | 77 | 91 | 73 | 46 | 44 |

Kesimpulan per kelas ditentukan oleh CI 95%: kalau CI memuat 0, selisihnya masih dalam rentang noise dan dicatat **setara**.

## 2. Pasangan kelas yang paling sering tertukar

Jumlah objek dengan kelas sebenarnya A yang diprediksi sebagai B (conf ≥ 0.25, IoU > 0.5). Kolom 'semua model' = jumlah dari semua model yang dievaluasi.

### Test full

| Pasangan (A ↔ B) | Semua model | A→B | B→A | YOLOv12s aug | YOLOv8s aug |
|---|---|---|---|---|---|
| kertas_tisu ↔ plastik | 2036 | 1226 | 810 | 152 | 172 |
| logam_kaleng_kaca ↔ plastik | 858 | 520 | 338 | 67 | 76 |
| kertas_tisu ↔ logam_kaleng_kaca | 56 | 11 | 45 | 2 | 2 |
| plastik ↔ puntung_rokok | 13 | 6 | 7 | 1 | 1 |
| kertas_tisu ↔ puntung_rokok | 2 | 2 | 0 | 1 | 0 |

### Test clean

| Pasangan (A ↔ B) | Semua model | A→B | B→A | YOLOv12s aug | YOLOv8s aug |
|---|---|---|---|---|---|
| kertas_tisu ↔ plastik | 530 | 352 | 178 | 42 | 42 |
| logam_kaleng_kaca ↔ plastik | 295 | 211 | 84 | 24 | 26 |
| kertas_tisu ↔ logam_kaleng_kaca | 17 | 1 | 16 | 1 | 0 |
| plastik ↔ puntung_rokok | 2 | 2 | 0 | 1 | 0 |
| kertas_tisu ↔ puntung_rokok | 1 | 1 | 0 | 1 | 0 |

## 3. Stratifikasi pada test clean

Recall pada conf ≥ 0.25 (objek dihitung benar bila terdeteksi dengan kelas benar). mAP50 = rata-rata AP50 kelas yang punya objek di kelompok tsb. Untuk strata ukuran, deteksi yang cocok dengan objek di luar strata diabaikan, dan deteksi yang tidak cocok dihitung FP hanya bila ukurannya sendiri masuk strata (konvensi COCO area range).

### Ukuran objek (luas bbox pada 640×640)

| Kelompok | Citra | Objek | Recall v12 | Recall v8 | mAP50 v12 | mAP50 v8 |
|---|---|---|---|---|---|---|
| small (<32²) | 76 | 235 | 0.489 | 0.481 | 0.502 | 0.488 |
| medium | 61 | 96 | 0.583 | 0.562 | 0.506 | 0.507 |
| large (>96²) | 15 | 19 | 0.368 | 0.579 | 0.396 | 0.587 |

### Kepadatan objek per citra

| Kelompok | Citra | Objek | Recall v12 | Recall v8 | mAP50 v12 | mAP50 v8 |
|---|---|---|---|---|---|---|
| 1-2 | 77 | 102 | 0.647 | 0.667 | 0.470 | 0.559 |
| 3-5 | 36 | 126 | 0.540 | 0.524 | 0.537 | 0.559 |
| >5 | 10 | 122 | 0.361 | 0.361 | 0.355 | 0.352 |

Kelompok dengan sedikit objek (lihat kolom Objek) punya ketidakpastian besar; selisih kecil di sana jangan dibaca sebagai perbedaan nyata.

## 4. Cek label campuran (poligon + box dalam satu file)

Aturan Ultralytics: kalau satu file label berisi baris poligon (>6 kolom) dan baris box (5 kolom), seluruh citra dibuang dari training/evaluasi, beserta semua anotasinya.

| dataset | split | file_label | file_campuran | objek_terbuang |
|---|---|---|---|---|
| TA-1 | test | 550 | 2 | 4 |
| TA-1 | train | 2803 | 18 | 71 |
| TA-1 | valid | 595 | 3 | 16 |
| augmented | test | 550 | 2 | 4 |
| augmented | train | 8409 | 54 | 212 |
| augmented | valid | 595 | 3 | 16 |

File bermasalah:

| dataset | split | file | n_box | n_seg | classes |
|---|---|---|---|---|---|
| TA-1 | train | 000015_jpg.rf.cf37811655b805d45b7cf01877d4b28d.txt | 1 | 1 | 2 |
| TA-1 | train | 000021_jpg.rf.83a96f88b62c567161dcf280199b56cf.txt | 1 | 2 | 1,2 |
| TA-1 | train | 000021_jpg.rf.a56d28cf8135996baad7c0f9dd2b234f.txt | 1 | 2 | 1,2 |
| TA-1 | train | 000022_jpg.rf.62deed49b84b8a6d65e115f535bb2f6c.txt | 1 | 1 | 1,2 |
| TA-1 | train | 000022_jpg.rf.a1c1a73d12001707acf7b358160882e3.txt | 1 | 1 | 1,2 |
| TA-1 | train | 000072_JPG.rf.0187256c39ccd6107cf2ef6d04c9cea4.txt | 1 | 1 | 0,2 |
| TA-1 | train | 000072_JPG.rf.8b5c74a01a3c5afb8766181f3adf4122.txt | 1 | 1 | 0,2 |
| TA-1 | train | 000072_JPG.rf.d5eb7ec53ce5cc1d04eb35d6792b98cc.txt | 1 | 1 | 0,2 |
| TA-1 | train | 000093_jpg.rf.789e3e8544680ecac4fe5ad4d01464df.txt | 1 | 11 | 0,1,2 |
| TA-1 | train | 000096_jpg.rf.7f2bcbaa130fd3f2b294f306f61fd5ce.txt | 1 | 3 | 0,2 |
| TA-1 | train | batch_s02_img_104_jpg.rf.eef89ba052823cd602589490a9d63c55.txt | 1 | 16 | 0,1,2 |
| TA-1 | train | batch_s02_img_114_jpg.rf.f17eceb57fc33acb6fded612b9714967.txt | 3 | 1 | 0,1,2 |
| TA-1 | train | IMG_4875_JPG.rf.6a9444d7d9ea8dded568abcf2803d204.txt | 1 | 1 | 0 |
| TA-1 | train | IMG_4875_JPG.rf.6caa76c84dabc6e99fe5e5f3d5cedfff.txt | 1 | 1 | 0 |
| TA-1 | train | IMG_4922_JPG.rf.b57ea4bbf00cdbd59d428b495e127886.txt | 1 | 1 | 2 |
| TA-1 | train | IMG_4948_JPG.rf.bfaaba45616ebc5b3420046b892b23b4.txt | 1 | 5 | 1,2 |
| TA-1 | train | IMG_4994_JPG.rf.81ec5e7a0f470e79e6ac5dd95261c121.txt | 1 | 1 | 2 |
| TA-1 | train | IMG_4994_JPG.rf.cc5d4af37ced7e5667f8c4afe4610798.txt | 1 | 1 | 1,2 |
| TA-1 | valid | 000022_jpg.rf.67eea131affcbaa0ba96c5b2947859c4.txt | 1 | 1 | 1,2 |
| TA-1 | valid | 000093_jpg.rf.effe2fcd9f5be0238eaf8a144b3ec907.txt | 1 | 10 | 0,1,2 |
| TA-1 | valid | 000096_jpg.rf.a88cded28c7e3ebe6732fbaa59cfdbf0.txt | 1 | 2 | 2 |
| TA-1 | test | 000015_jpg.rf.a6dd372912ce2f5a7950ed34a1a6a2d0.txt | 1 | 1 | 2 |
| TA-1 | test | 000077_jpg.rf.59307395b28a31c9a0bb1fa813c3db1d.txt | 1 | 1 | 2 |
| augmented | train | 000015_jpg.rf.4132f539c8a574cad6748b559dcfd0ff.txt | 1 | 1 | 2 |
| augmented | train | 000015_jpg.rf.b2e66e2b8e087fc4a72781c6ac0ff6c2.txt | 1 | 1 | 2 |
| augmented | train | 000015_jpg.rf.b839aee71c1be6ba61f45822b66d978d.txt | 1 | 1 | 2 |
| augmented | train | 000021_jpg.rf.048e2cdb77e5803050e8579ba5e6066e.txt | 1 | 2 | 1,2 |
| augmented | train | 000021_jpg.rf.15bf5acb73ca83806dea0ca6444e9944.txt | 1 | 2 | 1,2 |
| augmented | train | 000021_jpg.rf.5a0c79ad5f67752e41270fb189256e6c.txt | 1 | 2 | 1,2 |
| augmented | train | 000021_jpg.rf.5e35c7cb3d5fd200085f75ff1403f86c.txt | 1 | 2 | 1,2 |
| augmented | train | 000021_jpg.rf.74b16f0393fc005e573c93bfc8655a34.txt | 1 | 2 | 1,2 |
| augmented | train | 000021_jpg.rf.b07b0c8cd91faba0774816c8d12cfe6e.txt | 1 | 2 | 1,2 |
| augmented | train | 000022_jpg.rf.51a787aab11b97eb7b6b74fe4ea52483.txt | 1 | 1 | 1,2 |
| augmented | train | 000022_jpg.rf.6eb6c32bd2e07537eea2282f5f68a722.txt | 1 | 1 | 1,2 |
| augmented | train | 000022_jpg.rf.c03175ac3b3fc484186f242c181b0887.txt | 1 | 1 | 1,2 |
| augmented | train | 000022_jpg.rf.ca25f38a27969deca72650ae39c3e988.txt | 1 | 1 | 1,2 |
| augmented | train | 000022_jpg.rf.d20d0e99ee62b72c88e14fd6a39b7052.txt | 1 | 1 | 1,2 |
| augmented | train | 000022_jpg.rf.f984cdf71573a14cc888dd21bc994c75.txt | 1 | 1 | 1,2 |
| augmented | train | 000072_JPG.rf.61110f2e6acc5e6da67ed0c58e717e16.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.8608abb2ecdcb052c7ee560a751096b8.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.88468fd0d8e24e29b0ec448aeade8d59.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.a8eb2bd686cf535151e36e0472d3f1d1.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.acd3f869f8545a398df23f1a86272c04.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.c3409818a710ca920950b7c82cebe0d6.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.cd4f3242284eec2e71274382d8b2b134.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.d15222c2216d20e701b3489cbe3b6404.txt | 1 | 1 | 0,2 |
| augmented | train | 000072_JPG.rf.f80e1cabb528e50e5091f78203809edd.txt | 1 | 1 | 0,2 |
| augmented | train | 000093_jpg.rf.410ccd19c3a958cbda7293478227dabe.txt | 1 | 11 | 0,1,2 |
| augmented | train | 000093_jpg.rf.ab8b59b3e198f19c23231c43e4608fac.txt | 1 | 11 | 0,1,2 |
| augmented | train | 000093_jpg.rf.bb8dbe10c896835ec0926fd04cff676b.txt | 1 | 11 | 0,1,2 |
| augmented | train | 000096_jpg.rf.aee1d079d7884f04cf6a3dfc345e19cb.txt | 1 | 3 | 0,2 |
| augmented | train | 000096_jpg.rf.b9b3bd4c1d4db0c5de18a41390ecc76d.txt | 1 | 3 | 0,2 |
| augmented | train | 000096_jpg.rf.ba40da7407e4b1a4cc9608d2071656e4.txt | 1 | 3 | 0,2 |
| augmented | train | batch_s02_img_104_jpg.rf.0dde5fab73fcdb4dce4e5e8a801d4cff.txt | 1 | 16 | 0,1,2 |
| augmented | train | batch_s02_img_104_jpg.rf.b69f07b42d62d457e25cdfa5c65205d6.txt | 1 | 15 | 0,1,2 |
| augmented | train | batch_s02_img_104_jpg.rf.e722a8cd546aaa5e8d5977831c309146.txt | 1 | 16 | 0,1,2 |
| augmented | train | batch_s02_img_114_jpg.rf.77131df681aaae7cf0a2fbb708de7ffa.txt | 3 | 1 | 0,1,2 |
| augmented | train | batch_s02_img_114_jpg.rf.a38e628430239d39ebef9950e39e03b1.txt | 3 | 1 | 0,1,2 |
| augmented | train | batch_s02_img_114_jpg.rf.ab5b54cba9b00b66ee3ee6196329a0df.txt | 3 | 1 | 0,1,2 |
| augmented | train | IMG_4875_JPG.rf.08d7bff05116a8ae4ce4628827a9fe2d.txt | 1 | 1 | 0 |
| augmented | train | IMG_4875_JPG.rf.58a024789d5a8820cfad0703dfcd3555.txt | 1 | 1 | 0 |
| augmented | train | IMG_4875_JPG.rf.59eaba05cb5fd6a171ac5de773e9b57e.txt | 1 | 1 | 0 |
| augmented | train | IMG_4875_JPG.rf.88e8b36cb0a1b79ea90bb8856ed9c0ec.txt | 1 | 1 | 0 |
| augmented | train | IMG_4875_JPG.rf.dab801ead69f9d6baee873c5d3a8a1c4.txt | 1 | 1 | 0 |
| augmented | train | IMG_4875_JPG.rf.e9c33010cc5c2b81d1e09a9b0ad65ee4.txt | 1 | 1 | 0 |
| augmented | train | IMG_4922_JPG.rf.516ad755e5caa3ea203b6bccd3b58084.txt | 1 | 1 | 2 |
| augmented | train | IMG_4922_JPG.rf.7d1af92e516326affc5f6ef2bb5265b1.txt | 1 | 1 | 2 |
| augmented | train | IMG_4922_JPG.rf.ed1d41db22c3a8a8d885082de02183cc.txt | 1 | 1 | 2 |
| augmented | train | IMG_4948_JPG.rf.4b6c72cdefcfb0c0f684de68b4d0735c.txt | 1 | 5 | 1,2 |
| augmented | train | IMG_4948_JPG.rf.63ef8a01931b89b053480d1b25acc26e.txt | 1 | 5 | 1,2 |
| augmented | train | IMG_4948_JPG.rf.9de8edf9a7c9f75f5fe934341dcafc08.txt | 1 | 5 | 1,2 |
| augmented | train | IMG_4994_JPG.rf.0a6da35746800ed7e56543ef9b401375.txt | 1 | 1 | 2 |
| augmented | train | IMG_4994_JPG.rf.91baa0aa4b1fb469d7f189b75224ab23.txt | 1 | 1 | 1,2 |
| augmented | train | IMG_4994_JPG.rf.b48b59b368a68e162ec6cb1d7fd34176.txt | 1 | 1 | 2 |
| augmented | train | IMG_4994_JPG.rf.c0fe4323df8719af66da9ff3687027b8.txt | 1 | 1 | 2 |
| augmented | train | IMG_4994_JPG.rf.e6f78400bf39493c50693892a222f688.txt | 1 | 1 | 1,2 |
| augmented | train | IMG_4994_JPG.rf.ebefcd0ba78235d1e6c02b5e11affc7e.txt | 1 | 1 | 1,2 |
| augmented | valid | 000022_jpg.rf.f0d199eb6ee32859a33ae2f6b11392b6.txt | 1 | 1 | 1,2 |
| augmented | valid | 000093_jpg.rf.64b77eafec99ec3b81c3e1145aab569d.txt | 1 | 10 | 0,1,2 |
| augmented | valid | 000096_jpg.rf.1e2e20a60d284d68e93019f2e6b216a6.txt | 1 | 2 | 2 |
| augmented | test | 000015_jpg.rf.3ed90ca25fe0213cf3c6f49507a5d758.txt | 1 | 1 | 2 |
| augmented | test | 000077_jpg.rf.f1130789a9060377206138758ad5e55c.txt | 1 | 1 | 2 |

Ada 72 file di split train (TA-1, augmented). Citra-citra ini **tidak pernah dipakai saat training**, beserta seluruh anotasinya.

## 5. Kesimpulan

Pada test clean, YOLOv12s aug menghasilkan 172 FN dan YOLOv8s aug 172 FN (Δ +0, 95% CI [-19, +17]). CI memuat 0, sehingga perbedaan jumlah FN masih dalam rentang noise dan hipotesis bahwa YOLOv12 menghasilkan FN lebih sedikit **tidak terdukung** oleh data ini. Tidak ada kelas dengan selisih FN yang bermakna. Secara arah saja (tanpa signifikansi), YOLOv12 punya FN lebih sedikit pada kertas_tisu, plastik dan YOLOv8 punya FN lebih sedikit pada logam_kaleng_kaca. Kesimpulan ini terbatas pada pasangan varian s augmented, satu seed training, dan test clean berukuran 192 citra tanpa instance puntung_rokok.

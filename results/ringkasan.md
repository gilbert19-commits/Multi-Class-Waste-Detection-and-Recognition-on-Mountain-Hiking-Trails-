# Ringkasan Evaluasi Test Set

## Ukuran test set

- Full: 550 citra, **548 dievaluasi**.
- Clean: **192 citra** (550 − 358 citra di scene group lintas split), 192 dievaluasi. Sesuai target 192.

Citra yang ditolak Ultralytics di test full (tidak ikut dihitung, untuk semua model sama):

- `000015_jpg.rf.a6dd372912ce2f5a7950ed34a1a6a2d0.jpg`: labels mix segment and detection rows
- `000077_jpg.rf.59307395b28a31c9a0bb1fa813c3db1d.jpg`: labels mix segment and detection rows

## Catatan: jumlah kelas berbeda

Di test clean, kelas puntung_rokok tidak punya instance, sehingga mAP clean dihitung dari 3 kelas, sedangkan mAP full dari 4 kelas. Kedua angka itu **tidak setara** dan tidak boleh dibandingkan langsung. Untuk perbandingan full vs clean dipakai mAP full versi 3 kelas (tanpa puntung_rokok).

## Varian YOLOv12 terbaik (mAP@50-95, test clean)

**YOLOv12s aug**: mAP@50-95 clean 0.251 (3 kelas), mAP@50 clean 0.465. Di test full: mAP@50-95 0.279 (4 kelas) / 0.305 (3 kelas).

Peringkat YOLOv12 di test clean:

| Peringkat | Model | mAP@50-95 clean | mAP@50 clean |
|---|---|---|---|
| 1 | YOLOv12s aug | 0.251 | 0.465 |
| 2 | YOLOv12n aug | 0.238 | 0.439 |
| 3 | YOLOv12m aug | 0.225 | 0.427 |
| 4 | YOLOv12l aug | 0.209 | 0.409 |
| 5 | YOLOv12x aug | 0.206 | 0.394 |
| 6 | YOLOv12s | 0.205 | 0.398 |
| 7 | YOLOv12l | 0.194 | 0.370 |
| 8 | YOLOv12n | 0.192 | 0.387 |
| 9 | YOLOv12x | 0.186 | 0.342 |
| 10 | YOLOv12m | 0.180 | 0.351 |

## YOLOv12s aug vs YOLOv8s aug (YOLOv8 terbaik), per kelas

**Test full**

| Kelas | Instances | mAP@50-95 YOLOv12s aug | mAP@50-95 YOLOv8s aug | Δ | mAP@50 YOLOv12s aug | mAP@50 YOLOv8s aug | Δ |
|---|---|---|---|---|---|---|---|
| kertas_tisu | 233 | 0.232 | 0.212 | +0.020 | 0.400 | 0.360 | +0.040 |
| logam_kaleng_kaca | 129 | 0.336 | 0.345 | -0.009 | 0.540 | 0.530 | +0.011 |
| plastik | 763 | 0.346 | 0.346 | -0.000 | 0.634 | 0.637 | -0.003 |
| puntung_rokok | 2 | 0.203 | 0.298 | -0.095 | 0.506 | 0.663 | -0.157 |

**Test clean**

| Kelas | Instances | mAP@50-95 YOLOv12s aug | mAP@50-95 YOLOv8s aug | Δ | mAP@50 YOLOv12s aug | mAP@50 YOLOv8s aug | Δ |
|---|---|---|---|---|---|---|---|
| kertas_tisu | 75 | 0.186 | 0.196 | -0.010 | 0.363 | 0.361 | +0.002 |
| logam_kaleng_kaca | 34 | 0.298 | 0.275 | +0.022 | 0.476 | 0.492 | -0.016 |
| plastik | 241 | 0.270 | 0.282 | -0.012 | 0.555 | 0.559 | -0.004 |
| puntung_rokok | 0 | – | – | – | – | – | – |

## Dampak overlap antar split (clean − full)

Rata-rata dari 13 model, pembanding full 3 kelas (setara):

- Δ mAP@50: **-0.058** (min -0.093, maks -0.024)
- Δ mAP@50-95: **-0.053** (min -0.083, maks -0.015)

Sebagai referensi saja (tidak setara, 3 vs 4 kelas): Δ mAP@50-95 rata-rata -0.073.

Nilai negatif berarti model lebih buruk pada citra yang adegannya tidak pernah terlihat saat training. Besarnya selisih itu adalah ukuran seberapa angka test full terdongkrak oleh overlap.

Catatan: subset clean juga berbeda komposisi sumbernya dari full, jadi Δ mencampur efek leakage dan efek perbedaan tingkat kesulitan citra.

## Run yang dilewati

- v8n_test_3ep: smoke test (3 epoch)

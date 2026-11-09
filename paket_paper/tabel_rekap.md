# Rekap Evaluasi Test Set

Pengaturan identik untuk semua model: imgsz=640, conf=0.001, iou=0.6, batch=16, device=0, seed=0, deterministic=True.

- **Full**: TA-1/test, 548 citra dievaluasi, 4 kelas (citra yang labelnya ditolak Ultralytics tidak dihitung, lihat ringkasan.md).
- **Clean**: 192 citra, tanpa scene group yang juga ada di train/val (D0). puntung_rokok tidak punya instance, jadi metrik clean = rata-rata 3 kelas.
- **Δ** = clean − full (3 kelas). Pembanding full memakai 3 kelas yang sama supaya setara.

## Akurasi

| Model | Varian | Augmentasi | P full | R full | mAP@50 full (4 kls) | mAP@50-95 full (4 kls) | mAP@50 full (3 kls) | mAP@50-95 full (3 kls) | P clean | R clean | mAP@50 clean (3 kls) | mAP@50-95 clean (3 kls) | Δ mAP@50 | Δ mAP@50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv12 | n | tidak | 0.361 | 0.408 | 0.346 | 0.197 | 0.452 | 0.256 | 0.415 | 0.523 | 0.387 | 0.192 | -0.065 | -0.064 |
| YOLOv12 | s | tidak | 0.579 | 0.554 | 0.586 | 0.344 | 0.450 | 0.251 | 0.432 | 0.414 | 0.398 | 0.205 | -0.052 | -0.046 |
| YOLOv12 | m | tidak | 0.623 | 0.528 | 0.546 | 0.312 | 0.396 | 0.217 | 0.345 | 0.412 | 0.351 | 0.180 | -0.045 | -0.037 |
| YOLOv12 | l | tidak | 0.537 | 0.543 | 0.555 | 0.245 | 0.409 | 0.225 | 0.396 | 0.388 | 0.370 | 0.194 | -0.038 | -0.032 |
| YOLOv12 | x | tidak | 0.426 | 0.450 | 0.408 | 0.211 | 0.365 | 0.201 | 0.403 | 0.371 | 0.342 | 0.186 | -0.024 | -0.015 |
| YOLOv12 | n | ya | 0.654 | 0.549 | 0.619 | 0.265 | 0.493 | 0.287 | 0.519 | 0.481 | 0.439 | 0.238 | -0.054 | -0.049 |
| YOLOv12 | s | ya | 0.483 | 0.518 | 0.520 | 0.279 | 0.525 | 0.305 | 0.514 | 0.489 | 0.465 | 0.251 | -0.060 | -0.053 |
| YOLOv12 | m | ya | 0.650 | 0.576 | 0.633 | 0.401 | 0.512 | 0.302 | 0.490 | 0.474 | 0.427 | 0.225 | -0.085 | -0.077 |
| YOLOv12 | l | ya | 0.614 | 0.587 | 0.625 | 0.269 | 0.502 | 0.292 | 0.519 | 0.435 | 0.409 | 0.209 | -0.093 | -0.083 |
| YOLOv12 | x | ya | 0.599 | 0.604 | 0.608 | 0.328 | 0.479 | 0.271 | 0.407 | 0.504 | 0.394 | 0.206 | -0.085 | -0.065 |
| YOLOv8 | n | ya | 0.620 | 0.602 | 0.618 | 0.310 | 0.493 | 0.281 | 0.431 | 0.502 | 0.427 | 0.228 | -0.066 | -0.053 |
| YOLOv8 | s | ya | 0.556 | 0.581 | 0.548 | 0.300 | 0.509 | 0.301 | 0.526 | 0.509 | 0.471 | 0.251 | -0.038 | -0.050 |
| YOLOv8 | m | ya | 0.533 | 0.609 | 0.507 | 0.296 | 0.511 | 0.304 | 0.534 | 0.500 | 0.458 | 0.243 | -0.052 | -0.061 |

## Efisiensi

ms/img = waktu inference per citra (batch 16, GPU), diukur pada test full.

| Model | Varian | Augmentasi | Params | GFLOPs | ms/img |
|---|---|---|---|---|---|
| YOLOv12 | n | tidak | 2.57 M | 7.5 | 1.82 |
| YOLOv12 | s | tidak | 9.25 M | 23.5 | 3.86 |
| YOLOv12 | m | tidak | 20.14 M | 71.0 | 8.88 |
| YOLOv12 | l | tidak | 26.39 M | 96.0 | 12.68 |
| YOLOv12 | x | tidak | 59.12 M | 209.7 | 22.00 |
| YOLOv12 | n | ya | 2.57 M | 7.5 | 2.20 |
| YOLOv12 | s | ya | 9.25 M | 23.5 | 3.84 |
| YOLOv12 | m | ya | 20.14 M | 71.0 | 8.68 |
| YOLOv12 | l | ya | 26.39 M | 96.0 | 12.66 |
| YOLOv12 | x | ya | 59.12 M | 209.7 | 21.95 |
| YOLOv8 | n | ya | 3.01 M | 8.2 | 1.07 |
| YOLOv8 | s | ya | 11.14 M | 28.6 | 2.78 |
| YOLOv8 | m | ya | 25.86 M | 79.1 | 6.19 |

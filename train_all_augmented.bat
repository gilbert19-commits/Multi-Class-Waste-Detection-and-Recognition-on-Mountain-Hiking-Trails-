@echo off
REM =====================================================================
REM Training Otomatis Semua Varian YOLOv12 pada Dataset Augmented
REM Konfigurasi: Batch Size = -1 (AutoBatch), Epochs = 100, Device = 0
REM =====================================================================

call conda activate ultralytics_env

echo =====================================================================
echo       MEMULAI TRAINING SEMUA VARIAN YOLOV12 (AUGMENTED DATASET)
echo =====================================================================

python augmented\train_all_augmented.py

echo.
echo =====================================================================
echo        SEMUA VARIAN AUGMENTASI SELESAI DILATIH DENGAN SUKSES!
echo        Hasil tersimpan di folder: runs\augmented\
echo =====================================================================
pause

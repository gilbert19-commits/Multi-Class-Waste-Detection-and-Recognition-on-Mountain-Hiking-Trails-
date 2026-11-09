@echo off
REM =====================================================================
REM Training Otomatis Semua Varian YOLOv12 di Anaconda Prompt
REM Konfigurasi: Batch Size = -1 (AutoBatch), Epochs = 100, Device = 0
REM =====================================================================

call conda activate ultralytics_env

echo =====================================================================
echo                MEMULAI TRAINING SEMUA VARIAN YOLOV12
echo =====================================================================

echo [1/5] Training YOLOv12n (Nano)...
python train.py --model yolov12n.pt --epochs 100 --batch -1 --device 0 --imgsz 640 --name waste_yolov12n

echo.
echo [2/5] Training YOLOv12s (Small)...
python train.py --model yolov12s.pt --epochs 100 --batch -1 --device 0 --imgsz 640 --name waste_yolov12s

echo.
echo [3/5] Training YOLOv12m (Medium)...
python train.py --model yolov12m.pt --epochs 100 --batch -1 --device 0 --imgsz 640 --name waste_yolov12m

echo.
echo [4/5] Training YOLOv12l (Large)...
python train.py --model yolov12l.pt --epochs 100 --batch -1 --device 0 --imgsz 640 --name waste_yolov12l

echo.
echo [5/5] Training YOLOv12x (XLarge)...
python train.py --model yolov12x.pt --epochs 100 --batch -1 --device 0 --imgsz 640 --name waste_yolov12x

echo.
echo =====================================================================
echo             SEMUA VARIAN SELESAI DILATIH DENGAN SUKSES!
echo      Hasil tersimpan di folder: runs\detect\
echo =====================================================================
pause

@echo off
REM =====================================================================
REM Resume dan Selesaikan Training YOLOv12 Augmented (YOLOv12l dan YOLOv12x)
REM Dilanjutkan Evaluasi Otomatis Test Split untuk Semua Varian (n,s,m,l,x)
REM =====================================================================

REM 1. Aktifkan Conda Environment ultralytics_env
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat" ultralytics_env
) else (
    call conda activate ultralytics_env 2>nul
)

REM 2. Tentukan Path Python ultralytics_env secara presisi
set "PYTHON_CMD=python"
if exist "C:\Users\User\.conda\envs\ultralytics_env\python.exe" (
    set "PYTHON_CMD=C:\Users\User\.conda\envs\ultralytics_env\python.exe"
)

echo =====================================================================
echo       MELANJUTKAN TRAINING YOLOV12L DAN YOLOV12X (AUGMENTED DATASET)
echo =====================================================================

"%PYTHON_CMD%" resume_training_augmented.py

echo.
echo =====================================================================
echo    SEMUA MODEL TELAH SELESAI DILATIH DAN DIEVALUASI DENGAN SUKSES!
echo    Hasil tersimpan di folder: runs\detect\runs\augmented\
echo =====================================================================
pause

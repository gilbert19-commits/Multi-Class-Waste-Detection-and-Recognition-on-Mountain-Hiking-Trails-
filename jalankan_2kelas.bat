@echo off
REM =====================================================================
REM Training dataset 2 kelas: 10 model berurutan (lihat train_2kelas.py)
REM Output konsol disimpan di train_2kelas.log
REM Hentikan antrian setelah model berjalan selesai: buat file STOP_ANTRIAN_2kelas
REM =====================================================================

cd /d "%~dp0"

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

"%PYTHON_CMD%" train_2kelas.py %*

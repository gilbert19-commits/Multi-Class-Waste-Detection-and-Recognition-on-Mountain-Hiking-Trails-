@echo off
REM =====================================================================
REM Training YOLOv8l dan YOLOv8x (dataset augmented), protokol sama dengan v8n/s/m
REM Output konsol juga disimpan di train_v8lx.log
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

"%PYTHON_CMD%" train_v8lx.py %*

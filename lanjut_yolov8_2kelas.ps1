# Penunggu antrian 2 kelas (2026-11-01, permintaan pengguna: langsung lanjut ke YOLOv8).
# Menunggu proses training YOLOv12 (PID 58456, mulai 2026-10-31 17:20:58) selesai, lalu menjalankan
# YOLOv8 s, n, m, l, x lewat train_2kelas.py. Tidak pernah ada dua proses GPU bersamaan.
# Batal: buat file STOP_ANTRIAN_2kelas sebelum YOLOv12x selesai (dicek juga oleh train_2kelas.py per model).
$ErrorActionPreference = "Continue"
$root = "D:\Goji\gilbert\Halomok\Multi-Class-Waste-Detection-and-Recognition-on-Mountain-Hiking-Trails-"
Set-Location $root
$timeFile = Join-Path $root "waktu_training_2kelas.txt"
$waitPid = 58456
$waitStart = [datetime]"2026-10-31 17:20:58"

function Catat($msg) { Add-Content -Path $timeFile -Value $msg -Encoding utf8 }

function Masih-Jalan {
    $p = Get-Process -Id $waitPid -ErrorAction SilentlyContinue
    # Cek waktu mulai juga, supaya PID yang dipakai ulang proses lain tidak dianggap training
    return ($p -and [math]::Abs(($p.StartTime - $waitStart).TotalSeconds) -lt 5)
}

Catat "PENUNGGU YOLOv8 | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | PID penunggu $PID menunggu PID $waitPid (antrian YOLOv12) selesai"
while (Masih-Jalan) { Start-Sleep -Seconds 60 }

if (Test-Path (Join-Path $root "STOP_ANTRIAN_2kelas")) {
    Catat "PENUNGGU YOLOv8 | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | STOP_ANTRIAN_2kelas ditemukan, YOLOv8 tidak dijalankan"
    exit
}
Catat "PENUNGGU YOLOv8 | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | PID $waitPid selesai, mulai antrian YOLOv8"
& "C:\Users\User\.conda\envs\ultralytics_env\python.exe" (Join-Path $root "train_2kelas.py") "yolov8s,yolov8n,yolov8m,yolov8l,yolov8x"

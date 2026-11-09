"""
Laporan progres training 2 kelas (read-only): model yang sedang jalan, epoch, metrik validasi terakhir,
model yang sudah selesai, dan estimasi jam selesai tiap model dalam antrian.

Estimasi waktu per epoch untuk model yang belum mulai = waktu per epoch run baru yang sudah ada
(YOLOv12s untuk keluarga YOLOv12, YOLOv8s untuk YOLOv8 bila sudah ada, kalau belum pakai YOLOv12s)
x rasio median waktu per epoch run lama (model / acuan). Ini ESTIMASI kasar: batch AutoBatch,
imgsz dan jumlah citra berbeda dari run lama.
"""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "runs_2kelas"
QUEUE = ["yolov12s", "yolov12n", "yolov12m", "yolov12l", "yolov12x"]  # antrian aktif: YOLOv12 dulu
EPOCHS = 200


def old_run(m):
    v = m[-1]
    if m.startswith("yolov12"):
        return ROOT / "runs/detect/runs/augmented" / f"aug_waste_yolov12{v}" / "results.csv"
    return ROOT / "runs/detect/runs_baseline" / f"v8{v}" / "results.csv"


def read_results(p):
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    return d


def epoch_seconds(d):
    return d["time"].diff().fillna(d["time"].iloc[0])


def run_dir(m):
    for name in (f"{m}_2kelas_b8", f"{m}_2kelas"):
        if (PROJECT / name).exists():
            return PROJECT / name
    return None


def status(m):
    rd = run_dir(m)
    if rd is None:
        return "antri", None, None
    last = rd / "weights" / "last.pt"
    res = rd / "results.csv"
    d = read_results(res) if res.exists() else None
    if last.exists() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", 0) == -1:
        return "selesai", rd, d
    return "berjalan", rd, d


def main():
    old_med = {m: epoch_seconds(read_results(old_run(m))).median() for m in QUEUE}
    st = {m: status(m) for m in QUEUE}
    new_med = {m: epoch_seconds(d).median() for m, (s, _, d) in st.items() if d is not None and len(d)}

    def est_epoch(m):
        if m in new_med:
            return new_med[m]
        ref = "yolov8s" if m.startswith("yolov8") and "yolov8s" in new_med else "yolov12s"
        if ref not in new_med:
            return None
        return new_med[ref] * old_med[m] / old_med[ref]

    print(f"Waktu sekarang: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    print("| # | model | status | run | epoch | val mAP50 | val mAP50-95 | val P | val R | s/epoch | estimasi selesai |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    t = datetime.now()
    for i, m in enumerate(QUEUE, 1):
        s, rd, d = st[m]
        ep = met = ""
        if d is not None and len(d):
            r = d.iloc[-1]
            ep = f"{int(r['epoch'])}/{EPOCHS}"
            met = (f"{r['metrics/mAP50(B)']:.5f} | {r['metrics/mAP50-95(B)']:.5f} | "
                   f"{r['metrics/precision(B)']:.5f} | {r['metrics/recall(B)']:.5f}")
        else:
            met = " |  |  | "
        e = est_epoch(m)
        if s == "selesai":
            eta = "sudah selesai"
        elif e is None:
            eta = "TIDAK TERSEDIA (belum ada epoch selesai)"
        else:
            done = int(d.iloc[-1]["epoch"]) if d is not None and len(d) else 0
            if s == "berjalan" and d is not None and len(d):
                mtime = datetime.fromtimestamp((rd / "results.csv").stat().st_mtime)
                t = max(t, mtime + timedelta(seconds=e * (EPOCHS - done)))
            else:
                t = t + timedelta(seconds=e * (EPOCHS - done))
            eta = f"{t:%Y-%m-%d %H:%M}" + ("" if m in new_med else " (dari rasio run lama)")
        print(f"| {i} | {m} | {s} | {rd.name if rd else '-'} | {ep} | {met} | "
              f"{'' if e is None else f'{e:.1f}'} | {eta} |")


if __name__ == "__main__":
    main()

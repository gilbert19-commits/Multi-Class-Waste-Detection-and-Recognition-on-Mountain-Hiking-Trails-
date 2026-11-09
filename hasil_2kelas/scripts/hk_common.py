"""Konstanta dan fungsi bersama script hasil_2kelas. Hanya membaca runs_2kelas/ dan dataset-2kelas/."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset-2kelas"  # hasil ekstrak dataset_2class.zip
RUNS = ROOT / "runs_2kelas"
OUT = ROOT / "hasil_2kelas"
MIRROR = OUT / "_data"
CACHE = OUT / "_cache"
VALDIR = OUT / "_val"
SCALES = list("nsmlx")
FAMILIES = {"YOLOv8": "yolov8", "YOLOv12": "yolov12", "YOLO26": "yolo26"}
MODELS = [f"{p}{s}" for p in FAMILIES.values() for s in SCALES]
NAMES = ["kertas_plastik_puntungRokok", "logam_kaleng_kaca"]
PAPER = {"kertas_plastik_puntungRokok": "lightweight litter", "logam_kaleng_kaca": "metal and glass"}
SEED = 0
EVAL_ARGS = dict(imgsz=512, batch=16, conf=0.001, iou=0.7, augment=False, half=False, device=0, workers=4,
                 seed=SEED, deterministic=True, save_json=False, verbose=False)
SIZE_BINS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, np.inf)]
_RF = re.compile(r"(_jpe?g\.rf\.[a-f0-9]+|_png\.rf\.[a-f0-9]+|\.rf\.[a-f0-9]+).*$", re.IGNORECASE)


def family(m):
    return next(f for f, p in FAMILIES.items() if m[:-1] == p)


def label(m):
    return family(m) + m[-1]


def base_name(fname):
    return _RF.sub("", Path(fname).stem)


def source_of(fname):
    """Aturan pola nama file dari scripts/fase1_scene_groups.py."""
    src = base_name(fname)
    if re.match(r"^(batch|BATCH)_", src):
        return "UAVVaste"
    if re.fullmatch(r"\d+", src):
        return "TACO"
    if re.match(r"^(WhatsApp-Video|IMG_|GOPR)", src):
        return "Burangrang"
    return "Unknown"


def size_bin(area):
    return next(n for n, lo, hi in SIZE_BINS if lo <= area < hi)


def read_results(p):
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    return d


def training_seconds(d):
    """Kolom time = detik kumulatif per sesi, reset saat resume. Return (total detik, jumlah sesi)."""
    t = d["time"].to_numpy()
    total, prev = 0.0, 0.0
    for x in t:
        if x < prev:
            total += prev
        prev = x
    return total + prev, int((np.diff(t) < 0).sum()) + 1


def best_epoch(d):
    """Epoch best.pt: fitness = mAP50-95 (Metric.fitness 8.4), seri diambil yang terakhir (EarlyStopping pakai >=)."""
    f = d["metrics/mAP50-95(B)"].to_numpy()
    return int(d["epoch"].to_numpy()[np.nonzero(f == f.max())[0][-1]])


def simulate_stop(d, patience):
    """Epoch EarlyStopping akan berhenti untuk patience tertentu; None kalau tidak pernah terpicu."""
    best, be = -np.inf, 0
    for e, f in zip(d["epoch"], d["metrics/mAP50-95(B)"]):
        if f >= best:
            best, be = f, e
        if e - be >= patience:
            return int(e)
    return None


def paper_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                         "grid.linewidth": 0.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                         "savefig.dpi": 300})


COL_W = 8.8 / 2.54  # lebar satu kolom IEEE (inci)

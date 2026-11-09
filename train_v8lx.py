"""
Tambahan eksperimen YOLOv8l dan YOLOv8x (dataset augmented/data.yaml).

Protokol sama persis dengan v8s/v8m di runs/detect/runs_baseline (lihat resume_yolov8.py):
yang berbeda hanya model dan name.

- Run disimpan di runs/detect/runs_baseline/v8l dan v8x.
- Jika batch=16 kena CUDA OOM, model diulang dari awal dengan batch=8 di folder v8{l,x}_b8.
  Ultralytics 8.4 otomatis menurunkan batch saat OOM di epoch pertama (trainer._oom_retries);
  itu dicegat lewat callback supaya batch tidak berubah diam-diam di run yang sama.
- Jika folder run sudah punya last.pt, dilanjutkan dengan resume=True.
- Waktu mulai/selesai -> waktu_training_v8lx.txt, seluruh output konsol -> train_v8lx.log,
  batch yang dipakai -> catatan_batch.txt.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "train_v8lx.log"
TIME_FILE = ROOT / "waktu_training_v8lx.txt"
BATCH_FILE = ROOT / "catatan_batch.txt"


class Tee:
    """Tulis ke konsol dan ke file log sekaligus."""

    encoding = "utf-8"

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        self.log.write(s)
        self.log.flush()
        return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        self.log.flush()

    def isatty(self):
        try:
            return self.stream.isatty()
        except Exception:
            return False


# Pasang Tee sebelum ultralytics di-import agar logger-nya ikut menulis ke file
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
_log = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
sys.stdout = Tee(sys.stdout, _log)
sys.stderr = Tee(sys.stderr, _log)

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

PROJECT_DIR = ROOT / "runs" / "detect" / "runs_baseline"
DATA = ROOT / "augmented" / "data.yaml"
WORKERS = 0  # sama dengan v8n/s/m (dataloader multi-worker hang di Windows saat close_mosaic)
BATCH = 16
BATCH_OOM = 8

TRAIN_ARGS = dict(
    data=str(DATA),
    epochs=100,
    patience=100,
    batch=BATCH,
    imgsz=640,
    device="0",
    workers=WORKERS,
    seed=0,
    deterministic=True,
    optimizer="auto",
    lr0=0.01,
    momentum=0.937,
    cache=False,
    amp=True,
    pretrained=True,
    project=str(PROJECT_DIR),
    exist_ok=True,
)


class BatchChanged(Exception):
    pass


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[CATAT] {path.name}: {line}")


def is_oom(e):
    msg = str(e)
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower() or any(
        s in msg for s in ("CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_INTERNAL_ERROR")
    )


def guard_batch(expected):
    """Callback: hentikan training jika Ultralytics menurunkan batch otomatis setelah OOM."""

    def cb(trainer):
        if trainer.batch_size != expected:
            raise BatchChanged(f"Ultralytics menurunkan batch {expected} -> {trainer.batch_size} karena OOM")

    return cb


def run(model_name, name, batch):
    """Train (atau resume) satu run. Return 'selesai' / 'sudah_selesai'."""
    run_dir = PROJECT_DIR / name
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        print(f"[INFO] Resume {name} dari {last}")
        model = YOLO(str(last))
        model.add_callback("on_train_batch_end", guard_batch(batch))
        try:
            model.train(resume=True, workers=WORKERS)
        except AssertionError as e:
            # Ultralytics menolak resume jika training sudah selesai
            print(f"[SKIP] {name}: {e}")
            return "sudah_selesai"
    else:
        print(f"[INFO] Train {name} dari awal ({model_name}, batch={batch})")
        model = YOLO(model_name)
        model.add_callback("on_train_batch_end", guard_batch(batch))
        model.train(**{**TRAIN_ARGS, "batch": batch, "name": name})
    return "selesai"


def train_variant(v):
    model_name = f"yolov8{v}.pt"
    name16, name8 = f"v8{v}", f"v8{v}_b8"
    # Kalau sebelumnya sudah jatuh ke batch=8, jangan kembali ke batch=16
    attempts = [(name8, BATCH_OOM)] if (PROJECT_DIR / name8).exists() else [(name16, BATCH), (name8, BATCH_OOM)]

    for name, batch in attempts:
        t0 = time.time()
        append(TIME_FILE, f"{name} | mulai   | {now()} | batch={batch}")
        try:
            status = run(model_name, name, batch)
        except (BatchChanged, RuntimeError) as e:
            if not (isinstance(e, BatchChanged) or is_oom(e)):
                append(TIME_FILE, f"{name} | GAGAL   | {now()} | {type(e).__name__}: {e}")
                raise
            append(TIME_FILE, f"{name} | OOM     | {now()} | {e}")
            if batch == BATCH_OOM:
                append(BATCH_FILE, f"{model_name} | batch=TIDAK TERSEDIA | batch=8 juga CUDA OOM, training dihentikan")
                return
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        append(TIME_FILE, f"{name} | {status:<7} | {now()} | durasi sesi ini {dt / 3600:.3f} jam")
        reason = "batch=16 berjalan tanpa CUDA OOM" if batch == BATCH else "batch=16 kena CUDA OOM, diulang dari awal dengan batch=8"
        append(BATCH_FILE, f"{model_name} | run={name} | batch={batch} | {reason}")
        return


def main(variants):
    print(f"[INFO] {now()} ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
          f"CUDA {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"[INFO] Python {sys.executable}")
    for v in variants:
        train_variant(v)
    print(f"[INFO] {now()} Semua selesai.")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["l", "x"])

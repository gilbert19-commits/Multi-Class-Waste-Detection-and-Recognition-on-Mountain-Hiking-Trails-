"""
Training dataset 2 kelas (dataset-2kelas/data.yaml), berurutan sesuai prioritas.
Atas permintaan pengguna (2026-10-30): epochs=200 dan antrian default hanya YOLOv12 (s, n, m, l, x);
YOLOv8 dijalankan nanti lewat argumen. Run 100 epoch yang dibatalkan ada di runs_2kelas/_dibatalkan_e100/.

Argumen disalin dari args.yaml run lama (acuan hyperparameter saja, hasil tidak dibandingkan):
  YOLOv12 <- runs/detect/runs/augmented/aug_waste_yolov12s/args.yaml  (batch=-1 AutoBatch, workers=4)
  YOLOv8  <- runs/detect/runs_baseline/v8s/args.yaml                    (batch=16, workers=0)
Yang diganti: data, epochs=200 (permintaan pengguna), imgsz=512 (resize README 514x514 -> kelipatan stride 32 terdekat, dipilih pengguna),
name=<model>_2kelas, project=runs_2kelas. Bobot pretrained: yolov12*.pt / yolov8*.pt di root project.
Patience: yolov12s/n/m selesai dengan patience=100 (seperti run lama). Atas permintaan pengguna (2026-10-31)
model yang belum selesai (yolov12l dilanjutkan, yolov12x, semua YOLOv8) memakai patience=50.
Saat resume, state EarlyStopping (fitness & epoch terbaik) dipulihkan dari checkpoint/results.csv, karena
Ultralytics membuat EarlyStopping baru saat resume (hitungan epoch tanpa perbaikan akan mulai dari nol).

- Resume: jika runs_2kelas/<name>/weights/last.pt ada dan belum selesai -> resume=True.
- Lewati model yang sudah selesai (checkpoint last.pt dengan epoch == -1, ditulis Ultralytics di akhir training).
- YOLOv8 dengan batch tetap (yolov8s, yolov8n: 16): jika kena CUDA OOM (termasuk penurunan batch otomatis
  Ultralytics di epoch 1), diulang dari awal dengan batch=8 di folder <name>_b8 dan dicatat.
  yolov8m/l/x memakai AutoBatch (-1) atas permintaan pengguna (2026-11-02).
- "stop antrian": buat file STOP_ANTRIAN_2kelas di root project -> antrian berhenti setelah model
  yang sedang berjalan selesai (dicek sebelum setiap model dimulai).
- Output konsol -> train_2kelas.log; waktu mulai/selesai, batch, imgsz efektif -> waktu_training_2kelas.txt.

Pemakaian: python train_2kelas.py            (antrian default: semua YOLOv12)
           python train_2kelas.py yolov8s,yolov8n,yolov8m,yolov8l,yolov8x   (YOLOv8, nanti)
           python train_2kelas.py yolo26s,yolo26n,yolo26m,yolo26l,yolo26x   (YOLO26, permintaan 2026-11-04;
               argumen = acuan v12, batch -1, workers 4, patience 50; bobot yolo26*.pt di root project)
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "train_2kelas.log"
TIME_FILE = ROOT / "waktu_training_2kelas.txt"
STOP_FILE = ROOT / "STOP_ANTRIAN_2kelas"
PID_FILE = ROOT / "train_2kelas.pid"


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

DATA = ROOT / "dataset-2kelas" / "data.yaml"
PROJECT = ROOT / "runs_2kelas"
IMGSZ = 512
EPOCHS = 200
PATIENCE = 50  # model yang belum selesai per 2026-10-31; s/n/m selesai dengan 100
QUEUE = ["yolov12s", "yolov12n", "yolov12m", "yolov12l", "yolov12x"]
REF_ARGS = {
    "yolov12": ROOT / "runs" / "detect" / "runs" / "augmented" / "aug_waste_yolov12s" / "args.yaml",
    "yolov8": ROOT / "runs" / "detect" / "runs_baseline" / "v8s" / "args.yaml",
    # YOLO26 (permintaan pengguna 2026-11-04): tidak ada run YOLO26 lama di project; memakai acuan v12.
    # Acuan v12 dan v8 hanya berbeda di batch/workers, yang di sini sudah diseragamkan (-1 dan 4).
    "yolo26": ROOT / "runs" / "detect" / "runs" / "augmented" / "aug_waste_yolov12s" / "args.yaml",
}
# YOLOv8: acuan batch=16 (yolov8s & yolov8n dilatih dengan 16). Atas permintaan pengguna (2026-11-02)
# model yang belum mulai (yolov8m/l/x) memakai AutoBatch -1 seperti YOLOv12; run yang di-resume tetap
# memakai batch dari checkpoint-nya. Fallback OOM ke batch=8 hanya berlaku untuk batch tetap (> 0).
BATCH = {"yolov12": -1, "yolov8": -1, "yolo26": -1}
BATCH_OOM_V8 = 8
# workers: args.yaml acuan v8 memakai 0 (GPU menunggu data, util ~20%). Atas permintaan pengguna
# (2026-11-02) semua run memakai 4 seperti acuan v12; yolov8s sudah selesai dengan workers=0,
# yolov8n dilanjutkan dari epoch 67 dengan workers=4. Hanya mempengaruhi pemuatan data, bukan hyperparameter.
WORKERS = 4
# Kunci args.yaml yang tidak diteruskan: ditentukan ulang untuk run baru
DROP = {"model", "data", "imgsz", "epochs", "name", "project", "save_dir", "resume", "mode", "task"}


class BatchChanged(Exception):
    pass


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(line):
    with open(TIME_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[CATAT] {TIME_FILE.name}: {line}")


def family(model):
    return next(f for f in ("yolov12", "yolo26", "yolov8") if model.startswith(f))


def train_args(model, name, batch):
    ref = yaml.safe_load(REF_ARGS[family(model)].read_text())
    args = {k: v for k, v in ref.items() if k not in DROP}
    args.update(data=str(DATA), imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=batch, workers=WORKERS,
                name=name, project=str(PROJECT))
    return args


def is_done(last):
    if not last.exists():
        return False
    ck = torch.load(last, map_location="cpu", weights_only=False)
    return ck.get("epoch", 0) == -1


def is_oom(e):
    msg = str(e)
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower() or any(
        s in msg for s in ("CUBLAS_STATUS_ALLOC_FAILED", "CUDNN_STATUS_INTERNAL_ERROR")
    )


def add_callbacks(model, name, guard_batch=None):
    def on_start(trainer):
        restored = ""
        res = Path(trainer.save_dir) / "results.csv"
        if trainer.start_epoch > 0 and res.exists() and trainer.best_fitness is not None:
            # Pulihkan state EarlyStopping (dibuat baru oleh Ultralytics saat resume)
            import pandas as pd

            d = pd.read_csv(res)
            d.columns = [c.strip() for c in d.columns]
            d = d[d["epoch"] <= trainer.start_epoch]
            trainer.stopper.best_fitness = float(trainer.best_fitness)
            trainer.stopper.best_epoch = int(d.loc[d["metrics/mAP50-95(B)"].idxmax(), "epoch"])
            restored = (f" | EarlyStopping dipulihkan: best_epoch={trainer.stopper.best_epoch}, "
                        f"best_fitness={trainer.stopper.best_fitness:.5f}")
        append(f"{name} | awal training (epoch {trainer.start_epoch + 1}) | args.batch={trainer.args.batch} | "
               f"batch dipakai={trainer.batch_size} | imgsz efektif={trainer.args.imgsz} | "
               f"workers={trainer.args.workers} | epochs={trainer.epochs} | "
               f"patience={trainer.stopper.patience}{restored}")

    def on_fit_epoch_end(trainer):
        if trainer.epoch == trainer.start_epoch:
            append(f"{name} | epoch {trainer.epoch + 1} selesai | {now()} | batch dipakai={trainer.batch_size} | "
                   f"durasi epoch {trainer.epoch_time:.1f} s (termasuk validasi)")

    def on_end(trainer):
        append(f"{name} | akhir training | batch dipakai={trainer.batch_size} | epoch terakhir={trainer.epoch + 1}")

    def guard(trainer):
        if trainer.batch_size != guard_batch:
            raise BatchChanged(f"Ultralytics menurunkan batch {guard_batch} -> {trainer.batch_size} karena OOM")

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_end)
    if guard_batch is not None:
        model.add_callback("on_train_epoch_start", guard)
        model.add_callback("on_train_batch_end", guard)


def run(model_name, name, batch):
    """Train atau resume satu run."""
    last = PROJECT / name / "weights" / "last.pt"
    guard_batch = batch if family(model_name) == "yolov8" and batch > 0 else None
    if last.exists():
        print(f"[INFO] Resume {name} dari {last}")
        model = YOLO(str(last))
        add_callbacks(model, name, guard_batch)
        model.train(resume=True, patience=PATIENCE, workers=WORKERS)
    else:
        args = train_args(model_name, name, batch)
        print(f"[INFO] Train {name} dari awal ({model_name}.pt) | argumen: {args}")
        model = YOLO(str(ROOT / f"{model_name}.pt"))
        add_callbacks(model, name, guard_batch)
        model.train(**args)


def train_model(model_name):
    fam = family(model_name)
    name = f"{model_name}_2kelas"
    attempts = [(name, BATCH[fam])]
    if fam == "yolov8" and BATCH[fam] > 0:
        name8 = f"{name}_b8"
        # Kalau sebelumnya sudah jatuh ke batch=8, jangan kembali ke batch=16
        attempts = [(name8, BATCH_OOM_V8)] if (PROJECT / name8).exists() else attempts + [(name8, BATCH_OOM_V8)]

    for run_name, batch in attempts:
        if is_done(PROJECT / run_name / "weights" / "last.pt"):
            print(f"[SKIP] {run_name} sudah selesai")
            return
        t0 = time.time()
        append(f"{run_name} | mulai   | {now()} | batch={batch}")
        try:
            run(model_name, run_name, batch)
        except (BatchChanged, RuntimeError) as e:
            if not (isinstance(e, BatchChanged) or is_oom(e)) or fam != "yolov8":
                append(f"{run_name} | GAGAL   | {now()} | {type(e).__name__}: {str(e)[:300]}")
                raise
            append(f"{run_name} | OOM     | {now()} | batch={batch} | {str(e)[:300]}")
            torch.cuda.empty_cache()
            if batch == BATCH_OOM_V8:
                append(f"{run_name} | batch=8 juga CUDA OOM, model dilewati")
                return
            continue
        append(f"{run_name} | selesai | {now()} | durasi sesi ini {(time.time() - t0) / 3600:.3f} jam")
        return


def main(models):
    PID_FILE.write_text(str(os.getpid()))
    print(f"[INFO] {now()} PID {os.getpid()} | ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
          f"CUDA {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"[INFO] Python {sys.executable} | antrian: {models}")
    for m in models:
        if STOP_FILE.exists():
            append(f"ANTRIAN DIHENTIKAN | {now()} | {STOP_FILE.name} ditemukan, sebelum {m}")
            break
        try:
            train_model(m)
        except Exception as e:  # sudah dicatat sebagai GAGAL; lanjut ke model berikutnya
            print(f"[ERROR] {m}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
    print(f"[INFO] {now()} Antrian selesai.")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else QUEUE)

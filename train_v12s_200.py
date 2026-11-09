"""
YOLOv12s 200 epoch (permintaan pembimbing), training BARU dari yolov12s.pt (pretrained COCO),
bukan lanjutan checkpoint epoch 100.

Argumen diambil langsung dari args.yaml run 100 epoch yang lama, jadi identik; yang diganti hanya
epochs=200 dan name:
  (a) runs/detect/runs/augmented/aug_waste_yolov12s/args.yaml -> aug_waste_yolov12s_e200
  (b) runs/detect/runs/detect/waste_yolov12s/args.yaml        -> waste_yolov12s_e200
Run baru disimpan di folder induk yang sama dengan run lamanya.

- batch=-1 (AutoBatch) seperti run lama; batch yang dipilih dicatat di catatan_batch_v12s_200.txt.
  (Run lama: AutoBatch memilih batch=7, tersimpan di train_args checkpoint-nya.)
- Jika folder run sudah punya last.pt, dilanjutkan dengan resume=True (batch hasil AutoBatch ikut
  tersimpan di checkpoint, jadi tidak dihitung ulang).
- Waktu mulai/selesai -> waktu_training_v12s_200.txt, seluruh output konsol -> train_v12s_200.log.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "train_v12s_200.log"
TIME_FILE = ROOT / "waktu_training_v12s_200.txt"
BATCH_FILE = ROOT / "catatan_batch_v12s_200.txt"


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

RUNS = ROOT / "runs" / "detect" / "runs"
EPOCHS = 200
JOBS = {
    "a": dict(old=RUNS / "augmented" / "aug_waste_yolov12s", name="aug_waste_yolov12s_e200"),
    "b": dict(old=RUNS / "detect" / "waste_yolov12s", name="waste_yolov12s_e200"),
}
# Kunci args.yaml yang tidak diteruskan: ditentukan ulang oleh run baru
DROP = {"model", "epochs", "name", "project", "save_dir", "resume", "mode", "task"}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[CATAT] {path.name}: {line}")


def batch_logger(name):
    def on_start(trainer):
        append(BATCH_FILE, f"{name} | awal training (epoch {trainer.start_epoch + 1}) | "
                           f"args.batch={trainer.args.batch} | batch dipakai={trainer.batch_size}")

    def on_end(trainer):
        append(BATCH_FILE, f"{name} | akhir training | batch dipakai={trainer.batch_size}")

    return on_start, on_end


def train_job(key):
    job = JOBS[key]
    old_args = yaml.safe_load((job["old"] / "args.yaml").read_text())
    project = job["old"].parent
    run_dir = project / job["name"]
    last = run_dir / "weights" / "last.pt"

    t0 = time.time()
    append(TIME_FILE, f"{job['name']} | mulai   | {now()}")
    if last.exists():
        print(f"[INFO] Resume {job['name']} dari {last}")
        model = YOLO(str(last))
        on_start, on_end = batch_logger(job["name"])
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_end", on_end)
        try:
            model.train(resume=True)
        except AssertionError as e:
            # Ultralytics menolak resume jika training sudah selesai
            print(f"[SKIP] {job['name']}: {e}")
            append(TIME_FILE, f"{job['name']} | sudah selesai sebelumnya | {now()}")
            return
    else:
        args = {k: v for k, v in old_args.items() if k not in DROP}
        args.update(epochs=EPOCHS, name=job["name"], project=str(project))
        print(f"[INFO] Train {job['name']} dari awal (yolov12s.pt, epochs={EPOCHS}, batch={args['batch']}, "
              f"data={args['data']}) | argumen lain disalin dari {job['old'] / 'args.yaml'}")
        model = YOLO(str(ROOT / "yolov12s.pt"))
        on_start, on_end = batch_logger(job["name"])
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_end", on_end)
        model.train(**args)
    append(TIME_FILE, f"{job['name']} | selesai | {now()} | durasi sesi ini {(time.time() - t0) / 3600:.3f} jam")


def main(keys):
    print(f"[INFO] {now()} ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
          f"CUDA {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"[INFO] Python {sys.executable}")
    for k in keys:
        try:
            train_job(k)
        except Exception as e:
            append(TIME_FILE, f"{JOBS[k]['name']} | GAGAL   | {now()} | {type(e).__name__}: {str(e)[:200]}")
            raise
    print(f"[INFO] {now()} Semua selesai.")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["a", "b"])

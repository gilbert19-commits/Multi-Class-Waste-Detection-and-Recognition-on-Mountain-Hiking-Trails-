"""
Lanjutkan eksperimen YOLOv8 (dataset augmented/data.yaml) yang macet di epoch 90 v8n.

- v8n: resume dari last.pt
- v8s, v8m: train dari awal dengan hyperparameter yang sama seperti v8n

workers=0 dipakai karena di Windows dataloader multi-worker hang saat close_mosaic
membuat ulang dataloader (epoch 91/100).
"""

import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "runs" / "detect" / "runs_baseline"
DATA = ROOT / "augmented" / "data.yaml"
WORKERS = 0


def main(variants):
    for v in variants:
        run_dir = PROJECT_DIR / f"v8{v}"
        last = run_dir / "weights" / "last.pt"
        if last.exists():
            print(f"[INFO] Resume v8{v} dari {last}")
            try:
                YOLO(str(last)).train(resume=True, workers=WORKERS)
            except AssertionError as e:
                # Ultralytics menolak resume jika training sudah selesai
                print(f"[SKIP] v8{v}: {e}")
        else:
            print(f"[INFO] Train v8{v} dari awal")
            YOLO(f"yolov8{v}.pt").train(
                data=str(DATA),
                epochs=100,
                batch=16,
                imgsz=640,
                device="0",
                workers=WORKERS,
                project=str(PROJECT_DIR),
                name=f"v8{v}",
                exist_ok=True,
            )


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["n", "s", "m"])

"""
Script Training YOLOv12 untuk Dataset yang Telah Diaugmentasi (Version 2)
Folder Dataset: augmented/
Output Runs: runs/augmented/
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Root directory proyek
ROOT_DIR = Path(__file__).resolve().parent.parent


def fix_data_yaml_paths(yaml_path: Path) -> Path:
    """Memastikan path absolut terkonfigurasi di data.yaml agar bebas error path."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    dataset_root = yaml_path.parent.resolve()
    data["path"] = str(dataset_root).replace("\\", "/")

    if "train" in data and not os.path.isabs(str(data["train"])):
        data["train"] = "train/images"
    if "val" in data and not os.path.isabs(str(data["val"])):
        data["val"] = "valid/images"
    if "test" in data and not os.path.isabs(str(data["test"])):
        data["test"] = "test/images"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(description="Training YOLOv12 pada Dataset Augmented (Versi 2)")

    # Model & Data
    parser.add_argument(
        "--model",
        type=str,
        default="yolov12s.pt",
        help="Varian model pre-trained (yolov12n.pt, yolov12s.pt, yolov12m.pt, yolov12l.pt, yolov12x.pt)",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=str(Path(__file__).resolve().parent / "data.yaml"),
        help="Path ke data.yaml dataset augmentasi",
    )

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epoch training (default: 100)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (default: -1 untuk AutoBatch)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi gambar input (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device GPU (default: '0') atau 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer: auto, SGD, Adam, AdamW")

    # Output Management
    parser.add_argument("--project-dir", type=str, default="runs/augmented", help="Direktori penyimpanan run augmentasi")
    parser.add_argument("--name", type=str, default=None, help="Nama eksperimen (default otomatis: aug_waste_<model>)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("      TRAINING YOLOV12 - AUGMENTED DATASET (VERSION 2)      ")
    print("=" * 70)

    # 1. Hardware Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[HARDWARE] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        device = args.device
    else:
        print("[HARDWARE] GPU tidak ditemukan, menggunakan CPU.")
        device = "cpu"

    # 2. Cek dataset data.yaml
    yaml_path = Path(args.data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di: {yaml_path}")
    yaml_path = fix_data_yaml_paths(yaml_path)
    print(f"[DATASET] data.yaml: {yaml_path}")

    # 3. Cari model weights (bisa di folder root atau di folder saat ini)
    model_name = args.model
    if not Path(model_name).exists() and (ROOT_DIR / model_name).exists():
        model_weight = str(ROOT_DIR / model_name)
    else:
        model_weight = model_name

    print(f"[MODEL] Menggunakan pre-trained weights: {model_weight}")
    model = YOLO(model_weight)

    # 4. Training
    exp_name = args.name if args.name else f"aug_waste_{Path(model_name).stem}"
    print(f"[RUN] Eksperimen: {exp_name}")
    print(f"[RUN] Output: {Path(args.project_dir) / exp_name}")

    train_results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        lr0=args.lr0,
        patience=args.patience,
        optimizer=args.optimizer,
        project=args.project_dir,
        name=exp_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # 5. Evaluasi pada Test Split
    best_weight = Path(args.project_dir) / exp_name / "weights" / "best.pt"
    if best_weight.exists():
        print("\n[EVALUASI] Menjalankan evaluasi pada Test Split (unseen data)...")
        best_model = YOLO(str(best_weight))
        test_metrics = best_model.val(data=str(yaml_path), split="test", project=args.project_dir, name=f"{exp_name}_test_eval")
        print("\n" + "=" * 70)
        print(f"[HASIL TEST SET - {exp_name}]")
        print(f"  - mAP@50    : {test_metrics.box.map50:.4f}")
        print(f"  - mAP@50-95 : {test_metrics.box.map:.4f}")
        print(f"  - Presisi   : {test_metrics.box.mp:.4f}")
        print(f"  - Recall    : {test_metrics.box.mr:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()

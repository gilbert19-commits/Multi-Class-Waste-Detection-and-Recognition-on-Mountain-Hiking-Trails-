"""
Script Otomatis untuk Melatih Semua Varian YOLOv12 Secara Berurutan:
- YOLOv12n (Nano)
- YOLOv12s (Small)
- YOLOv12m (Medium)
- YOLOv12l (Large)
- YOLOv12x (XLarge)

Parameter:
- batch: -1 (AutoBatch - otomatis memaksimalkan VRAM GPU)
- epochs: 100
- device: 0 (NVIDIA RTX 5070)
- imgsz: 640

Menghasilkan tabel ringkasan perbandingan mAP50, mAP50-95, presisi, recall, dan waktu training.
"""

import sys
import time
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# Import helper normalisasi path dari train.py
from train import ensure_dataset, fix_data_yaml_paths


VARIANTS = [
    {"name": "YOLOv12n", "model_file": "yolov12n.pt"},
    {"name": "YOLOv12s", "model_file": "yolov12s.pt"},
    {"name": "YOLOv12m", "model_file": "yolov12m.pt"},
    {"name": "YOLOv12l", "model_file": "yolov12l.pt"},
    {"name": "YOLOv12x", "model_file": "yolov12x.pt"},
]


def run_benchmark(
    epochs: int = 100,
    batch: int = -1,
    device: str = "0",
    imgsz: int = 640,
    dataset_dir: str = "TA-1",
    project_dir: str = "runs/detect",
):
    print("=" * 75)
    print("      BENCHMARK TRAINING SEMUA VARIAN YOLOv12 (N, S, M, L, X)      ")
    print("=" * 75)
    print(f"Batch Size   : {batch} (AutoBatch)")
    print(f"Epochs       : {epochs}")
    print(f"Device       : {device}")
    print(f"Image Size   : {imgsz}")
    print(f"Total Varian : {len(VARIANTS)}")
    print("=" * 75 + "\n")

    # 1. Pastikan dataset siap dan normalisasi data.yaml
    yaml_path = ensure_dataset(dataset_dir=dataset_dir)
    yaml_path = fix_data_yaml_paths(yaml_path)

    summary_records = []

    for idx, var in enumerate(VARIANTS, 1):
        v_name = var["name"]
        model_file = var["model_file"]
        exp_name = f"waste_{model_file.replace('.pt', '')}"

        print("\n" + "#" * 75)
        print(f"[{idx}/{len(VARIANTS)}] Memulai Training: {v_name} ({model_file})")
        print(f"Eksperimen : {exp_name}")
        print("#" * 75)

        start_time = time.time()
        try:
            model = YOLO(model_file)

            # Training
            train_results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=4,
                project=project_dir,
                name=exp_name,
                exist_ok=True,
                plots=True,
                verbose=True,
            )

            elapsed_minutes = (time.time() - start_time) / 60.0

            # Evaluasi Test Split
            best_weight = Path(project_dir) / exp_name / "weights" / "best.pt"
            if best_weight.exists():
                print(f"\n[EVALUASI] Menjalankan test split untuk {v_name}...")
                test_model = YOLO(str(best_weight))
                test_metrics = test_model.val(
                    data=str(yaml_path),
                    split="test",
                    project=project_dir,
                    name=f"{exp_name}_test_eval",
                )

                map50 = float(test_metrics.box.map50)
                map50_95 = float(test_metrics.box.map)
                precision = float(test_metrics.box.mp)
                recall = float(test_metrics.box.mr)
            else:
                map50, map50_95, precision, recall = 0.0, 0.0, 0.0, 0.0

            summary_records.append({
                "Model Variant": v_name,
                "Weights": model_file,
                "mAP@50": round(map50, 4),
                "mAP@50-95": round(map50_95, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "Training Time (min)": round(elapsed_minutes, 2),
                "Best Weights": str(best_weight),
            })

            print(f"\n[SUKSES] Selesai training {v_name} dalam {elapsed_minutes:.1f} menit.")
            print(f"-> mAP@50: {map50:.4f} | mAP@50-95: {map50_95:.4f}")

        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan saat training {v_name}: {e}")
            summary_records.append({
                "Model Variant": v_name,
                "Weights": model_file,
                "mAP@50": "ERROR",
                "mAP@50-95": "ERROR",
                "Precision": "ERROR",
                "Recall": "ERROR",
                "Training Time (min)": "-",
                "Best Weights": "-",
            })

    # Simpan dan Tampilkan Tabel Ringkasan
    print("\n" + "=" * 75)
    print("                    TABEL RINGKASAN HASIL BENCHMARK                    ")
    print("=" * 75)
    df = pd.DataFrame(summary_records)
    print(df.to_string(index=False))
    print("=" * 75)

    output_dir = Path(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_summary.csv"
    md_path = output_dir / "benchmark_summary.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"\nRingkasan telah disimpan ke:")
    print(f"  - CSV: {csv_path.resolve()}")
    print(f"  - MD : {md_path.resolve()}")


if __name__ == "__main__":
    run_benchmark(epochs=100, batch=-1, device="0", imgsz=640)

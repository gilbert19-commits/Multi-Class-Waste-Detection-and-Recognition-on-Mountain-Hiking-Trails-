"""
Script Otomatis untuk Melatih Semua Varian YOLOv12 pada Dataset Augmented (Versi 2):
- YOLOv12n (Nano)
- YOLOv12s (Small)
- YOLOv12m (Medium)
- YOLOv12l (Large)
- YOLOv12x (XLarge)

Parameter:
- batch: -1 (AutoBatch)
- epochs: 100
- device: 0 (NVIDIA RTX 5070)
- imgsz: 640
- Output: runs/augmented/
"""

import time
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from train_augmented import fix_data_yaml_paths

ROOT_DIR = Path(__file__).resolve().parent.parent

VARIANTS = [
    {"name": "YOLOv12n", "file": "yolov12n.pt"},
    {"name": "YOLOv12s", "file": "yolov12s.pt"},
    {"name": "YOLOv12m", "file": "yolov12m.pt"},
    {"name": "YOLOv12l", "file": "yolov12l.pt"},
    {"name": "YOLOv12x", "file": "yolov12x.pt"},
]


def run_all_augmented(
    epochs: int = 100,
    batch: int = -1,
    device: str = "0",
    imgsz: int = 640,
    project_dir: str = "runs/augmented",
):
    print("=" * 75)
    print("   BENCHMARK SEMUA VARIAN YOLOV12 - AUGMENTED DATASET (VERSION 2)   ")
    print("=" * 75)

    yaml_path = Path(__file__).resolve().parent / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di: {yaml_path}")
    yaml_path = fix_data_yaml_paths(yaml_path)

    summary_records = []

    for idx, var in enumerate(VARIANTS, 1):
        v_name = var["name"]
        model_file = var["file"]
        weight_path = str(ROOT_DIR / model_file if (ROOT_DIR / model_file).exists() else model_file)
        exp_name = f"aug_waste_{model_file.replace('.pt', '')}"

        print("\n" + "#" * 75)
        print(f"[{idx}/{len(VARIANTS)}] Memulai Training Augmented: {v_name}")
        print(f"Weights : {weight_path}")
        print(f"Run Dir : {Path(project_dir) / exp_name}")
        print("#" * 75)

        start_time = time.time()
        try:
            model = YOLO(weight_path)

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

            # Evaluasi pada Test Split
            best_weight = Path(project_dir) / exp_name / "weights" / "best.pt"
            if best_weight.exists():
                print(f"\n[EVALUASI] Menjalankan test split untuk {v_name}...")
                test_model = YOLO(str(best_weight))
                test_metrics = test_model.val(
                    data=str(yaml_path),
                    split="test",
                    device=device,
                    workers=0,
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
            print(f"\n[ERROR] Kesalahan pada {v_name}: {e}")
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

    # Simpan dan Tampilkan Hasil
    df = pd.DataFrame(summary_records)
    print("\n" + "=" * 75)
    print("            TABEL BENCHMARK AUGMENTED DATASET (VERSI 2)            ")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75)

    out_dir = Path(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_summary.csv"
    md_path = out_dir / "benchmark_summary.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    print(f"\nRingkasan benchmark tersimpan di:")
    print(f"  - CSV: {csv_path.resolve()}")
    print(f"  - MD : {md_path.resolve()}")


if __name__ == "__main__":
    run_all_augmented(epochs=100, batch=-1, device="0", imgsz=640)

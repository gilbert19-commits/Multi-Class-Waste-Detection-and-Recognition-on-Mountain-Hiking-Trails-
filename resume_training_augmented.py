"""
Script Otomatis untuk Me-resume Training yang Terhenti & Melanjutkan Sisa Varian (YOLOv12l & YOLOv12x)
serta Evaluasi Lengkap Test Split untuk Semua Varian (YOLOv12n, s, m, l, x).
"""

import sys
import time
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
YAML_PATH = ROOT_DIR / "augmented" / "data.yaml"
RUNS_AUG_DIR = ROOT_DIR / "runs" / "detect" / "runs" / "augmented"
PROJECT_DIR = "runs/augmented"


def fix_data_yaml_paths(yaml_path: Path):
    """Pastikan data.yaml memiliki path absolut yang valid."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    dataset_root = yaml_path.parent.resolve()
    data["path"] = str(dataset_root).replace("\\", "/")

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def resume_and_finish():
    fix_data_yaml_paths(YAML_PATH)

    print("=" * 75)
    print("   RESUME & SELESAIKAN TRAINING YOLOV12 AUGMENTED (RTX 5070)")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. RESUME YOLOV12l (Large)
    # -------------------------------------------------------------
    l_dir = RUNS_AUG_DIR / "aug_waste_yolov12l"
    l_csv = l_dir / "results.csv"
    l_last = l_dir / "weights" / "last.pt"

    l_completed = False
    if l_csv.exists():
        try:
            df_l = pd.read_csv(l_csv)
            if len(df_l) >= 100:
                l_completed = True
        except Exception:
            pass

    if l_completed:
        print("\n[INFO] YOLOv12l sudah selesai 100 epochs sebelumnya. Melewati tahap ini.")
    elif l_last.exists():
        print("\n" + "#" * 75)
        print("[RESUME] Melanjutkan training YOLOv12l langsung dari checkpoint: last.pt...")
        print("#" * 75)
        model_l = YOLO(str(l_last))
        model_l.train(resume=True)
        print("\n[SUKSES] Training YOLOv12l selesai!")
    else:
        print("\n" + "#" * 75)
        print("[START] Memulai training YOLOv12l dari awal (weights: yolov12l.pt)...")
        print("#" * 75)
        model_l = YOLO(str(ROOT_DIR / "yolov12l.pt"))
        model_l.train(
            data=str(YAML_PATH),
            epochs=100,
            batch=-1,
            imgsz=640,
            device="0",
            workers=4,
            project=PROJECT_DIR,
            name="aug_waste_yolov12l",
            exist_ok=True,
            plots=True,
            verbose=True,
        )

    # -------------------------------------------------------------
    # 2. TRAINING YOLOV12x (XLarge)
    # -------------------------------------------------------------
    x_dir = RUNS_AUG_DIR / "aug_waste_yolov12x"
    x_csv = x_dir / "results.csv"
    x_last = x_dir / "weights" / "last.pt"

    x_completed = False
    if x_csv.exists():
        try:
            df_x = pd.read_csv(x_csv)
            if len(df_x) >= 100:
                x_completed = True
        except Exception:
            pass

    if x_completed:
        print("\n[INFO] YOLOv12x sudah selesai 100 epochs sebelumnya. Melewati tahap ini.")
    elif x_last.exists():
        print("\n" + "#" * 75)
        print("[RESUME] Melanjutkan training YOLOv12x dari checkpoint: last.pt...")
        print("#" * 75)
        model_x = YOLO(str(x_last))
        model_x.train(resume=True)
        print("\n[SUKSES] Training YOLOv12x selesai!")
    else:
        print("\n" + "#" * 75)
        print("[START] Memulai training YOLOv12x (100 epochs, AutoBatch)...")
        print(f"Weights : {ROOT_DIR / 'yolov12x.pt'}")
        print("#" * 75)
        model_x = YOLO(str(ROOT_DIR / "yolov12x.pt"))
        model_x.train(
            data=str(YAML_PATH),
            epochs=100,
            batch=-1,
            imgsz=640,
            device="0",
            workers=4,
            project=PROJECT_DIR,
            name="aug_waste_yolov12x",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        print("\n[SUKSES] Training YOLOv12x selesai!")

    # -------------------------------------------------------------
    # 3. EVALUASI TEST SET UNTUK SELURUH VARIAN (n, s, m, l, x)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("   MENJALANKAN EVALUASI TEST SET (UNSEEN DATA) UNTUK SEMUA MODEL")
    print("=" * 75)

    variants = [
        {"name": "YOLOv12n", "exp": "aug_waste_yolov12n"},
        {"name": "YOLOv12s", "exp": "aug_waste_yolov12s"},
        {"name": "YOLOv12m", "exp": "aug_waste_yolov12m"},
        {"name": "YOLOv12l", "exp": "aug_waste_yolov12l"},
        {"name": "YOLOv12x", "exp": "aug_waste_yolov12x"},
    ]

    summary_records = []
    eval_project_dir = "runs/augmented/eval"

    for var in variants:
        v_name = var["name"]
        exp_name = var["exp"]
        best_pt = RUNS_AUG_DIR / exp_name / "weights" / "best.pt"

        if not best_pt.exists():
            print(f"\n[SKIP] {v_name}: File best.pt belum ditemukan di {best_pt}")
            continue

        print(f"\n[EVALUASI] Menjalankan test split untuk {v_name}...")
        test_model = YOLO(str(best_pt))
        test_metrics = test_model.val(
            data=str(YAML_PATH),
            split="test",
            device="0",
            workers=0,
            project=eval_project_dir,
            name=f"{exp_name}_test_eval",
            exist_ok=True,
        )

        map50 = float(test_metrics.box.map50)
        map50_95 = float(test_metrics.box.map)
        precision = float(test_metrics.box.mp)
        recall = float(test_metrics.box.mr)

        summary_records.append({
            "Model Variant": v_name,
            "mAP@50": round(map50, 4),
            "mAP@50-95": round(map50_95, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Best Weights": str(best_pt),
        })

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        print("\n" + "=" * 75)
        print("     HASIL BENCHMARK LENGKAP TEST SPLIT (AUGMENTED DATASET)     ")
        print("=" * 75)
        print(df_summary.to_string(index=False))
        print("=" * 75)

        out_csv = RUNS_AUG_DIR / "augmented_benchmark_summary.csv"
        out_md = RUNS_AUG_DIR / "augmented_benchmark_summary.md"
        df_summary.to_csv(out_csv, index=False)
        df_summary.to_markdown(out_md, index=False)

        print(f"\nRingkasan benchmark berhasil disimpan di:")
        print(f"  - CSV: {out_csv}")
        print(f"  - MD : {out_md}")


if __name__ == "__main__":
    resume_and_finish()

"""
Script untuk mengekstrak metrik per-model dan per-kelas secara komprehensif:
- Precision, Recall, F1
- mAP@50 (4C & 3C), mAP@50-95 (4C & 3C)
- AP@50 dan AP@50-95 per kelas (kertas_tisu, logam_kaleng_kaca, plastik, puntung_rokok)
- Params (M), GFLOPs, dan inference latency (ms)
Menyimpan ke analysis/tables/C_per_class_full.csv dan analysis/tables/C_per_class_strict.csv.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT_DIR / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"

OFFICIAL_CLASSES = ['kertas_tisu', 'logam_kaleng_kaca', 'plastik', 'puntung_rokok']

CHECKPOINTS = [
    {"variant": "YOLOv12n", "group": "Moderat (2.27x)", "weights": ROOT_DIR / "runs/detect/runs/detect/waste_yolov12n/weights/best.pt", "gflops": 7.3},
    {"variant": "YOLOv12s", "group": "Moderat (2.27x)", "weights": ROOT_DIR / "runs/detect/runs/detect/waste_yolov12s/weights/best.pt", "gflops": 21.6},
    {"variant": "YOLOv12m", "group": "Moderat (2.27x)", "weights": ROOT_DIR / "runs/detect/runs/detect/waste_yolov12m/weights/best.pt", "gflops": 49.3},
    {"variant": "YOLOv12l", "group": "Moderat (2.27x)", "weights": ROOT_DIR / "runs/detect/runs/detect/waste_yolov12l/weights/best.pt", "gflops": 95.1},
    {"variant": "YOLOv12x", "group": "Moderat (2.27x)", "weights": ROOT_DIR / "runs/detect/runs/detect/waste_yolov12x/weights/best.pt", "gflops": 208.4},
    {"variant": "YOLOv12n", "group": "Intensif (6.81x)", "weights": ROOT_DIR / "runs/detect/runs/augmented/aug_waste_yolov12n/weights/best.pt", "gflops": 7.3},
    {"variant": "YOLOv12s", "group": "Intensif (6.81x)", "weights": ROOT_DIR / "runs/detect/runs/augmented/aug_waste_yolov12s/weights/best.pt", "gflops": 21.6},
    {"variant": "YOLOv12m", "group": "Intensif (6.81x)", "weights": ROOT_DIR / "runs/detect/runs/augmented/aug_waste_yolov12m/weights/best.pt", "gflops": 49.3},
    {"variant": "YOLOv12l", "group": "Intensif (6.81x)", "weights": ROOT_DIR / "runs/detect/runs/augmented/aug_waste_yolov12l/weights/best.pt", "gflops": 95.1},
    {"variant": "YOLOv12x", "group": "Intensif (6.81x)", "weights": ROOT_DIR / "runs/detect/runs/augmented/aug_waste_yolov12x/weights/best.pt", "gflops": 208.4},
]

def extract_detailed_metrics():
    test_configs = [
        {"name": "full", "yaml": ANALYSIS_DIR / "test_full.yaml"},
        {"name": "strict", "yaml": ANALYSIS_DIR / "test_strict.yaml"},
    ]

    for tc in test_configs:
        tname = tc["name"]
        tyaml = tc["yaml"]
        records = []

        print(f"\n=======================================================")
        print(f" Ekstraksi Metrik Detail Per-Kelas: {tname.upper()}")
        print(f"=======================================================")

        for item in CHECKPOINTS:
            vname = item["variant"]
            grp = item["group"]
            wpath = item["weights"]
            gflops = item["gflops"]

            print(f"Evaluating {vname} [{grp}]...")
            model = YOLO(str(wpath))
            param_m = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)

            metrics = model.val(
                data=str(tyaml),
                imgsz=640,
                conf=0.001,
                iou=0.6,
                split="test",
                device="0",
                verbose=False,
                plots=False
            )

            prec = float(metrics.box.mp)
            rec = float(metrics.box.mr)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            map50_4c = float(metrics.box.map50)
            map50_95_4c = float(metrics.box.map)

            ap50_list = [float(x) for x in metrics.box.ap50]
            ap_list = [float(x) for x in metrics.box.ap]

            map50_3c = float(np.mean(ap50_list[:3])) if len(ap50_list) >= 3 else map50_4c
            map50_95_3c = float(np.mean(ap_list[:3])) if len(ap_list) >= 3 else map50_95_4c
            inf_ms = round(float(metrics.speed.get("inference", 0.0)), 2)

            records.append({
                "Model Variant": vname,
                "Regime": grp,
                "Params (M)": param_m,
                "GFLOPs": gflops,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "mAP@50 (4C)": round(map50_4c, 4),
                "mAP@50-95 (4C)": round(map50_95_4c, 4),
                "mAP@50 (3C)": round(map50_3c, 4),
                "mAP@50-95 (3C)": round(map50_95_3c, 4),
                "AP50_kertas": round(ap50_list[0], 4) if len(ap50_list) > 0 else 0.0,
                "AP50-95_kertas": round(ap_list[0], 4) if len(ap_list) > 0 else 0.0,
                "AP50_logam": round(ap50_list[1], 4) if len(ap50_list) > 1 else 0.0,
                "AP50-95_logam": round(ap_list[1], 4) if len(ap_list) > 1 else 0.0,
                "AP50_plastik": round(ap50_list[2], 4) if len(ap50_list) > 2 else 0.0,
                "AP50-95_plastik": round(ap_list[2], 4) if len(ap_list) > 2 else 0.0,
                "AP50_puntung": round(ap50_list[3], 4) if len(ap50_list) > 3 else 0.0,
                "AP50-95_puntung": round(ap_list[3], 4) if len(ap_list) > 3 else 0.0,
                "Inference (ms)": inf_ms,
            })

        df = pd.DataFrame(records)
        out_csv = TABLES_DIR / f"C_per_class_{tname}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[OK] Disimpan di: {out_csv}")
        print(df[["Model Variant", "Regime", "mAP@50 (4C)", "mAP@50 (3C)", "AP50_kertas", "AP50_logam", "AP50_plastik", "AP50_puntung"]].to_string())

if __name__ == "__main__":
    extract_detailed_metrics()

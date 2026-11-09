"""
Script Training YOLOv12 / Ultralytics untuk Multi-Class Waste Detection
Dataset: Roboflow (workspace: weapon-mpr3p, project: ta-rhzr5, version: 1)
Kelas: ['kertas_tisu', 'logam_kaleng_kaca', 'plastik', 'puntung_rokok']
Hardware: NVIDIA GPU (CUDA) / CPU
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


# URL resmi rilis bobot pre-trained YOLOv12 dari repository sunsmarterjie/yolov12
YOLOV12_WEIGHTS_URLS = {
    "yolov12n.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt",
    "yolov12s.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12s.pt",
    "yolov12m.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12m.pt",
    "yolov12l.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12l.pt",
    "yolov12x.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt",
}


def download_pretrained_weights(model_name: str) -> str:
    """Mengunduh bobot pre-trained YOLOv12 jika belum tersedia di lokal."""
    weight_path = Path(model_name)
    if weight_path.exists():
        print(f"[INFO] Bobot pre-trained lokal ditemukan: {weight_path}")
        return str(weight_path)

    filename = weight_path.name
    if filename in YOLOV12_WEIGHTS_URLS:
        url = YOLOV12_WEIGHTS_URLS[filename]
        print(f"[INFO] Mengunduh pre-trained model {filename} dari {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"[INFO] Berhasil mengunduh: {filename}")
            return filename
        except Exception as e:
            print(f"[WARNING] Gagal mengunduh {filename} secara otomatis: {e}")
            print(f"[INFO] Mencoba melanjutkan dengan nama model langsung...")
            return model_name
    return model_name


def ensure_dataset(
    dataset_dir: str = "TA-1",
    api_key: str = os.environ.get("ROBOFLOW_API_KEY"),
    workspace: str = "weapon-mpr3p",
    project_name: str = "ta-rhzr5",
    version_num: int = 1,
) -> Path:
    """
    Memeriksa keberadaan dataset di lokal.
    Jika belum ada, unduh otomatis dari Roboflow.
    """
    dataset_path = Path(dataset_dir)
    yaml_file = dataset_path / "data.yaml"

    if yaml_file.exists():
        print(f"[INFO] Dataset ditemukan di folder: {dataset_path.resolve()}")
        print(f"[INFO] Konfigurasi dataset: {yaml_file.resolve()}")
        return yaml_file

    print(f"[INFO] Folder dataset '{dataset_dir}' belum ditemukan.")
    print(f"[INFO] Mengunduh dataset dari Roboflow (Workspace: {workspace}, Project: {project_name}, Version: {version_num})...")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] Library roboflow belum terinstal. Silakan jalankan: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    downloaded_dataset = version.download("yolov12")

    downloaded_path = Path(downloaded_dataset.location)
    downloaded_yaml = downloaded_path / "data.yaml"

    if not downloaded_yaml.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {downloaded_path}")

    print(f"[INFO] Dataset berhasil diunduh ke: {downloaded_path.resolve()}")
    return downloaded_yaml


def fix_data_yaml_paths(yaml_path: Path) -> Path:
    """
    Memastikan path di dalam data.yaml terkonfigurasi dengan benar
    agar tidak terjadi error path not found saat training di Ultralytics.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    dataset_root = yaml_path.parent.resolve()
    # Atur 'path' absolut ke folder root dataset
    data["path"] = str(dataset_root).replace("\\", "/")

    # Normalisasi train, val, test
    if "train" in data and not os.path.isabs(str(data["train"])):
        # Jika ada '../' bawaan Roboflow, ubah ke folder relatif dataset
        data["train"] = "train/images"
    if "val" in data and not os.path.isabs(str(data["val"])):
        data["val"] = "valid/images"
    if "test" in data and not os.path.isabs(str(data["test"])):
        data["test"] = "test/images"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[INFO] Path data.yaml telah dinormalisasi ke: {dataset_root}")
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(description="Training YOLOv12 pada Dataset Roboflow")

    # Model & Data
    parser.add_argument(
        "--model",
        type=str,
        default="yolov12s.pt",
        help="Model pre-trained (pilihan: yolov12n.pt, yolov12s.pt, yolov12m.pt, yolov12l.pt, yolo11s.pt, dll.)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="TA-1",
        help="Folder tempat dataset disimpan (default: TA-1)",
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=None,
        help="Path manual ke data.yaml (opsional)",
    )

    # Roboflow Credential
    parser.add_argument("--api-key", type=str, default=os.environ.get("ROBOFLOW_API_KEY"), help="Default: variabel lingkungan ROBOFLOW_API_KEY")
    parser.add_argument("--workspace", type=str, default="weapon-mpr3p")
    parser.add_argument("--project", type=str, default="ta-rhzr5")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--skip-download", action="store_true", help="Lewati pengecekan / download Roboflow")

    # Hyperparameters Training
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epoch training (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16, gunakan -1 untuk AutoBatch)")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran resolusi gambar input (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device GPU (contoh '0' atau '0,1') atau 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="Jumlah worker DataLoader (default: 4)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    parser.add_argument("--save-period", type=int, default=-1, help="Simpan checkpoint setiap N epoch (-1 hanya best & last)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer: auto, SGD, Adam, AdamW")

    # Output Management
    parser.add_argument("--project-dir", type=str, default="runs/detect", help="Direktori penyimpanan run")
    parser.add_argument("--name", type=str, default=None, help="Nama subfolder eksperimen (default otomatis: waste_<model>)")
    parser.add_argument("--export-onnx", action="store_true", help="Ekspor model terbaik ke format ONNX setelah selesai")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("      MULTI-CLASS WASTE DETECTION TRAINING (YOLOv12)      ")
    print("=" * 65)

    # 1. Cek Ketersediaan Hardware
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[HARDWARE] Menggunakan GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        device = args.device
    else:
        print("[HARDWARE] GPU CUDA tidak terdeteksi. Menggunakan CPU.")
        device = "cpu"

    # 2. Siapkan Dataset
    if args.data_yaml:
        yaml_path = Path(args.data_yaml)
    elif args.skip_download:
        yaml_path = Path(args.dataset_dir) / "data.yaml"
    else:
        yaml_path = ensure_dataset(
            dataset_dir=args.dataset_dir,
            api_key=args.api_key,
            workspace=args.workspace,
            project_name=args.project,
            version_num=args.version,
        )

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di: {yaml_path}")

    # Normalisasi path pada data.yaml
    yaml_path = fix_data_yaml_paths(yaml_path)

    # 3. Siapkan Bobot Pre-trained
    model_weight = download_pretrained_weights(args.model)
    print(f"[INFO] Menginisialisasi model YOLO: {model_weight}")
    model = YOLO(model_weight)

    # 4. Mulai Training
    exp_name = args.name if args.name else f"waste_{Path(model_weight).stem}"
    print("\n" + "=" * 65)
    print(f"[TRAINING] Memulai training:")
    print(f"  - Model: {model_weight}")
    print(f"  - Data: {yaml_path}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch} (AutoBatch jika -1)")
    print(f"  - Image Size: {args.imgsz}")
    print(f"  - Device: {device}")
    print(f"  - Output Dir: {Path(args.project_dir) / exp_name}")
    print("=" * 65 + "\n")

    train_results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        lr0=args.lr0,
        patience=args.patience,
        save_period=args.save_period,
        optimizer=args.optimizer,
        project=args.project_dir,
        name=exp_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("[TRAINING SELESAI]")
    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(args.project_dir) / exp_name
    best_weight = save_dir / "weights" / "best.pt"
    print(f"[OUTPUT] Model terbaik disimpan di: {best_weight.resolve()}")
    print("=" * 65)

    # 5. Evaluasi pada Test Set (Split Test)
    if best_weight.exists():
        print("\n[EVALUASI] Menjalankan validasi akhir pada Test Split...")
        best_model = YOLO(str(best_weight))
        test_metrics = best_model.val(data=str(yaml_path), split="test", project=args.project_dir, name=f"{exp_name}_test_eval")
        print("\n[HASIL METRIK TEST SET]")
        print(f"  - mAP50    : {test_metrics.box.map50:.4f}")
        print(f"  - mAP50-95 : {test_metrics.box.map:.4f}")
        print(f"  - Precision: {test_metrics.box.mp:.4f}")
        print(f"  - Recall   : {test_metrics.box.mr:.4f}")

        # 6. Ekspor Model (Opsional)
        if args.export_onnx:
            print("\n[EXPORT] Mengekspor model ke format ONNX...")
            onnx_path = best_model.export(format="onnx", dynamic=True)
            print(f"[EXPORT] Berhasil diekspor ke: {onnx_path}")

    print("\nSelesai! Anda dapat menggunakan weights/best.pt untuk inferensi atau deployment.")


if __name__ == "__main__":
    main()

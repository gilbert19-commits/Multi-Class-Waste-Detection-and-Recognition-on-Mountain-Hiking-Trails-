"""
Script Inferensi / Deteksi Objek Sampah (Waste Detection) menggunakan Model YOLO
Mendukung input: Gambar tunggal, Folder gambar, Video, atau Webcam (0)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Deteksi Objek Sampah dengan Model YOLO")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/waste_yolov12/weights/best.pt",
        help="Path bobot model (.pt). Jika belum ada, gunakan yolov12n.pt atau yolov12s.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="TA-1/test/images",
        help="Sumber gambar/video/webcam (contoh: path/to/image.jpg, folder/, video.mp4, atau 0 untuk webcam)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolusi gambar input (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device GPU (0) atau 'cpu'")
    parser.add_argument("--project", type=str, default="runs/detect", help="Folder output prediksi")
    parser.add_argument("--name", type=str, default="predict", help="Subfolder output prediksi")
    parser.add_argument("--save-txt", action="store_true", help="Simpan label bounding box ke file .txt")
    parser.add_argument("--show", action="store_true", help="Tampilkan jendela preview langsung")

    return parser.parse_args()


def main():
    args = parse_args()

    # Periksa ketersediaan bobot model
    weight_path = Path(args.weights)
    if not weight_path.exists():
        fallback_candidates = [
            "runs/detect/waste_yolov12/weights/best.pt",
            "yolov12s.pt",
            "yolov12n.pt",
        ]
        found = False
        for candidate in fallback_candidates:
            if Path(candidate).exists():
                print(f"[INFO] Weights '{args.weights}' tidak ditemukan. Menggunakan fallback: '{candidate}'")
                weight_path = Path(candidate)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"File bobot model tidak ditemukan di: {args.weights}")

    print("=" * 65)
    print("           YOLO WASTE DETECTION INFERENCE           ")
    print("=" * 65)
    print(f"Model Weights : {weight_path}")
    print(f"Source Input  : {args.source}")
    print(f"Confidence    : {args.conf}")
    print(f"Device        : {args.device}")
    print("=" * 65)

    model = YOLO(str(weight_path))

    # Jalankan prediksi
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        show=args.show,
        stream=False,
    )

    print("\n" + "=" * 65)
    print(f"[SELESAI] Hasil anotasi bounding box telah disimpan ke:")
    save_dir = results[0].save_dir if results else f"{args.project}/{args.name}"
    print(f"-> {Path(save_dir).resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()

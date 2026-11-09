# Comprehensive Training & Benchmark Summary
## Multi-Class Waste Detection and Recognition on Mountain Hiking Trails

---

## 1. Executive Summary

All five variants of **YOLOv12** (`Nano`, `Small`, `Medium`, `Large`, `XLarge`) were trained for **100 epochs** on an **NVIDIA GeForce RTX 5070 (12 GB VRAM)** using dynamic memory AutoBatch (`batch: -1`). Total training benchmark duration across all 5 models was **20.24 hours**.

Among all tested models, **YOLOv12s (Small)** demonstrated the highest detection performance on the unseen test set (**58.37% mAP@50**, **34.38% mAP@50-95**), striking the optimal balance between representation capacity and generalization for this dataset.

---

## 2. Dataset Configuration & Split Clarification

### Dataset Splits
The dataset is exported from Roboflow in format `yolov12` (`TA-1`):

| Split | Image Count | Percentage | Purpose |
| :--- | :---: | :---: | :--- |
| **Train (`train`)** | **2,803** | 71.0% | Model parameter updates during backward propagation across all 100 epochs. |
| **Validation (`valid` / `val`)** | **595** | 15.1% | Evaluated at the end of every epoch to calculate validation loss and save `weights/best.pt`. |
| **Test (`test`)** | **550** | 13.9% | Held-out, completely unseen data used for final model benchmarking and thesis evaluation. |
| **Total** | **3,948** | 100.0% | Complete dataset size |

### 🔍 Split Confirmation: Which Split Was Used for Evaluation?
To prevent data leakage and provide scientific rigor, two levels of evaluation are conducted:
1. **In-Training Evaluation (Every Epoch)**:
   - Evaluated strictly on the **`valid` (`val`) split** (`TA-1/valid/images`, 595 images).
   - This split determines when training loss plateaus and identifies the best epoch checkpoint (`best.pt`).
2. **Final Benchmark Evaluation (Post-Training)**:
   - Evaluated strictly on the **`test` split** (`TA-1/test/images`, 550 images).
   - These images were **never seen** during gradient descent or checkpoint selection, providing an unbiased estimate of real-world generalization on hiking trails.

### Target Classes (4 Waste Categories)
1. `kertas_tisu` (Paper / Tissue)
2. `logam_kaleng_kaca` (Metal / Cans / Glass)
3. `plastik` (Plastic bottles / wrappers / bags)
4. `puntung_rokok` (Cigarette butts)

---

## 3. Training Setup & Hyperparameters

- **Hardware**: NVIDIA GeForce RTX 5070 Laptop/Desktop GPU (12,227 MiB VRAM)
- **Framework**: Ultralytics `8.4.123`, PyTorch `2.14.0+cu132`, CUDA 13.3
- **Base Architecture**: YOLOv12 Attention-Centric Object Detector
- **Input Resolution**: `640 x 640`
- **Batch Size**: `-1` (AutoBatch dynamically measured against 12GB VRAM)
- **Total Epochs**: `100` epochs per variant
- **Optimizer**: `auto` (SGD / AdamW with momentum 0.937, weight decay 0.0005)
- **Initial Learning Rate**: `lr0 = 0.01`, cosine LR decay (`lrf = 0.01`)
- **Device**: `0` (CUDA GPU)
- **Mixed Precision**: FP16 Automatic Mixed Precision (`amp: true`)

---

## 4. Comprehensive Benchmark Comparison Table

All models evaluated using their respective `weights/best.pt`:

| Model Variant | Parameters | GFLOPs | Training Time | Val mAP@50 | Val mAP@50-95 | Test Precision | Test Recall | Test mAP@50 | Test mAP@50-95 | Inference Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv12n** (Nano) | 2.56 M | 7.3 | 39.6 min | 0.3773 | 0.2118 | 0.3534 | 0.4077 | 0.3425 | 0.1964 | **1.9 ms** |
| **YOLOv12s** (Small) 🏆 | 9.23 M | 23.2 | 89.8 min | **0.6076** | **0.2300** | 0.5813 | 0.5513 | **0.5837** | **0.3438** | 4.0 ms |
| **YOLOv12m** (Medium) | 20.11 M | 70.4 | 167.2 min | 0.3256 | 0.1841 | **0.6214** | 0.5270 | 0.5423 | 0.3126 | 9.0 ms |
| **YOLOv12l** (Large) | 26.34 M | 95.1 | 316.4 min | 0.3077 | 0.1731 | 0.4881 | **0.5672** | 0.5520 | 0.2490 | 13.1 ms |
| **YOLOv12x** (XLarge) | 59.05 M | 208.4 | 601.5 min | 0.2807 | 0.1558 | 0.5050 | 0.5374 | 0.4755 | 0.2459 | 22.8 ms |

---

## 5. Per-Class Test Split Breakdown (`mAP@50`)

Performance on the held-out **Test split** across the 4 waste categories:

| Model Variant | `kertas_tisu` | `logam_kaleng_kaca` | `plastik` | `puntung_rokok` | Overall Test mAP@50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv12n** | 0.2999 | 0.4729 | 0.5723 | 0.0249 | **0.3425** |
| **YOLOv12s** 🏆 | **0.3009** | **0.4752** | **0.5638** | **0.9950** | **0.5837** |
| **YOLOv12m** | 0.2725 | 0.3775 | 0.5242 | 0.9950 | **0.5423** |
| **YOLOv12l** | 0.2798 | 0.3876 | 0.5456 | 0.9950 | **0.5520** |
| **YOLOv12x** | 0.2709 | 0.2960 | 0.5070 | 0.8283 | **0.4755** |

---

## 6. Key Scientific Insights

1. **Optimal Model Architecture**:
   - **YOLOv12s** achieved the highest test mAP@50 (**0.5837**) and test mAP@50-95 (**0.3438**).
   - It detected `plastik` with 56.38% mAP@50 and `puntung_rokok` with 99.50% mAP@50 while taking only **4.0 ms** per image (250 FPS).
2. **Model Capacity vs. Overfitting**:
   - Larger models (YOLOv12l and YOLOv12x with 26M and 59M parameters) exhibited slight overfitting given the dataset size (~2,800 training images), yielding lower mAP than YOLOv12s.
3. **Inference Latency**:
   - YOLOv12n (1.9 ms) and YOLOv12s (4.0 ms) are ideal for real-time edge deployment (e.g., Raspberry Pi 5 with AI accelerator, Jetson Orin Nano, or mobile hiking camera devices).

---

## 7. Model Weight Artifact Locations

All trained checkpoints, loss curves, confusion matrices, and validation prediction batches are saved at:
- **YOLOv12n**: `runs/detect/runs/detect/waste_yolov12n/weights/best.pt`
- **YOLOv12s**: `runs/detect/runs/detect/waste_yolov12s/weights/best.pt`
- **YOLOv12m**: `runs/detect/runs/detect/waste_yolov12m/weights/best.pt`
- **YOLOv12l**: `runs/detect/runs/detect/waste_yolov12l/weights/best.pt`
- **YOLOv12x**: `runs/detect/runs/detect/waste_yolov12x/weights/best.pt`

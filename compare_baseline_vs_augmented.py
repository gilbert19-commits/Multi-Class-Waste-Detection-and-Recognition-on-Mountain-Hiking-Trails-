"""
Script untuk Membandingkan Hasil Training:
Baseline (Sebelum Augmentasi) vs Augmented (Setelah Augmentasi)
Menghasilkan tabel komparasi siap pakai untuk Bab 4 Skripsi/Tugas Akhir.
"""

from pathlib import Path
import pandas as pd


def compare_results():
    baseline_csv = Path("runs/detect/complete_evaluation_summary.csv")
    augmented_csv_candidates = [
        Path("runs/detect/runs/augmented/augmented_benchmark_summary.csv"),
        Path("runs/augmented/benchmark_summary.csv"),
    ]
    augmented_csv = None
    for cand in augmented_csv_candidates:
        if cand.exists():
            augmented_csv = cand
            break

    if not baseline_csv.exists():
        print(f"[ERROR] File baseline tidak ditemukan di: {baseline_csv}")
        return

    if augmented_csv is None:
        print(f"[INFO] File hasil augmentasi belum ditemukan.")
        return

    df_base = pd.read_csv(baseline_csv)
    df_aug = pd.read_csv(augmented_csv)

    # Standardisasi kolom
    # df_base columns: Model, Val_Precision, Val_Recall, Val_mAP50, Val_mAP50-95, Test_Precision, Test_Recall, Test_mAP50, Test_mAP50-95
    # df_aug columns: Model Variant, Weights, mAP@50, mAP@50-95, Precision, Recall, Training Time (min)
    
    records = []
    for _, row_aug in df_aug.iterrows():
        model_name = row_aug.get("Model Variant")
        
        # Cari di baseline
        base_match = df_base[df_base["Model"] == model_name]
        if base_match.empty:
            continue
        row_base = base_match.iloc[0]

        base_map50 = float(row_base.get("Test_mAP50", 0))
        aug_map50 = float(row_aug.get("mAP@50", 0))
        delta_map50 = aug_map50 - base_map50

        base_map = float(row_base.get("Test_mAP50-95", 0))
        aug_map = float(row_aug.get("mAP@50-95", 0))
        delta_map = aug_map - base_map

        records.append({
            "Model Variant": model_name,
            "Baseline mAP@50": f"{base_map50*100:.2f}%",
            "Augmented mAP@50": f"{aug_map50*100:.2f}%",
            "Delta mAP@50": f"{'+' if delta_map50 >= 0 else ''}{delta_map50*100:.2f}%",
            "Baseline mAP@50-95": f"{base_map*100:.2f}%",
            "Augmented mAP@50-95": f"{aug_map*100:.2f}%",
            "Delta mAP@50-95": f"{'+' if delta_map >= 0 else ''}{delta_map*100:.2f}%",
        })

    df_comp = pd.DataFrame(records)

    print("\n" + "=" * 80)
    print("      KOMPARASI MODEL: SEBELUM AUGMENTASI (BASELINE) VS SETELAH AUGMENTASI      ")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print("=" * 80)

    out_md = Path("runs/comparison_baseline_vs_augmented.md")
    out_csv = Path("runs/comparison_baseline_vs_augmented.csv")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_markdown(out_md, index=False)
    df_comp.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nTabel komparasi tersimpan di:")
    print(f"  - Markdown: {out_md.resolve()}")
    print(f"  - CSV     : {out_csv.resolve()}")


if __name__ == "__main__":
    compare_results()

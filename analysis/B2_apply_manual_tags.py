
"""
TASK B2 — APPLY MANUAL TAGS
Script untuk membaca CSV hasil export tool manual (manual_source_tags.csv)
dan menghasilkan mapping terstruktur analysis/source_mapping.json.
Melakukan validasi:
- Memastikan seluruh citra sudah tertag (tidak ada 'Belum_Ditag')
- Melaporkan daftar citra bertanda 'Ragu' secara terpisah
- Menghitung proporsi akhir per sumber (UAVVaste, TACO, Burangrang, Ragu)
"""

import sys
import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT_DIR / "analysis"
CSV_EXPORT_PATH = ANALYSIS_DIR / "tagger" / "manual_source_tags.csv"
OUTPUT_JSON_PATH = ANALYSIS_DIR / "source_mapping.json"


def apply_tags(csv_file=CSV_EXPORT_PATH):
    print("=" * 80)
    print("           TASK B2 — PEMROSESAN HASIL MANUAL TAGGING SUMBER CITRA           ")
    print("=" * 80)

    csv_path = Path(csv_file)
    if not csv_path.exists():
        # Coba cari di root direktori jika pengguna menyimpannya di luar
        fallback = ROOT_DIR / "manual_source_tags.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            print(f"[ERROR] File CSV manual tags belum ditemukan di: {csv_path}")
            print(f"Silakan selesaikan tagging via index.html dan klik 'Export manual_source_tags.csv',")
            print(f"lalu simpan file tersebut ke: {CSV_EXPORT_PATH.resolve()}")
            return False

    df = pd.read_csv(csv_path)
    print(f"\nBerhasil membaca {len(df)} baris dari {csv_path.name}")

    if "filename" not in df.columns or "source" not in df.columns:
        print("[ERROR] Kolom CSV harus memiliki 'filename' dan 'source'.")
        return False

    # 1. Validasi apakah ada yang belum ditag
    unlabeled = df[df["source"].isin(["Belum_Ditag", "", None])]
    if not unlabeled.empty:
        print(f"\n[PERINGATAN] Masih ada {len(unlabeled)} citra yang BELUM DITAG:")
        for idx, row in unlabeled.head(10).iterrows():
            print(f"  - {row['filename']}")
        print("Harap selesaikan seluruh penandaan sebelum mengekspor.")
    else:
        print("\n[VALIDASI SUKSES] Seluruh citra telah berhasil ditandai!")

    # 2. Rekapitulasi per sumber
    counts = df["source"].value_counts()
    total = len(df)
    
    print("\n--- RINGKASAN SUMBER CITRA TEST SET ---")
    summary_data = []
    for src, cnt in counts.items():
        pct = (cnt / total) * 100
        print(f"  - {src:<15}: {cnt:>4} citra ({pct:>6.2f}%)")
        summary_data.append({
            "Source": src,
            "Count": cnt,
            "Percentage": f"{pct:.2f}%"
        })

    # 3. Pisahkan daftar citra bertanda Ragu
    ragu_list = df[df["source"].str.lower() == "ragu"]["filename"].tolist()
    print(f"\nCitra bertanda 'Ragu': {len(ragu_list)} citra (akan dieksklusi dari domain split Burangrang vs Public)")

    # 4. Simpan ke analysis/source_mapping.json
    mapping_dict = {
        "metadata": {
            "total_images": total,
            "completed": len(unlabeled) == 0,
            "counts": counts.to_dict(),
            "ragu_count": len(ragu_list),
        },
        "ragu_images": ragu_list,
        "mapping": dict(zip(df["filename"], df["source"])),
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=2)

    print(f"\n[OK] Mapping sumber berhasil disimpan di: {OUTPUT_JSON_PATH.resolve()}")
    return True


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else CSV_EXPORT_PATH
    apply_tags(path_arg)

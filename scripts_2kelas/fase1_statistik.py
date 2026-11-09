"""
FASE 1 — STATISTIK DATASET 2 KELAS (dataset-2kelas/, read-only)

- nc, names, jumlah citra per split, citra background
- instance per kelas per split (baris poligon dikonversi ke bbox min/max seperti Ultralytics)
- ukuran citra (dicek dari file, bukan diasumsikan)
- distribusi ukuran objek dalam piksel citra: bin luas COCO kecil <32², sedang 32²–96², besar >96²
  (sama dengan SIZE_BINS di scripts/02_error_analysis.py), plus objek dengan lebar DAN tinggi < 32 px
- file label yang mencampur poligon dan box per split (definisi sama dengan scan_mixed_labels lama:
  ada baris 5 kolom DAN ada baris > 6 kolom)

Output: analysis_2kelas/fase1_statistik.md + analysis_2kelas/tables/*.csv
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset-2kelas"
OUT = ROOT / "analysis_2kelas"
TABLES = OUT / "tables"
SPLITS = ["train", "valid", "test"]
SIZE_BINS = [("kecil (<32²)", 0, 32 ** 2), ("sedang (32²–96²)", 32 ** 2, 96 ** 2), ("besar (>96²)", 96 ** 2, np.inf)]


def parse(path, w, h):
    """Return rows [(cls, bw_px, bh_px, kind)], n_box, n_seg, masalah."""
    rows, n_box, n_seg, issues = [], 0, 0, []
    for ln in path.read_text(encoding="utf-8").splitlines():
        x = ln.split()
        if not x:
            continue
        v = np.array(x[1:], dtype=float)
        if len(x) == 5:
            n_box += 1
            bw, bh = v[2], v[3]
            kind = "box"
        elif len(x) > 6:
            n_seg += 1
            pts = v[: len(v) // 2 * 2].reshape(-1, 2)
            bw, bh = pts.max(0) - pts.min(0)
            kind = "poligon"
        else:
            issues.append(f"{path.name}: baris {len(x)} kolom")
            continue
        if (v < 0).any() or (v > 1.0001).any():
            issues.append(f"{path.name}: koordinat di luar [0,1]")
        rows.append((int(x[0]), bw * w, bh * h, kind))
    return rows, n_box, n_seg, issues


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    TABLES.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((DATA_DIR / "data.yaml").read_text())
    names = cfg["names"]
    nc = cfg["nc"]

    objs, files, issues, sizes = [], [], [], Counter()
    for s in SPLITS:
        imgs = sorted((DATA_DIR / s / "images").iterdir())
        lbls = {p.stem for p in (DATA_DIR / s / "labels").glob("*.txt")}
        for img in imgs:
            with Image.open(img) as im:
                w, h = im.size
            sizes[(s, w, h)] += 1
            lp = DATA_DIR / s / "labels" / (img.stem + ".txt")
            if not lp.exists():
                issues.append(f"{s}/{img.name}: file label tidak ada")
                rows, nb, ns, iss = [], 0, 0, []
            else:
                rows, nb, ns, iss = parse(lp, w, h)
            issues += [f"{s}/{m}" for m in iss]
            files.append(dict(split=s, file=img.name, n_box=nb, n_seg=ns, mixed=nb > 0 and ns > 0,
                              hanya_poligon=ns > 0 and nb == 0, background=len(rows) == 0))
            for c, bw, bh, kind in rows:
                if not 0 <= c < nc:
                    issues.append(f"{s}/{lp.name}: class id {c} di luar 0..{nc - 1}")
                    continue
                objs.append(dict(split=s, file=img.name, kelas=names[c], w_px=bw, h_px=bh,
                                 area_px=bw * bh, jenis=kind))
        orphan = lbls - {p.stem for p in imgs}
        issues += [f"{s}/labels/{o}.txt: tanpa citra" for o in sorted(orphan)]

    O = pd.DataFrame(objs)
    F = pd.DataFrame(files)
    O["bin_ukuran"] = pd.cut(O["area_px"], [b[1] for b in SIZE_BINS] + [np.inf], right=False,
                             labels=[b[0] for b in SIZE_BINS])
    O["wh_lt32"] = (O["w_px"] < 32) & (O["h_px"] < 32)
    O.to_csv(TABLES / "fase1_objek.csv", index=False)
    F.to_csv(TABLES / "fase1_file_label.csv", index=False)

    # ---- tabel
    t_img = F.groupby("split", sort=False).agg(
        citra=("file", "size"), background=("background", "sum"),
        file_campuran_poligon_box=("mixed", "sum"), file_hanya_poligon=("hanya_poligon", "sum"),
        baris_box=("n_box", "sum"), baris_poligon=("n_seg", "sum")).reindex(SPLITS)
    t_img.loc["total"] = t_img.sum()
    t_inst = O.pivot_table(index="split", columns="kelas", values="file", aggfunc="size", fill_value=0
                           ).reindex(index=SPLITS, columns=names, fill_value=0)
    t_inst["total"] = t_inst.sum(1)
    t_inst.loc["total"] = t_inst.sum()
    t_size = O.pivot_table(index=["split", "kelas"], columns="bin_ukuran", values="file", aggfunc="size",
                           fill_value=0, observed=False)
    t_size["total"] = t_size.sum(1)
    t_size["% kecil (<32²)"] = (t_size[SIZE_BINS[0][0]] / t_size["total"] * 100)
    lt32 = O.groupby(["split", "kelas"])["wh_lt32"].mean() * 100
    t_size["% lebar&tinggi <32px"] = lt32
    t_size = t_size.reindex(pd.MultiIndex.from_product([SPLITS, names], names=["split", "kelas"]))

    def agg_rows(df, key):
        g = df.groupby(key)
        return pd.DataFrame({"objek": g.size(), "% kecil (<32²)": g["area_px"].apply(lambda a: (a < 1024).mean() * 100),
                             "% lebar&tinggi <32px": g["wh_lt32"].mean() * 100,
                             "median luas (px²)": g["area_px"].median()})

    t_size_split = agg_rows(O, "split").reindex(SPLITS)
    t_size_split.loc["total"] = [len(O), (O["area_px"] < 1024).mean() * 100, O["wh_lt32"].mean() * 100,
                                 O["area_px"].median()]
    t_img.to_csv(TABLES / "fase1_citra_per_split.csv")
    t_inst.to_csv(TABLES / "fase1_instance_per_kelas.csv")
    t_size.to_csv(TABLES / "fase1_ukuran_objek.csv")
    t_size_split.to_csv(TABLES / "fase1_ukuran_objek_per_split.csv")

    md = ["# Fase 1 — Statistik dataset-2kelas", "",
          f"- data.yaml: `nc: {nc}`, `names: {names}` (indeks 0 = {names[0]}, 1 = {names[1]})",
          "- Ukuran citra (dibaca dari file): " + ", ".join(f"{s} {w}×{h}: {n}" for (s, w, h), n in sorted(sizes.items())),
          "", "## Citra per split", "", t_img.to_markdown(), "",
          "## Instance per kelas per split", "", t_inst.to_markdown(), "",
          "## Ukuran objek (piksel citra, bbox dari poligon bila poligon)", "",
          "Bin luas COCO: kecil < 32² px, sedang 32²–96², besar > 96².", "",
          t_size_split.to_markdown(floatfmt=".4f"), "", t_size.to_markdown(floatfmt=".4f"), "",
          "## Masalah label", "", f"Jumlah: {len(issues)}", ""]
    md += [f"- {m}" for m in issues[:50]]
    if len(issues) > 50:
        md.append(f"- ... ({len(issues) - 50} lagi, lihat fase1_masalah_label.txt)")
    (OUT / "fase1_masalah_label.txt").write_text("\n".join(issues), encoding="utf-8")
    (OUT / "fase1_statistik.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

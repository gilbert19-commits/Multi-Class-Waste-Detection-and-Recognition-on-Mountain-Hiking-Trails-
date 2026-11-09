"""BAGIAN B4 (lanjutan): mencocokkan nama file dataset dengan daftar nama file resmi TACO dan UAVVaste.

Aturan lama (hk_common.source_of) tidak diubah. Hasilnya dibandingkan dengan aturan usulan (up_common.source_bukti), lalu recall
model terbaik tiap family dihitung ulang dengan kedua aturan. Yang dicocokkan hanya nama file, bukan isi citra.
"""
import json
import re
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

from up_common import CACHE, DATA_DIR, HK, NAMES, OUT, PAPER, base_name, best_std, label, load_preds, source_bukti, source_of

sys.path.insert(0, str(HK / "scripts"))
from s5_error import match_image  # noqa: E402

URLS = {"TACO": "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json",
        "UAVVaste": "https://raw.githubusercontent.com/UAVVaste/UAVVaste/main/annotations/annotations.json"}
SPLITS = ("train", "valid", "test")


def official_names():
    d = CACHE / "anotasi"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, url in URLS.items():
        f = d / f"{k}.json"
        if not f.exists():
            urllib.request.urlretrieve(url, f)
        names = [x["file_name"] for x in json.loads(f.read_text(encoding="utf-8"))["images"]]
        out[k] = {re.sub(r"\.[^.]+$", "", n.split("/")[-1]).lower() for n in names}
        print(k, "jumlah citra di anotasi:", len(names), "| nama unik:", len(out[k]))
    return out


def in_list(base, names):
    q = base.lower()
    return q in names or re.sub(r"_jpg$", "", q) in names


def object_counts(sp, stem):
    rows = [r.split() for r in (DATA_DIR / sp / "labels" / f"{stem}.txt").read_text().splitlines() if r.strip()]
    c = [int(r[0]) for r in rows]
    return c.count(0), c.count(1)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)
    dest = OUT / "sumber"
    official = official_names()
    stats = pd.read_csv(HK / "dataset_stats.csv")
    thr = int(stats[stats.item == "ambang_inlier_ORB_maks_kontrol_acak"].nilai.iloc[0])
    P = pd.read_csv(HK / "tumpang_tindih_train.csv")
    leak = set(P[(P.split == "test") & (P.inlier_nama_sama > thr)].file)

    pat = defaultdict(lambda: defaultdict(int))
    per_src = defaultdict(lambda: defaultdict(int))
    for sp in SPLITS:
        for p in (DATA_DIR / sp / "images").iterdir():
            b = base_name(p.name)
            k = re.sub(r"\d+", "#", b)
            lw, mg = object_counts(sp, p.stem)
            r = pat[k]
            r[sp] += 1
            r["cocok_TACO"] += in_list(b, official["TACO"])
            r["cocok_UAVVaste"] += in_list(b, official["UAVVaste"])
            r[f"{sp}_lightweight"] += lw
            r[f"{sp}_metal"] += mg
            r["test_tumpang_tindih"] += p.name in leak
            r["lama"], r["usulan"] = source_of(p.name), source_bukti(p.name)
            for rule, src in (("aturan_lama", source_of(p.name)), ("aturan_usulan", source_bukti(p.name))):
                q = per_src[(rule, src, sp)]
                q["citra"] += 1
                q["objek_lightweight_litter"] += lw
                q["objek_metal_and_glass"] += mg
                q["citra_tanpa_objek"] += not (lw + mg)
    rows = []
    for k, r in pat.items():
        total = r["train"] + r["valid"] + r["test"]
        rows.append(dict(pola_nama=k, sumber_aturan_lama=r["lama"], sumber_aturan_usulan=r["usulan"], train=r["train"], valid=r["valid"], test=r["test"], total=total,
                         cocok_nama_TACO=r["cocok_TACO"], cocok_nama_UAVVaste=r["cocok_UAVVaste"], tidak_cocok_keduanya=total - r["cocok_TACO"] - r["cocok_UAVVaste"],
                         objek_lightweight_train=r["train_lightweight"], objek_metal_train=r["train_metal"], objek_lightweight_valid=r["valid_lightweight"], objek_metal_valid=r["valid_metal"],
                         objek_lightweight_test=r["test_lightweight"], objek_metal_test=r["test_metal"], test_tumpang_tindih_daftar_hasil_2kelas=r["test_tumpang_tindih"]))
    M = pd.DataFrame(rows).sort_values("total", ascending=False)
    M.to_csv(dest / "bukti_nama_TACO_UAVVaste.csv", index=False)
    S = pd.DataFrame([dict(aturan=a, sumber=s, split=sp, **v) for (a, s, sp), v in per_src.items()]).sort_values(["aturan", "sumber", "split"])
    S.to_csv(dest / "jumlah_per_sumber_dua_aturan.csv", index=False)
    print(M.iloc[:, :10].to_string(index=False))
    print(M[["pola_nama", "objek_lightweight_train", "objek_metal_train", "objek_lightweight_valid", "objek_metal_valid", "objek_lightweight_test", "objek_metal_test", "test_tumpang_tindih_daftar_hasil_2kelas"]].to_string(index=False))
    print(S.to_string(index=False))

    # recall model terbaik tiap family (conf 0.25, IoU 0.5), dengan fungsi yang sama seperti recall_breakdown.csv
    best = best_std()
    out = []
    for f, m in best.items():
        recs = []
        for fname, x in load_preds(m).items():
            gs, _, _ = match_image(x)
            for c, g in zip(x["gt_cls"], gs):
                recs.append(dict(file=fname, lama=source_of(fname), usulan=source_bukti(fname), pola=re.sub(r"\d+", "#", base_name(fname)), kelas=PAPER[NAMES[int(c)]], state=int(g)))
        R = pd.DataFrame(recs)
        for dim, col in (("aturan_lama", "lama"), ("aturan_usulan", "usulan"), ("pola_nama", "pola")):
            for k, g in R.groupby(col):
                for kelas, gg in [("semua", g)] + list(g.groupby("kelas")):
                    out.append(dict(model=label(m), family=f, dimensi=dim, kategori=k, kelas=kelas, n_objek=len(gg), n_gambar_berobjek=gg.file.nunique(), TP=int((gg.state == 1).sum()),
                                    FN_salah_kelas=int((gg.state == 2).sum()), FN_terlewat=int((gg.state == 0).sum()), recall=float((gg.state == 1).mean()), conf=0.25, iou=0.5))
    B = pd.DataFrame(out)
    B.to_csv(dest / "recall_per_sumber_dua_aturan.csv", index=False)
    print(B[(B.kelas == "semua") & (B.dimensi != "pola_nama")].round(3).drop(columns=["family", "conf", "iou", "kelas"]).to_string(index=False))
    print(B[(B.kelas == "semua") & (B.dimensi == "pola_nama")].pivot(index="kategori", columns="model", values="recall").round(3).to_string())
    print(np.__version__, "[SELESAI b3]")


if __name__ == "__main__":
    main()

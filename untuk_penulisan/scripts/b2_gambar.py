"""BAGIAN B4-B5: pola nama file, aturan sumber, contact sheet IMG_, dan contoh dataset untuk gambar paper.

Citra contoh diambil dari split test (labelnya dari salinan hasil_2kelas/_data, sehingga label campuran sudah jadi box). Sumbernya memakai
aturan usulan (source_bukti) karena aturan lama terbukti salah untuk IMG_#, GOPR, dan Unknown. Kotak digambar pada piksel asli, tanpa resize.
"""
import re
import shutil
import sys
import textwrap
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from up_common import DATA_DIR, MIRROR, NAMES, OUT, PAPER, SEED, base_name, source_bukti, source_of  # noqa: E402

SPLITS = ("train", "valid", "test")
SOURCES = ("Burangrang", "UAVVaste", "TACO")
N_SHEET, N_FULL, N_CROP = 24, 4, 8
COLOR = {"lightweight litter": "#1baf7a", "metal and glass": "#eda100"}
MIN_SIDE = 12  # piksel, supaya crop terkecil masih bisa dikenali
MARGIN = 0.10
RULE = """Aturan penentuan sumber. Keduanya hanya berdasarkan nama file setelah akhiran Roboflow (_jpg.rf.<hash> atau _JPG.rf.<hash>) dibuang.

ATURAN LAMA, dipakai di seluruh hasil_2kelas (hk_common.source_of, dari scripts/fase1_scene_groups.py)
1. Awalan batch_ atau BATCH_                 -> UAVVaste
2. Nama seluruhnya angka (contoh 000015)     -> TACO
3. Awalan WhatsApp-Video, IMG_, atau GOPR    -> Burangrang
4. Selain itu (camera_img_, photo_, DJI_)    -> Unknown

ATURAN USULAN, menunggu konfirmasi (up_common.source_bukti; bukti di bukti_nama_TACO_UAVVaste.csv)
1. batch_, BATCH_, GOPR, camera_img_, photo_, DJI_    -> UAVVaste  (semua nama ada di anotasi UAVVaste)
2. Nama seluruhnya angka, file IMG_####_JPG.rf.*      -> TACO      (semua nama ada di anotasi TACO; IMG_ ada di batch_3)
3. WhatsApp-Video, file IMG_####_JPG_jpg.rf.*         -> Burangrang (tidak ada di kedua anotasi; asal ini perlu dipastikan pemilik data)
"""


def b4_pattern():
    dest = OUT / "sumber"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "aturan_sumber.txt").write_text(RULE, encoding="utf-8")
    cnt, bases, example = defaultdict(lambda: dict(train=0, valid=0, test=0)), defaultdict(lambda: defaultdict(set)), {}
    all_bases = defaultdict(dict)
    for sp in SPLITS:
        for p in sorted((DATA_DIR / sp / "images").iterdir()):
            b = base_name(p.name)
            k = re.sub(r"\d+", "#", b)
            cnt[k][sp] += 1
            bases[k][sp].add(b)
            example.setdefault(k, p.name)
            all_bases[b].setdefault(sp, p.name)
    rows = []
    for k, c in cnt.items():
        src = source_of(example[k])
        rows.append(dict(pola_nama=k, sumber_menurut_aturan=src, train=c["train"], valid=c["valid"], test=c["test"], total=sum(c.values()),
                         nama_dasar_unik_train=len(bases[k]["train"]), nama_dasar_unik_valid=len(bases[k]["valid"]), nama_dasar_unik_test=len(bases[k]["test"]), contoh_file=example[k]))
    P = pd.DataFrame(rows).sort_values(["sumber_menurut_aturan", "total"], ascending=[True, False])
    tot = P[["train", "valid", "test"]].sum()
    if tot.to_dict() != dict(train=8409, valid=595, test=550):
        raise RuntimeError(f"total pola tidak sama dengan jumlah citra: {tot.to_dict()}")
    P.to_csv(dest / "pola_nama_file.csv", index=False)
    print(P.drop(columns="contoh_file").to_string(index=False))

    # Contact sheet: 24 nama dasar acak (seed 0), supaya tidak habis untuk augmentasi dari foto yang sama. Prioritas file: test, valid, train.
    img = {b: v for b, v in all_bases.items() if b.startswith("IMG_")}
    contact_sheet(img, dest / "contact_sheet_IMG")
    contact_sheet({b: v for b, v in img.items() if not b.endswith("_JPG")}, dest / "contact_sheet_IMG_pola_tanpa_akhiran_JPG")
    contact_sheet({b: v for b, v in all_bases.items() if b.startswith("GOPR")}, dest / "contact_sheet_GOPR")
    print(f"IMG_: {len(img)} nama dasar unik, {sum(len(v) for v in img.values())} kombinasi nama-split")


def contact_sheet(by_base, path):
    names = sorted(by_base)
    pick = np.random.default_rng(SEED).choice(len(names), N_SHEET, replace=False)
    fig, axes = plt.subplots(4, 6, figsize=(15, 12.4))
    sheet = []
    for ax, i in zip(axes.flat, sorted(pick)):
        b = names[i]
        sp = next(s for s in ("test", "valid", "train") if s in by_base[b])
        fname = by_base[b][sp]
        ax.imshow(Image.open(DATA_DIR / sp / "images" / fname).convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"[{sp}]\n" + "\n".join(textwrap.wrap(fname, 34)), fontsize=7, family="monospace")
        sheet.append(dict(nama_dasar=b, split_ditampilkan=sp, file=fname, ada_di_split=";".join(by_base[b])))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.04, wspace=0.05, hspace=0.32)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)
    pd.DataFrame(sheet).to_csv(path.with_suffix(".csv"), index=False)


def read_boxes(stem):
    """[(kelas_paper, x1, y1, x2, y2 ternormalisasi)] dari label salinan; poligon jadi kotak pembungkus."""
    out = []
    for r in (MIRROR / "test" / "labels" / f"{stem}.txt").read_text().splitlines():
        p = r.split()
        if not p:
            continue
        v = np.array(p[1:], float)
        if len(v) == 4:
            x1, y1, x2, y2 = v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2
        else:
            xy = v.reshape(-1, 2)
            x1, y1, x2, y2 = xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()
        out.append((PAPER[NAMES[int(p[0])]], x1, y1, x2, y2))
    return out


def bgr(hexcolor):
    h = hexcolor.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def b5_examples():
    dest = OUT / "contoh_dataset"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "utuh").mkdir(parents=True)
    (dest / "crop").mkdir()
    pool = defaultdict(list)
    for p in sorted((DATA_DIR / "test" / "images").iterdir()):
        src = source_bukti(p.name)
        boxes = read_boxes(p.stem)
        if boxes:
            pool[src].append((p, boxes))
    rng = np.random.default_rng(SEED)
    rows = []
    for src in SOURCES:
        order = [pool[src][i] for i in rng.permutation(len(pool[src]))]
        ok = [x for x in order if len(x[1]) <= 10]
        metal = [x for x in ok if any(b[0] == "metal and glass" for b in x[1])]
        video = [x for x in ok if x[0].name.startswith("WhatsApp-Video")][:1]  # Burangrang: dua pola nama sama-sama terwakili
        chosen, seen = [], set()
        for x in metal[:2] + video + ok:  # sampai 2 citra berisi metal and glass, sisanya acak
            if base_name(x[0].name) not in seen and len(chosen) < N_FULL:
                chosen.append(x)
                seen.add(base_name(x[0].name))
        for i, (p, boxes) in enumerate(chosen, 1):
            im = cv2.imread(str(p))
            h, w = im.shape[:2]
            for cname, x1, y1, x2, y2 in boxes:
                cv2.rectangle(im, (round(x1 * w), round(y1 * h)), (round(x2 * w), round(y2 * h)), bgr(COLOR[cname]), 2, cv2.LINE_AA)
            name = f"{src}_{i}.png"
            cv2.imwrite(str(dest / "utuh" / name), im)
            rows.append(dict(jenis="utuh", file_png=f"utuh/{name}", sumber=src, kelas=";".join(sorted({b[0] for b in boxes})), lebar=w, tinggi=h, n_objek=len(boxes), file_asal=p.name))
    objs = {c: [] for c in COLOR}
    for src in SOURCES:
        for p, boxes in pool[src]:
            with Image.open(p) as im:
                w, h = im.size
            for cname, x1, y1, x2, y2 in boxes:
                bw, bh = (x2 - x1) * w, (y2 - y1) * h
                if min(bw, bh) >= MIN_SIDE:
                    objs[cname].append(dict(file=p, sumber=src, x1=x1 * w, y1=y1 * h, x2=x2 * w, y2=y2 * h, W=w, H=h, area=bw * bh))
    for cname, lst in objs.items():
        O = pd.DataFrame(lst).sort_values("area").reset_index(drop=True)
        used, targets = set(), np.linspace(np.log(O.area.min()), np.log(O.area.max()), N_CROP)  # rentang penuh ukuran, jarak rata skala log
        for i, t in enumerate(targets, 1):
            jarak = (np.log(O.area) - t).abs()
            dekat = O[(O.sumber == "Burangrang") & (jarak < 0.35)].index  # utamakan data sendiri kalau ukurannya berdekatan
            for j in list(jarak[dekat].sort_values().index) + list(jarak.sort_values().index):
                o = O.loc[j]
                if o.file.name not in used:  # satu crop per citra
                    used.add(o.file.name)
                    break
            mx, my = (o.x2 - o.x1) * MARGIN, (o.y2 - o.y1) * MARGIN
            x1, y1 = max(0, int(round(o.x1 - mx))), max(0, int(round(o.y1 - my)))
            x2, y2 = min(o.W, int(round(o.x2 + mx))), min(o.H, int(round(o.y2 + my)))
            crop = cv2.imread(str(o.file))[y1:y2, x1:x2]
            name = f"{cname.replace(' ', '_')}_{i}_{x2 - x1}x{y2 - y1}.png"
            cv2.imwrite(str(dest / "crop" / name), crop)
            rows.append(dict(jenis="crop", file_png=f"crop/{name}", sumber=o.sumber, kelas=cname, lebar=x2 - x1, tinggi=y2 - y1, n_objek=1, file_asal=o.file.name))
        print(f"{cname}: {len(O)} objek kandidat dengan sisi >= {MIN_SIDE} px")
    pd.DataFrame(rows).to_csv(dest / "daftar_contoh.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="file_asal").to_string(index=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    b4_pattern()
    b5_examples()
    print("[SELESAI b2]")


if __name__ == "__main__":
    main()

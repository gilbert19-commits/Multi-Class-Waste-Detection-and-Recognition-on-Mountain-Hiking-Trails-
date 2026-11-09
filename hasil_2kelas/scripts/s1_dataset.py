"""LANGKAH 1: statistik dataset dari file label. Tidak mengubah dataset."""
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from hk_common import DATA_DIR, NAMES, OUT, base_name, size_bin, source_of

SPLITS = ("train", "valid", "test")
EXPECT = dict(train=8409, valid=595, test=550)


def read_label(path):
    """Return [(kelas, jenis, w_norm, h_norm)]. Baris poligon diubah ke kotak pembungkus, sama dengan segments2boxes."""
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        v = np.array(p[1:], float)
        if len(v) == 4:
            out.append((int(p[0]), "box", v[2], v[3]))
        else:
            xy = v.reshape(-1, 2)
            out.append((int(p[0]), "polygon", np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    return out


def orb_features(path, orb):
    return orb.detectAndCompute(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), None)


def orb_inliers(fa, fb, bf):
    """Jumlah inlier homografi RANSAC. Tahan terhadap crop, rotasi, shear, flip tidak (flip dicek terpisah)."""
    ka, da = fa
    kb, db = fb
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = [m for m, n in (x for x in bf.knnMatch(da, db, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return 0 if mask is None else int(mask.sum())


def same_source_score(pa, pb, orb, bf):
    """Skor kemiripan, mencoba citra asli dan versi flip horizontal (Roboflow memakai flip 50%)."""
    a = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
    fb = orb_features(pb, orb)
    return max(orb_inliers(orb.detectAndCompute(x, None), fb, bf) for x in (a, cv2.flip(a, 1)))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []

    def add(bagian, split, item, nilai, kelas=""):
        rows.append(dict(bagian=bagian, split=split, item=item, kelas=kelas, nilai=nilai))

    names_by_split = {}
    for sp in SPLITS:
        imgs = sorted(p for p in (DATA_DIR / sp / "images").iterdir() if p.is_file())
        names_by_split[sp] = [p.name for p in imgs]
        n_cls, img_cls, kinds, src_img, src_obj = Counter(), Counter(), Counter(), Counter(), Counter()
        empty = mixed = 0
        size_cnt = defaultdict(Counter)
        for p in imgs:
            lb = read_label(DATA_DIR / sp / "labels" / f"{p.stem}.txt")
            src = source_of(p.name)
            src_img[src] += 1
            src_obj[src] += len(lb)
            empty += not lb
            mixed += len({k[1] for k in lb}) > 1
            for c in {k[0] for k in lb}:
                img_cls[c] += 1
            for c, kind, w, h in lb:
                n_cls[c] += 1
                kinds[kind] += 1
                if sp == "test":
                    size_cnt[size_bin(w * 512 * h * 512)][c] += 1
        add("gambar", sp, "jumlah_gambar", len(imgs))
        add("gambar", sp, "target_jumlah_gambar", EXPECT[sp])
        add("gambar", sp, "gambar_tanpa_objek", empty)
        add("gambar", sp, "gambar_label_campuran_poligon_dan_box", mixed)
        add("objek", sp, "baris_label_poligon", kinds["polygon"])
        add("objek", sp, "baris_label_box", kinds["box"])
        for c, n in enumerate(NAMES):
            add("objek", sp, "jumlah_objek", n_cls[c], n)
            add("objek", sp, "gambar_berisi_kelas", img_cls[c], n)
        for s in ("UAVVaste", "TACO", "Burangrang", "Unknown"):
            add("sumber", sp, f"gambar_{s}", src_img[s])
            add("sumber", sp, f"objek_{s}", src_obj[s])
        if sp == "test":
            for b in ("small", "medium", "large"):
                add("ukuran_objek_pada_512", sp, b, sum(size_cnt[b].values()))
                for c, n in enumerate(NAMES):
                    add("ukuran_objek_pada_512", sp, b, size_cnt[b][c], n)
            sz = Counter(Image.open(p).size for p in imgs)
            for k, v in sz.items():
                add("resolusi_file", sp, f"{k[0]}x{k[1]}", v)

    # Nama dasar sama di train dan test/valid: bisa kebetulan (nama TACO 000025 berulang antar batch) atau citra
    # sumber yang sama. Citra train diaugmentasi berat, jadi dicek dengan ORB + RANSAC. Ambang = kontrol pasangan acak.
    train_by_base = defaultdict(list)
    for f in names_by_split["train"]:
        train_by_base[base_name(f)].append(f)
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    rng = np.random.default_rng(0)
    pair_rows = []
    for sp in ("valid", "test"):
        shared = [f for f in names_by_split[sp] if base_name(f) in train_by_base]
        for f in shared:
            pa = DATA_DIR / sp / "images" / f
            score = max(same_source_score(pa, DATA_DIR / "train" / "images" / t, orb, bf) for t in train_by_base[base_name(f)])
            ctrl = [t for t in rng.choice(names_by_split["train"], 8, replace=False) if base_name(t) != base_name(f)]
            cs = max(same_source_score(pa, DATA_DIR / "train" / "images" / t, orb, bf) for t in ctrl)
            pair_rows.append(dict(split=sp, file=f, sumber=source_of(f), n_train_nama_sama=len(train_by_base[base_name(f)]),
                                  inlier_nama_sama=score, inlier_kontrol_acak=cs))
    P = pd.DataFrame(pair_rows)
    P.to_csv(OUT / "tumpang_tindih_train.csv", index=False)
    thr = int(P.inlier_kontrol_acak.max())
    add("tumpang_tindih_train", "semua", "ambang_inlier_ORB_maks_kontrol_acak", thr)
    for sp in ("valid", "test"):
        q = P[P.split == sp]
        add("tumpang_tindih_train", sp, "gambar_dengan_nama_dasar_sama_di_train", len(q))
        add("tumpang_tindih_train", sp, "terkonfirmasi_visual_inlier_di_atas_ambang", int((q.inlier_nama_sama > thr).sum()))
        for s_, g in q.groupby("sumber"):
            add("tumpang_tindih_train", sp, f"nama_sama_{s_}", len(g))
            add("tumpang_tindih_train", sp, f"terkonfirmasi_{s_}", int((g.inlier_nama_sama > thr).sum()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dataset_stats.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

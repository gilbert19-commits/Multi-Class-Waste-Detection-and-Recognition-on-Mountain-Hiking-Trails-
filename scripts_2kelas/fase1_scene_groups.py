"""
FASE 1.4 — SCENE GROUP + TEST CLEAN UNTUK dataset-2kelas (read-only terhadap dataset)

Prosedur sama dengan analysis/D0_dataset_audit.py (yang menghasilkan D0_scene_groups.csv):
  kandidat pasangan = nama sumber Roboflow sama + K_SEQ frame/foto berurutan + K_EMB tetangga
  embedding ResNet50 (flip-invariant) + K_DHASH tetangga dHash (flip-aware);
  tiap kandidat diuji SIFT (1000 fitur, ratio test 0.75) + RANSAC affine (reproj 5 px, dicoba
  dengan dan tanpa mirror); inlier >= 100 = adegan sama; group = komponen terhubung (union-find).
Perbedaan yang disengaja:
  - IMG_SIZE = 514 (ukuran citra dataset baru, dipakai untuk koordinat mirror), dicek dari file.
  - Embedding ResNet50 dihitung di CPU (tidak memakai GPU agar tidak bentrok dengan proses GPU lain).
Test clean: citra test yang group_id-nya juga muncul di train/valid dibuang (logika sama dengan
build_test_sets di scripts/01_evaluate_test.py).

Output (analysis_2kelas/):
  tables/D0_2kelas_scene_groups.csv, tables/D0_2kelas_scene_edges.csv (inlier >= 25)
  clean_test_2kelas.txt, fase1_scene_groups.md
"""
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset-2kelas"
OUT = ROOT / "analysis_2kelas"
TABLES = OUT / "tables"
SPLITS = ["train", "valid", "test"]
CLASSES = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
SOURCE_CLEAN_RE = re.compile(r"(_jpe?g\.rf\.[a-f0-9]+|_png\.rf\.[a-f0-9]+|\.rf\.[a-f0-9]+).*$", re.IGNORECASE)

STRONG = 100
GRAY = 25
K_EMB = 10
K_DHASH = 5
K_SEQ = 3
IMG_SIZE = 514
SEED = 0
DEVICE = "cpu"


def clean_source_name(fname):
    return SOURCE_CLEAN_RE.sub("", Path(fname).stem)


def source_of(src):
    if re.match(r"^(batch|BATCH)_", src):
        return "UAVVaste"
    if re.fullmatch(r"\d+", src):
        return "TACO"
    if re.match(r"^(WhatsApp-Video|IMG_|GOPR)", src):
        return "Burangrang"
    return "Unknown"


def seq_key(src):
    m = re.match(r"^(.*?)(\d+)\D*$", src)
    return (m.group(1), int(m.group(2))) if m else (src, 0)


def read_classes(p):
    c = Counter()
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                c[int(parts[0])] += 1
    return c


def load_index():
    rows = []
    for split in SPLITS:
        for img in sorted((DATA_DIR / split / "images").glob("*")):
            src = clean_source_name(img.name)
            rows.append(dict(path=img, split=split, name=img.name, src=src, source=source_of(src),
                             cls=read_classes(DATA_DIR / split / "labels" / (img.stem + ".txt"))))
    return rows


def dhash_bits(img):
    a = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


@torch.no_grad()
def embeddings(rows):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(DEVICE)
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    t0 = time.time()
    for i in range(0, len(rows), 64):
        x = torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i + 64]]).to(DEVICE)
        e = net(x) + net(torch.flip(x, dims=[3]))
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
        if (i // 64) % 20 == 0:
            print(f"   embedding {i + len(x)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(out).numpy()


def candidate_pairs(rows):
    n = len(rows)
    pairs = set()
    src_count = Counter()

    def add(i, j, tag):
        if i != j:
            k = (min(i, j), max(i, j))
            if k not in pairs:
                src_count[tag] += 1
            pairs.add(k)

    by_src = defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r["src"]].append(i)
    for idx in by_src.values():
        for a in idx:
            for b in idx:
                add(a, b, "nama_sumber_sama")

    by_prefix = defaultdict(list)
    for i, r in enumerate(rows):
        p, num = seq_key(r["src"])
        by_prefix[p].append((num, i))
    for lst in by_prefix.values():
        lst.sort()
        for k in range(len(lst)):
            for d in range(1, K_SEQ + 1):
                if k + d < len(lst):
                    add(lst[k][1], lst[k + d][1], "berurutan")

    print("[INFO] Menghitung embedding ResNet50 (CPU) ...", flush=True)
    E = embeddings(rows)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    for i in range(n):
        for j in np.argpartition(-S[i], K_EMB)[:K_EMB]:
            add(i, int(j), "resnet50")
    del S

    print("[INFO] Menghitung dHash ...", flush=True)
    H = np.array([dhash_bits(Image.open(r["path"])) for r in rows])
    Hf = np.array([dhash_bits(Image.open(r["path"]).transpose(Image.FLIP_LEFT_RIGHT)) for r in rows])
    for i in range(n):
        d = np.minimum((H != H[i]).sum(1), (Hf != H[i]).sum(1))
        d[i] = 999
        for j in np.argpartition(d, K_DHASH)[:K_DHASH]:
            add(i, int(j), "dhash")
    return sorted(pairs), src_count


class SiftMatcher:
    def __init__(self, rows):
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.bf = cv2.BFMatcher()
        self.kp, self.desc = [], []
        print("[INFO] Ekstraksi fitur SIFT ...", flush=True)
        for r in rows:
            g = cv2.imread(str(r["path"]), cv2.IMREAD_GRAYSCALE)
            k, d = self.sift.detectAndCompute(g, None)
            self.kp.append(np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32))
            self.desc.append(np.clip(d, 0, 255).astype(np.uint8) if d is not None else None)

    def match(self, i, j):
        da, db = self.desc[i], self.desc[j]
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return 0, False
        knn = self.bf.knnMatch(da.astype(np.float32), db.astype(np.float32), k=2)
        good = [m for m, n2 in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n2.distance]
        if len(good) < 8:
            return 0, False
        A = self.kp[i][[m.queryIdx for m in good]]
        B = self.kp[j][[m.trainIdx for m in good]]
        best = (0, False)
        for mirror in (False, True):
            A2 = A.copy()
            if mirror:
                A2[:, 0] = (IMG_SIZE - 1) - A2[:, 0]
            M, mask = cv2.estimateAffine2D(A2, B, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if mask is not None and int(mask.sum()) > best[0]:
                best = (int(mask.sum()), mirror)
        return best


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_groups(n, edges, thr):
    uf = UnionFind(n)
    for i, j, inl in edges:
        if inl >= thr:
            uf.union(i, j)
    roots = {}
    return [roots.setdefault(uf.find(i), len(roots)) for i in range(n)]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = load_index()
    n = len(rows)
    bad = [r["name"] for r in rows if Image.open(r["path"]).size != (IMG_SIZE, IMG_SIZE)]
    if bad:
        raise RuntimeError(f"{len(bad)} citra bukan {IMG_SIZE}x{IMG_SIZE}, mis. {bad[:3]}")
    print(f"[INFO] {n} citra dataset-2kelas dimuat", flush=True)

    pairs, pair_src = candidate_pairs(rows)
    print(f"[INFO] {len(pairs)} pasangan kandidat {dict(pair_src)} ({time.time() - t0:.0f}s)", flush=True)
    matcher = SiftMatcher(rows)
    edges = []
    for k, (i, j) in enumerate(pairs):
        inl, _ = matcher.match(i, j)
        edges.append((i, j, inl))
        if (k + 1) % 5000 == 0:
            print(f"   SIFT {k + 1}/{len(pairs)}  ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    neg = [matcher.match(int(a), int(b))[0] for a, b in rng.integers(0, n, size=(200, 2)) if a != b]

    gid = build_groups(n, edges, STRONG)
    for r, g in zip(rows, gid):
        r["group"] = g
    sizes = Counter(gid)

    pd.DataFrame([{"split": r["split"], "filename": r["name"], "source_name": r["src"], "sumber": r["source"],
                   "group_id": r["group"], **{CLASSES[k]: r["cls"].get(k, 0) for k in range(len(CLASSES))}}
                  for r in rows]).to_csv(TABLES / "D0_2kelas_scene_groups.csv", index=False)
    key = lambda i: f"{rows[i]['split']}/{rows[i]['name']}"
    pd.DataFrame([{"a": key(i), "b": key(j), "inliers": v} for i, j, v in edges if v >= GRAY]
                 ).sort_values("inliers", ascending=False).to_csv(TABLES / "D0_2kelas_scene_edges.csv", index=False)

    # ---- test clean
    splits_per_group = defaultdict(set)
    for r in rows:
        splits_per_group[r["group"]].add(r["split"])
    cross = {g for g, s in splits_per_group.items() if "test" in s and len(s) > 1}
    test = [r for r in rows if r["split"] == "test"]
    clean = [r for r in test if r["group"] not in cross]
    leaked = [r for r in test if r["group"] in cross]
    (OUT / "clean_test_2kelas.txt").write_text("\n".join(r["name"] for r in clean) + "\n", encoding="utf-8")
    pd.DataFrame([{"filename": r["name"], "group_id": r["group"],
                   "split_lain_di_group": ",".join(sorted(splits_per_group[r["group"]] - {"test"}))}
                  for r in leaked]).to_csv(TABLES / "D0_2kelas_test_bocor.csv", index=False)

    def inst(rs):
        c = Counter()
        for r in rs:
            c.update(r["cls"])
        return c

    ci, cf = inst(clean), inst(test)
    bg = lambda rs: sum(1 for r in rs if not r["cls"])
    strong_pairs = sum(1 for _, _, v in edges if v >= STRONG)
    gray_pairs = sum(1 for _, _, v in edges if GRAY <= v < STRONG)
    bins = Counter("1" if s == 1 else "2-5" if s <= 5 else "6-10" if s <= 10 else ">10" for s in sizes.values())
    src_tbl = pd.DataFrame([{"sumber": s, **{sp: sum(1 for r in rows if r["source"] == s and r["split"] == sp)
                                              for sp in SPLITS},
                             "test_clean": sum(1 for r in clean if r["source"] == s)}
                            for s in ["UAVVaste", "TACO", "Burangrang", "Unknown"]])
    unknown = sorted({re.sub(r"[0-9]+", "#", r["src"]) for r in rows if r["source"] == "Unknown"})

    md = ["# Fase 1.4 — Scene group dan test clean (dataset-2kelas)", "",
          f"- Citra: {n} (semua split), ukuran {IMG_SIZE}×{IMG_SIZE} (dicek dari file)",
          f"- Kandidat pasangan diuji: **{len(pairs):,}** (sumber kandidat pertama yang menambahkan pasangan: "
          f"{dict(pair_src)})",
          f"- Kalibrasi 200 pasangan acak: inlier maks **{max(neg)}**, p95 **{int(np.percentile(neg, 95))}**, "
          f"median **{int(np.median(neg))}**",
          f"- Pasangan adegan sama (inlier ≥ {STRONG}): **{strong_pairs:,}**; zona abu-abu {GRAY}–{STRONG - 1}: "
          f"**{gray_pairs:,}** (tidak dipakai untuk group, sama seperti group_id lama)",
          f"- Total group: **{len(sizes)}**; ukuran group: {dict(bins)}",
          f"- Group yang memuat test DAN train/valid: **{len(cross)}**", "",
          "## Test clean", "",
          "| | citra | background | " + " | ".join(CLASSES) + " | total instance |",
          "|---|---|---|" + "---|" * len(CLASSES) + "---|",
          f"| test full | {len(test)} | {bg(test)} | " + " | ".join(str(cf[k]) for k in range(len(CLASSES)))
          + f" | {sum(cf.values())} |",
          f"| test clean | {len(clean)} | {bg(clean)} | " + " | ".join(str(ci[k]) for k in range(len(CLASSES)))
          + f" | {sum(ci.values())} |",
          f"| dibuang | {len(leaked)} | {bg(leaked)} | "
          + " | ".join(str(cf[k] - ci[k]) for k in range(len(CLASSES))) + f" | {sum(cf.values()) - sum(ci.values())} |",
          "", "## Sumber data (pola nama file, sama seperti D0)", "", src_tbl.to_markdown(index=False), "",
          f"Pola nama Unknown: {', '.join(unknown) if unknown else '-'}", "",
          f"_Waktu komputasi: {(time.time() - t0) / 60:.1f} menit._"]
    (OUT / "fase1_scene_groups.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())

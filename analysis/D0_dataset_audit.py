"""
TASK D0 — AUDIT DATASET SUMBER (TA-1, NON-AUGMENTED) — READ-ONLY

1. Jumlah instance & citra per kelas di seluruh TA-1 (train+valid+test)
2. Breakdown asal data (UAVVaste / TACO / Burangrang) berbasis pola nama file
3. Scene group via SIFT + RANSAC (inlier >= 100 = adegan sama, 25-99 = zona abu-abu)
4. Simulasi split ulang 70:15:15 berbasis group, terstratifikasi per kelas
5. Kandidat anotasi tidak konsisten (adegan sama, label berbeda) via pemetaan box

Tidak mengubah apa pun di TA-1/ atau augmented/. Output:
  analysis/D0_dataset_audit.md
  analysis/tables/D0_*.csv
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
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "TA-1"
ANALYSIS_DIR = ROOT_DIR / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
REPORT_PATH = ANALYSIS_DIR / "D0_dataset_audit.md"

CLASSES = ["kertas_tisu", "logam_kaleng_kaca", "plastik", "puntung_rokok"]
SPLITS = ["train", "valid", "test"]
SOURCE_CLEAN_RE = re.compile(r"(_jpe?g\.rf\.[a-f0-9]+|_png\.rf\.[a-f0-9]+|\.rf\.[a-f0-9]+).*$", re.IGNORECASE)

STRONG = 100   # inlier >= STRONG  -> adegan sama
GRAY = 25      # GRAY <= inlier < STRONG -> zona abu-abu (kandidat)
K_EMB = 10     # tetangga terdekat via embedding ResNet50
K_DHASH = 5    # tetangga terdekat via dHash (flip-aware)
K_SEQ = 3      # tetangga urutan nomor frame/foto dalam pola nama yang sama
IMG_SIZE = 640
SEED = 0


# ---------------------------------------------------------------- helpers
def clean_source_name(fname: str) -> str:
    return SOURCE_CLEAN_RE.sub("", Path(fname).stem)


def source_of(src: str) -> str:
    """Heuristik asal data dari pola nama file (lihat E2/E3)."""
    if re.match(r"^(batch|BATCH)_", src):
        return "UAVVaste"
    if re.fullmatch(r"\d+", src):
        return "TACO"
    if re.match(r"^(WhatsApp-Video|IMG_|GOPR)", src):
        return "Burangrang"
    return "Unknown"


def seq_key(src: str):
    """(prefix, nomor) untuk mencari frame/foto berurutan, mis. IMG_5332 ~ IMG_5333."""
    m = re.match(r"^(.*?)(\d+)\D*$", src)
    return (m.group(1), int(m.group(2))) if m else (src, 0)


def read_labels(p: Path):
    """Box (c, x, y, w, h) ternormalisasi. Baris poligon (>5 kolom) dikonversi ke bbox seperti Ultralytics."""
    boxes = []
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                boxes.append((c, x, y, w, h))
            elif len(parts) > 5:
                pts = np.array(parts[1:], dtype=float).reshape(-1, 2)
                x1, y1 = pts.min(0)
                x2, y2 = pts.max(0)
                boxes.append((int(parts[0]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes


def load_index():
    rows = []
    for split in SPLITS:
        for img in sorted((DATA_DIR / split / "images").glob("*")):
            src = clean_source_name(img.name)
            boxes = read_labels(DATA_DIR / split / "labels" / (img.stem + ".txt"))
            rows.append(dict(path=img, split=split, name=img.name, src=src,
                             source=source_of(src), boxes=boxes,
                             cls=Counter(b[0] for b in boxes)))
    return rows


# ---------------------------------------------------------------- candidates
def dhash_bits(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


@torch.no_grad()
def embeddings(rows):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    for i in range(0, len(rows), 64):
        x = torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i + 64]]).to(dev)
        e = net(x) + net(torch.flip(x, dims=[3]))   # flip-invariant
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy()


def candidate_pairs(rows):
    n = len(rows)
    pairs = set()
    add = lambda i, j: pairs.add((min(i, j), max(i, j))) if i != j else None

    # (a) nama sumber Roboflow identik
    by_src = defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r["src"]].append(i)
    for idx in by_src.values():
        for a in idx:
            for b in idx:
                add(a, b)

    # (b) urutan frame/foto dalam pola nama yang sama
    by_prefix = defaultdict(list)
    for i, r in enumerate(rows):
        p, num = seq_key(r["src"])
        by_prefix[p].append((num, i))
    for lst in by_prefix.values():
        lst.sort()
        for k in range(len(lst)):
            for d in range(1, K_SEQ + 1):
                if k + d < len(lst):
                    add(lst[k][1], lst[k + d][1])

    # (c) embedding ResNet50
    print("[INFO] Menghitung embedding ResNet50 ...", flush=True)
    E = embeddings(rows)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    for i in range(n):
        for j in np.argpartition(-S[i], K_EMB)[:K_EMB]:
            add(i, int(j))

    # (d) dHash flip-aware
    H = np.array([dhash_bits(Image.open(r["path"])) for r in rows])
    Hf = np.array([dhash_bits(Image.open(r["path"]).transpose(Image.FLIP_LEFT_RIGHT)) for r in rows])
    for i in range(n):
        d = np.minimum((H != H[i]).sum(1), (Hf != H[i]).sum(1))
        d[i] = 999
        for j in np.argpartition(d, K_DHASH)[:K_DHASH]:
            add(i, int(j))
    return sorted(pairs)


# ---------------------------------------------------------------- SIFT
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
        """Return (inliers, mirror, M) — M memetakan koordinat piksel i (dimirror bila perlu) ke j."""
        da, db = self.desc[i], self.desc[j]
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return 0, False, None
        knn = self.bf.knnMatch(da.astype(np.float32), db.astype(np.float32), k=2)
        good = [m for m, n2 in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n2.distance]
        if len(good) < 8:
            return 0, False, None
        A = self.kp[i][[m.queryIdx for m in good]]
        B = self.kp[j][[m.trainIdx for m in good]]
        best = (0, False, None)
        for mirror in (False, True):
            A2 = A.copy()
            if mirror:
                A2[:, 0] = (IMG_SIZE - 1) - A2[:, 0]
            # affine penuh: rotasi + skala tak seragam (crop/stretch Roboflow)
            M, mask = cv2.estimateAffine2D(A2, B, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if mask is not None and int(mask.sum()) > best[0]:
                best = (int(mask.sum()), mirror, M)
        return best


# ---------------------------------------------------------------- label consistency
def box_xyxy(b):
    _, x, y, w, h = b
    return np.array([(x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)]) * IMG_SIZE


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def map_box(xyxy, mirror, M):
    x1, y1, x2, y2 = xyxy
    pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float64)
    if mirror:
        pts[:, 0] = (IMG_SIZE - 1) - pts[:, 0]
    pts = pts @ M[:, :2].T + M[:, 2]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def compare_labels(boxes_a, boxes_b, mirror, M):
    """Petakan box A ke B; hitung konflik kelas & box yang hilang (hanya box yang tetap di dalam frame)."""
    conflicts, missing = [], []
    for b in boxes_a:
        m = map_box(box_xyxy(b), mirror, M)
        area = max(1e-6, (m[2] - m[0]) * (m[3] - m[1]))
        clip = np.clip(m, 0, IMG_SIZE)
        inside = max(0, clip[2] - clip[0]) * max(0, clip[3] - clip[1]) / area
        if inside < 0.7:
            continue
        best_iou, best_cls = 0.0, None
        for c in boxes_b:
            v = iou(m, box_xyxy(c))
            if v > best_iou:
                best_iou, best_cls = v, c[0]
        if best_iou >= 0.3 and best_cls != b[0]:
            conflicts.append((CLASSES[b[0]], CLASSES[best_cls]))
        elif best_iou < 0.1:
            missing.append(CLASSES[b[0]])
    return conflicts, missing


# ---------------------------------------------------------------- grouping
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
    gid = [roots.setdefault(uf.find(i), len(roots)) for i in range(n)]
    return gid


# ---------------------------------------------------------------- split simulation
def simulate_split(rows, gid, ratios=(0.70, 0.15, 0.15), n_trials=300):
    G = defaultdict(lambda: np.zeros(1 + len(CLASSES)))
    for r, g in zip(rows, gid):
        G[g][0] += 1
        for c, v in r["cls"].items():
            G[g][1 + c] += v
    gids = list(G)
    vec = np.array([G[g] for g in gids])
    total = vec.sum(0)
    target = np.outer(ratios, total)
    rng = np.random.default_rng(SEED)

    # urutkan: grup yang memuat kelas paling langka dulu, lalu grup besar
    rarity = 1.0 / np.maximum(total[1:], 1)
    prio = (vec[:, 1:] > 0) @ rarity

    best = None
    for _ in range(n_trials):
        order = np.lexsort((rng.random(len(gids)), -vec[:, 0], -prio))
        acc = np.zeros_like(target)
        assign = np.empty(len(gids), dtype=int)
        for k in order:
            # pilih split dengan defisit relatif terbesar pada dimensi yang dimiliki grup ini
            dims = np.where(vec[k] > 0)[0]
            deficit = ((target - acc)[:, dims] / np.maximum(target[:, dims], 1e-9)).mean(1)
            s = int(np.argmax(deficit))
            assign[k] = s
            acc[s] += vec[k]
        # galat: deviasi relatif kuadrat terhadap target (citra + tiap kelas)
        err = (((acc - target) / np.maximum(target, 1e-9)) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, assign.copy(), acc.copy())
    _, assign, acc = best
    split_of_group = {g: ["train", "valid", "test"][a] for g, a in zip(gids, assign)}
    return split_of_group, acc, target


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    rows = load_index()
    n = len(rows)
    print(f"[INFO] {n} citra TA-1 dimuat", flush=True)
    TABLES_DIR.mkdir(exist_ok=True)

    # ---- 1. instance per kelas
    inst = Counter(); img_with = Counter(); per_split = defaultdict(Counter)
    for r in rows:
        inst.update(r["cls"]); img_with.update(r["cls"].keys()); per_split[r["split"]].update(r["cls"])
    n_bg = sum(1 for r in rows if not r["boxes"])
    t1 = pd.DataFrame([{
        "kelas": CLASSES[c], "instance": inst[c], "citra_mengandung": img_with[c],
        **{f"inst_{s}": per_split[s][c] for s in SPLITS},
    } for c in range(len(CLASSES))])
    t1.to_csv(TABLES_DIR / "D0_class_totals.csv", index=False)

    # ---- 2. asal data
    src_rows = []
    for s in ["UAVVaste", "TACO", "Burangrang", "Unknown"]:
        R = [r for r in rows if r["source"] == s]
        c = Counter()
        for r in R:
            c.update(r["cls"])
        src_rows.append({"sumber": s, "citra": len(R), "nama_sumber_unik": len({r['src'] for r in R}),
                         **{CLASSES[k]: c[k] for k in range(len(CLASSES))}})
    t2 = pd.DataFrame(src_rows)
    unknown_names = sorted({r["src"] for r in rows if r["source"] == "Unknown"})

    # ---- 3. SIFT pairs
    pairs = candidate_pairs(rows)
    print(f"[INFO] {len(pairs)} pasangan kandidat", flush=True)
    key = lambda i, j: f"{rows[i]['split']}/{rows[i]['name']}|{rows[j]['split']}/{rows[j]['name']}"
    matcher = SiftMatcher(rows)
    results = {}
    for k, (i, j) in enumerate(pairs):
        results[(i, j)] = matcher.match(i, j)
        if (k + 1) % 2000 == 0:
            print(f"   SIFT {k + 1}/{len(pairs)}  ({time.time() - t0:.0f}s)", flush=True)
    edges = [(i, j, v[0]) for (i, j), v in results.items()]

    # kalibrasi: pasangan acak (hampir pasti adegan berbeda)
    rng = np.random.default_rng(SEED)
    neg = [matcher.match(int(a), int(b))[0] for a, b in rng.integers(0, n, size=(200, 2)) if a != b]

    gid = build_groups(n, edges, STRONG)
    gid_cons = build_groups(n, edges, GRAY)
    # paling aman: konservatif + nama sumber sama (kecuali nama numerik TACO yang sering bertabrakan)
    by_name = defaultdict(list)
    for i, r in enumerate(rows):
        if not re.fullmatch(r"\d+", r["src"]):
            by_name[r["src"]].append(i)
    name_edges = [(idx[0], j, GRAY) for idx in by_name.values() for j in idx[1:]]
    gid_safe = build_groups(n, edges + name_edges, GRAY)
    for r, g, gc, gs in zip(rows, gid, gid_cons, gid_safe):
        r["group"], r["group_cons"], r["group_safe"] = g, gc, gs

    def group_stats(gids):
        sizes = Counter(gids)
        bins = Counter()
        for s in sizes.values():
            bins["1" if s == 1 else "2-5" if s <= 5 else "6-10" if s <= 10 else ">10"] += 1
        return sizes, bins

    sizes, bins = group_stats(gid)
    sizes_c, bins_c = group_stats(gid_cons)
    # indikator recall: citra dengan nama sumber sama (non-numerik, bukan tabrakan TACO) yang terpecah
    name_groups = defaultdict(set)
    for r in rows:
        if not re.fullmatch(r"\d+", r["src"]):
            name_groups[r["src"]].add(r["group"])
    name_split = sum(1 for s in name_groups.values() if len(s) > 1)
    gray_edges = [(i, j, v) for i, j, v in edges if GRAY <= v < STRONG]
    gray_cross = sum(1 for i, j, _ in gray_edges if gid[i] != gid[j])

    top = []
    for g, s in sizes.most_common(10):
        members = [r for r in rows if r["group"] == g]
        c = Counter()
        for r in members:
            c.update(r["cls"])
        top.append({
            "group_id": g, "citra": s,
            "split": dict(Counter(r["split"] for r in members)),
            "sumber": dict(Counter(r["source"] for r in members)),
            "contoh": members[0]["src"],
            **{CLASSES[k]: c[k] for k in range(len(CLASSES))},
        })
    t3 = pd.DataFrame(top)

    # leakage ulang berbasis group: group yang tersebar di >1 split
    split_spread = defaultdict(set)
    for r in rows:
        split_spread[r["group"]].add(r["split"])
    multi = {g for g, s in split_spread.items() if len(s) > 1}
    leak_imgs = Counter(r["split"] for r in rows if r["group"] in multi)

    # puntung_rokok per group
    punt = defaultdict(lambda: [0, 0, set()])
    for r in rows:
        if r["cls"].get(3):
            punt[r["group"]][0] += r["cls"][3]
            punt[r["group"]][1] += 1
            punt[r["group"]][2].add(r["src"])
    punt_tbl = sorted(([g, v[0], v[1], sizes[g], ", ".join(sorted(v[2])[:4])] for g, v in punt.items()),
                      key=lambda x: -x[1])
    punt_cons_groups = len({r["group_cons"] for r in rows if r["cls"].get(3)})

    pd.DataFrame([{"split": r["split"], "filename": r["name"], "source_name": r["src"], "sumber": r["source"],
                   "group_id": r["group"], "group_id_konservatif": r["group_cons"],
                   **{CLASSES[k]: r["cls"].get(k, 0) for k in range(len(CLASSES))}} for r in rows]
                 ).to_csv(TABLES_DIR / "D0_scene_groups.csv", index=False)
    pd.DataFrame([{"a": key(i, j).split("|")[0], "b": key(i, j).split("|")[1], "inliers": v}
                  for i, j, v in edges if v >= GRAY]).sort_values("inliers", ascending=False
                  ).to_csv(TABLES_DIR / "D0_scene_edges.csv", index=False)

    # ---- 4. simulasi split
    def split_report(gids_name):
        sog, acc, target = simulate_split(rows, [r[gids_name] for r in rows])
        tbl = pd.DataFrame(acc.astype(int), index=["train", "valid", "test"],
                           columns=["citra"] + CLASSES)
        tbl["citra_%"] = (tbl["citra"] / tbl["citra"].sum() * 100).round(1)
        return sog, tbl

    sog, sim = split_report("group")
    sog_c, sim_c = split_report("group_cons")
    sog_s, sim_s = split_report("group_safe")
    sim.to_csv(TABLES_DIR / "D0_split_simulation.csv")
    sim_c.to_csv(TABLES_DIR / "D0_split_simulation_konservatif.csv")
    sim_s.to_csv(TABLES_DIR / "D0_split_simulation_aman.csv")
    pd.DataFrame([{"split_asli": r["split"], "filename": r["name"],
                   "group_id": r["group"], "split_simulasi": sog[r["group"]],
                   "group_id_aman": r["group_safe"], "split_simulasi_aman": sog_s[r["group_safe"]]}
                  for r in rows]).to_csv(TABLES_DIR / "D0_split_simulation_assignment.csv", index=False)
    n_safe = len(set(gid_safe))
    punt_safe_groups = len({r["group_safe"] for r in rows if r["cls"].get(3)})

    # ---- 5. anotasi tidak konsisten
    incons = []
    for (i, j), (inl, mirror, M) in results.items():
        if inl < STRONG or M is None:
            continue
        a, b = rows[i], rows[j]
        try:
            Minv = cv2.invertAffineTransform(M)
        except cv2.error:
            continue
        c1, m1 = compare_labels(a["boxes"], b["boxes"], mirror, M)
        # arah balik: B -> A. Bila mirror, peta balik = invers lalu mirror.
        Mb = Minv.copy()
        if mirror:
            Mb = np.array([[-1, 0, IMG_SIZE - 1], [0, 1, 0]], dtype=np.float64) @ np.vstack([Minv, [0, 0, 1]])
        c2, m2 = compare_labels(b["boxes"], a["boxes"], False, Mb)
        if c1 or c2 or m1 or m2:
            incons.append(dict(
                a=f"{a['split']}/{a['name']}", b=f"{b['split']}/{b['name']}", sumber=a["source"], inliers=inl,
                konflik_kelas=len(c1) + len(c2), box_hilang=len(m1) + len(m2),
                detail_konflik="; ".join(f"{x}->{y}" for x, y in c1 + c2),
                kelas_hilang="; ".join(m1 + m2),
                label_a=dict((CLASSES[k], v) for k, v in a["cls"].items()),
                label_b=dict((CLASSES[k], v) for k, v in b["cls"].items()),
            ))
    t5 = pd.DataFrame(incons)
    if len(t5):
        t5 = t5.sort_values(["konflik_kelas", "box_hilang"], ascending=False)
    t5.to_csv(TABLES_DIR / "D0_label_inconsistency.csv", index=False)
    strong_pairs = sum(1 for v in results.values() if v[0] >= STRONG)
    n_conf_pairs = int((t5["konflik_kelas"] > 0).sum()) if len(t5) else 0
    n_miss_pairs = int((t5["box_hilang"] > 0).sum()) if len(t5) else 0
    conf_types = Counter()
    miss_cls = Counter()
    for x in incons:
        for t in filter(None, x["detail_konflik"].split("; ")):
            conf_types[t] += 1
        for t in filter(None, x["kelas_hilang"].split("; ")):
            miss_cls[t] += 1
    incons_imgs = len({x["a"] for x in incons} | {x["b"] for x in incons})

    # ---------------------------------------------------------------- report
    md = []
    w = md.append
    w("# Laporan Audit Dataset Sumber (TASK D0)")
    w(f"**Tanggal:** {time.strftime('%Y-%m-%d %H:%M')}  ")
    w("**Cakupan:** `TA-1/` (non-augmented) train + valid + test, "
      f"{n} citra. Read-only, tidak ada file dataset yang diubah.\n")
    w("---\n")
    w("## 1. Instance per Kelas (Seluruh TA-1)\n")
    w(t1.to_markdown(index=False))
    n_poly = sum(1 for lf in DATA_DIR.glob("*/labels/*.txt")
                 for line in lf.read_text().splitlines() if len(line.split()) > 5)
    w(f"\nFormat label: **{n_poly}** dari {sum(inst.values())} baris adalah poligon segmentasi "
      "(dikonversi ke bbox min/max seperti Ultralytics), sisanya bbox `xywh`.")
    w(f"\nCitra tanpa anotasi (background): **{n_bg}**. "
      f"Total instance puntung_rokok di seluruh dataset: **{inst[3]}** "
      f"(train {per_split['train'][3]}, valid {per_split['valid'][3]}, test {per_split['test'][3]}).\n")
    w("## 2. Asal Data\n")
    w("> Penentuan sumber berbasis **pola nama file** (heuristik, belum ada tagging manual — "
      "`manual_source_tags.csv` belum pernah diekspor): `batch_*`/`BATCH_*` = UAVVaste, "
      "nama numerik (`000005`) = TACO, `WhatsApp-Video*`/`IMG_*`/`GOPR*` = Burangrang. "
      f"Unknown = {', '.join(sorted({re.sub(r'[0-9]+', '#', s) for s in unknown_names}))}.\n")
    w(t2.to_markdown(index=False))
    w("\nCatatan: angka citra adalah jumlah **file** (termasuk varian rotasi/duplikat Roboflow), "
      "bukan citra mentah unik.\n")
    w("## 3. Scene Group (SIFT + RANSAC)\n")
    w(f"- Kandidat pasangan: nama sumber sama + {K_SEQ} frame/foto berurutan + {K_EMB} tetangga embedding "
      f"ResNet50 + {K_DHASH} tetangga dHash (flip-aware) → **{len(pairs):,} pasangan diuji**.")
    w(f"- Kalibrasi 200 pasangan acak: inlier maks **{max(neg)}**, p95 **{int(np.percentile(neg, 95))}**, "
      f"median **{int(np.median(neg))}**.")
    w(f"- Pasangan adegan sama (inlier ≥ {STRONG}): **{strong_pairs:,}**. "
      f"Zona abu-abu ({GRAY}–{STRONG - 1}): **{len(gray_edges):,}** pasangan, "
      f"**{gray_cross}** di antaranya menghubungkan group yang berbeda.")
    w(f"- Indikator recall: **{name_split}** dari {len(name_groups)} nama sumber non-numerik "
      "masih terpecah ke >1 group (idealnya 0; sisa ini adalah varian yang tidak tertangkap SIFT).\n")
    gtbl = pd.DataFrame([
        {"ukuran group": b, f"jumlah group (≥{STRONG})": bins.get(b, 0),
         f"citra (≥{STRONG})": sum(s for s in sizes.values()
                                    if (b == "1" and s == 1) or (b == "2-5" and 2 <= s <= 5)
                                    or (b == "6-10" and 6 <= s <= 10) or (b == ">10" and s > 10)),
         f"jumlah group (≥{GRAY}, konservatif)": bins_c.get(b, 0)}
        for b in ["1", "2-5", "6-10", ">10"]])
    w(f"**Total group: {len(sizes)}** (ambang ≥{STRONG}); "
      f"{len(sizes_c)} bila zona abu-abu ikut digabung (konservatif).\n")
    w(gtbl.to_markdown(index=False))
    w("\n### 10 Group Terbesar\n")
    w(t3.to_markdown(index=False))
    w("\n### Leakage berbasis group (split Roboflow saat ini)\n")
    w(f"- Group yang tersebar di lebih dari satu split: **{len(multi)}**")
    w(f"- Citra di dalam group tersebut: train {leak_imgs['train']}, valid {leak_imgs['valid']}, "
      f"test **{leak_imgs['test']}** dari {sum(1 for r in rows if r['split'] == 'test')}\n")
    w("### Sebaran puntung_rokok per group\n")
    w(pd.DataFrame(punt_tbl, columns=["group_id", "instance puntung", "citra dgn puntung",
                                      "ukuran group", "contoh nama"]).to_markdown(index=False))
    w(f"\nPuntung rokok tersebar di **{len(punt_tbl)} group** (ambang ≥{STRONG}) / "
      f"**{punt_cons_groups} group** (konservatif).\n")
    w("## 4. Simulasi Split Ulang 70:15:15 Berbasis Group\n")
    w("Semua citra dalam satu group masuk split yang sama. Greedy terstratifikasi "
      f"(grup dengan kelas paling langka ditempatkan dulu), {300} percobaan acak, dipilih "
      "galat relatif terkecil terhadap target per kelas.\n")
    w(f"### a) Group ambang ≥{STRONG}\n")
    w(sim.to_markdown())
    w(f"\n### b) Group konservatif (≥{GRAY}, zona abu-abu ikut digabung)\n")
    w(sim_c.to_markdown())
    w(f"\n### c) Group aman (konservatif + nama sumber sama, kecuali nama numerik TACO) — {n_safe} group\n")
    w(sim_s.to_markdown())
    w(f"\npuntung_rokok tersebar di {punt_safe_groups} group aman.\n")
    w(f"\n**puntung_rokok di test:** {int(sim.loc['test', 'puntung_rokok'])} instance (≥{STRONG}), "
      f"{int(sim_c.loc['test', 'puntung_rokok'])} instance (konservatif), "
      f"{int(sim_s.loc['test', 'puntung_rokok'])} instance (aman).\n")
    w("## 5. Kandidat Anotasi Tidak Konsisten\n")
    w(f"Untuk setiap pasangan adegan sama (inlier ≥ {STRONG}), box di citra A dipetakan ke citra B "
      "lewat transformasi RANSAC (dan sebaliknya). Box yang tetap ≥70% di dalam frame dicek:")
    w("- **konflik kelas**: box terpetakan tumpang tindih (IoU ≥ 0.3) dengan box berkelas lain")
    w("- **box hilang**: tidak ada box apa pun di lokasi itu (IoU < 0.1) — objek dianotasi di satu citra tapi tidak di pasangannya\n")
    w(f"- Pasangan adegan sama yang diperiksa: **{strong_pairs:,}**")
    w(f"- Pasangan dengan **konflik kelas**: **{n_conf_pairs}**")
    w(f"- Pasangan dengan **box hilang**: **{n_miss_pairs}**")
    w(f"- Citra unik yang terlibat: **{incons_imgs}**\n")
    if len(t5):
        strong_by_src = Counter(rows[i]["source"] for (i, j), v in results.items() if v[0] >= STRONG)
        per_src = t5.assign(konflik=t5.konflik_kelas > 0, hilang=t5.box_hilang > 0).groupby("sumber")[
            ["konflik", "hilang"]].sum().astype(int)
        per_src["pasangan_adegan_sama"] = [strong_by_src[s] for s in per_src.index]
        w("Per sumber (jumlah pasangan):\n")
        w(per_src.to_markdown())
        w("")
    w("Jenis konflik kelas (A→B):\n")
    w(pd.DataFrame(conf_types.most_common(), columns=["konflik", "jumlah box"]).to_markdown(index=False)
      if conf_types else "_tidak ada_")
    w("\nKelas pada box hilang:\n")
    w(pd.DataFrame(miss_cls.most_common(), columns=["kelas", "jumlah box"]).to_markdown(index=False)
      if miss_cls else "_tidak ada_")
    w("\n> Catatan: \"box hilang\" pada frame video berurutan bisa sah (objek bergerak/terhalang); "
      "konflik kelas pada lokasi yang sama hampir pasti kesalahan anotasi. Daftar lengkap: "
      "`analysis/tables/D0_label_inconsistency.csv`.\n")
    w("## File Output\n")
    for f in ["D0_class_totals.csv", "D0_scene_groups.csv", "D0_scene_edges.csv", "D0_split_simulation.csv",
              "D0_split_simulation_konservatif.csv", "D0_split_simulation_aman.csv",
              "D0_split_simulation_assignment.csv",
              "D0_label_inconsistency.csv"]:
        w(f"- `analysis/tables/{f}`")
    w(f"\n_Waktu komputasi: {(time.time() - t0) / 60:.1f} menit._")
    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"[DONE] {REPORT_PATH}  ({(time.time() - t0) / 60:.1f} menit)")


if __name__ == "__main__":
    sys.exit(main())

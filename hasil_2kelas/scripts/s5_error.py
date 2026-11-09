"""LANGKAH 5: confusion matrix (dari validator, conf 0.25, IoU 0.45) dan recall per ukuran/sumber (conf 0.25, IoU 0.5).

Model terbaik tiap family berdasarkan mAP@50-95 test. Pencocokan untuk recall mengikuti match_predictions Ultralytics
(IoU >= ambang, kelas sama, urut IoU, unik per prediksi lalu per GT), dihitung setelah prediksi disaring conf >= 0.25.
Sisa GT dan prediksi yang tidak cocok lalu dicocokkan tanpa memandang kelas untuk memisahkan salah kelas dari terlewat.
"""
import json
import pickle
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from hk_common import CACHE, COL_W, FAMILIES, MODELS, NAMES, OUT, PAPER, VALDIR, label, paper_style, size_bin, source_of  # noqa: E402

CONF, IOU = 0.25, 0.5
CM_CONF, CM_IOU = 0.25, 0.45
RAMP = LinearSegmentedColormap.from_list("ramp_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def unique_match(iou, mask):
    gi, pi = np.nonzero(mask)
    if not len(gi):
        return np.zeros((0, 2), int)
    m = np.stack([gi, pi], 1)
    m = m[np.argsort(-iou[gi, pi], kind="stable")]
    m = m[np.unique(m[:, 1], return_index=True)[1]]
    m = m[np.argsort(-iou[m[:, 0], m[:, 1]], kind="stable")]
    return m[np.unique(m[:, 0], return_index=True)[1]]


def match_image(im, conf=CONF, thr=IOU):
    """Return (gs, ps, keep). gs per GT: 1 benar, 2 salah kelas, 0 terlewat. ps per prediksi tersaring: 1 TP, 2 salah kelas, 0 FP."""
    keep = np.nonzero(im["pr_conf"] >= conf)[0]
    gc, gb = im["gt_cls"], im["gt_box"]
    pc, pb = im["pr_cls"][keep], im["pr_box"][keep]
    iou = iou_matrix(gb, pb)
    gs, ps = np.zeros(len(gc), int), np.zeros(len(pc), int)
    if iou.size:
        for g, p in unique_match(iou, (iou >= thr) & (gc[:, None] == pc[None, :])):
            gs[g], ps[p] = 1, 1
        rest = (iou >= thr) & (gs == 0)[:, None] & (ps == 0)[None, :]
        for g, p in unique_match(iou, rest):
            gs[g], ps[p] = 2, 2
    return gs, ps, keep


def load_test(m):
    d = pickle.loads((CACHE / f"{m}_2kelas_test.pkl").read_bytes())["images"]
    files = sorted(d)
    return files, [d[f] for f in files]


def best_models():
    T = pd.read_csv(OUT / "metrics_test.csv")
    lab2m = {label(m): m for m in MODELS}
    return {f: lab2m[T[T.family == f].sort_values("mAP50_95", ascending=False).iloc[0]["model"]] for f in FAMILIES}


def plot_cm(arr, labels, title, path, normalized):
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.86))
    vmax = 1.0 if normalized else max(arr[:-1, :-1].max(), 1)
    ax.imshow(np.nan_to_num(np.minimum(arr, vmax)), cmap=RAMP, vmin=0, vmax=vmax)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if i == j == arr.shape[0] - 1:
                ax.text(j, i, "n/a", ha="center", va="center", color="#52514e", fontsize=8)
                continue
            v = arr[i, j]
            ax.text(j, i, f"{v:.2f}" if normalized else f"{int(v)}", ha="center", va="center", fontsize=9,
                    color="white" if v > vmax * 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paper_style()
    raw = json.loads((OUT / "_eval_raw.json").read_text())
    (OUT / "confusion").mkdir(exist_ok=True)
    best = best_models()
    print("terbaik:", best)
    labels = ["lightweight\nlitter", "metal and\nglass", "background"]
    idx_names = [PAPER[n] for n in NAMES] + ["background"]
    rows = []
    for f, m in best.items():
        lab = label(m)
        cm = np.array(raw[f"{m}|test"]["confusion_matrix"])  # baris prediksi, kolom GT (konvensi Ultralytics)
        norm = cm / (cm.sum(0, keepdims=True) + 1e-9)        # dinormalisasi per kolom, sama dengan Ultralytics
        pd.DataFrame(cm.astype(int), index=[f"pred {x}" for x in idx_names], columns=[f"GT {x}" for x in idx_names]).to_csv(
            OUT / "confusion" / f"cm_{lab}_mentah.csv")
        pd.DataFrame(norm, index=[f"pred {x}" for x in idx_names], columns=[f"GT {x}" for x in idx_names]).to_csv(
            OUT / "confusion" / f"cm_{lab}_ternormalisasi.csv")
        plot_cm(cm, labels, f"{lab} (conf {CM_CONF}, IoU {CM_IOU})", OUT / "confusion" / f"cm_{lab}_mentah.png", False)
        plot_cm(norm, labels, f"{lab} (conf {CM_CONF}, IoU {CM_IOU}, normalized)", OUT / "confusion" / f"cm_{lab}_ternormalisasi.png", True)
        src_dir = VALDIR / "test" / f"{m}_2kelas"
        for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            shutil.copy2(src_dir / name, OUT / "confusion" / f"ultralytics_{lab}_{name}")

        files, imgs = load_test(m)
        recs = []
        for fname, im in zip(files, imgs):
            gs, _, _ = match_image(im)
            wh = im["gt_box"][:, 2:] - im["gt_box"][:, :2]
            for g in range(len(gs)):
                recs.append(dict(file=fname, sumber=source_of(fname), kelas=PAPER[NAMES[im["gt_cls"][g]]],
                                 ukuran=size_bin(wh[g, 0] * wh[g, 1]), state=gs[g]))
        R = pd.DataFrame(recs)
        n_img_src = pd.Series([source_of(x) for x in files]).value_counts()

        def row(dim, kat, kelas, g):
            n = len(g)
            return dict(model=lab, family=f, dimensi=dim, kategori=kat, kelas=kelas,
                        n_gambar=int(n_img_src.get(kat, 0)) if dim == "sumber" else "", GT=n, TP=int((g.state == 1).sum()),
                        FN_terlewat=int((g.state == 0).sum()), FN_salah_kelas=int((g.state == 2).sum()),
                        recall=(g.state == 1).mean() if n else np.nan, conf=CONF, iou=IOU)

        rows.append(row("semua", "semua", "semua", R))
        for k, g in R.groupby("kelas"):
            rows.append(row("kelas", "semua", k, g))
        for k in ("small", "medium", "large"):
            g = R[R.ukuran == k]
            rows.append(row("ukuran", k, "semua", g))
            for c, gg in g.groupby("kelas"):
                rows.append(row("ukuran", k, c, gg))
        for k in ("UAVVaste", "TACO", "Burangrang", "Unknown"):
            g = R[R.sumber == k]
            rows.append(row("sumber", k, "semua", g))
            for c, gg in g.groupby("kelas"):
                rows.append(row("sumber", k, c, gg))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "recall_breakdown.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(B[B.kelas == "semua"].round(4).to_string(index=False))
    print("[SELESAI s5]")


if __name__ == "__main__":
    main()

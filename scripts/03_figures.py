"""
03_figures.py — figure siap cetak (PNG 300 dpi) untuk laporan.

Sumber data (tidak ada training, tidak ada perubahan dataset/bobot):
  results/test_metrics_overall.csv   (01_evaluate_test.py)
  results/_preds/*.pkl               (cache prediksi validator dari 02_error_analysis.py)
  runs/**/results.csv                (log training Ultralytics)

Output: results/figures/
  01_map_per_varian.png              mAP@50 dan mAP@50-95, test full vs clean
  02_selisih_full_vs_clean.png       penurunan mAP dari test full ke test clean
  03_tradeoff_akurasi_vs_waktu.png   mAP@50-95 vs ms/citra
  04_kurva_training_yolov12.png      loss dan mAP per epoch, semua varian YOLOv12
  05_kurva_pr_v12s_vs_v8s_<set>.png  kurva PR per kelas, YOLOv12s aug vs YOLOv8s aug
  06_contoh_deteksi_benar.png        grid 3x3 deteksi benar (test clean)
  07_contoh_kegagalan.png            grid 2x3 kegagalan terparah + dugaan penyebab

Jalankan: python scripts/03_figures.py
"""
import importlib.util
import sys
import textwrap
import time
import traceback
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"

spec = importlib.util.spec_from_file_location("eval02", ROOT / "scripts" / "02_error_analysis.py")
eval02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval02)
eval01 = eval02.eval01

DPI = 300
SEED = 0
VARIANTS = list("nsmlx")
NAMES_ID = eval02.NAMES_ID
CM_CONF = eval02.CM_CONF
FOCUS = "s"  # varian utama untuk figure 5-7 (YOLOv12s aug = YOLOv12 terbaik di test clean)

# Palet kategorikal tervalidasi (dataviz reference palette, slot berurutan, tidak pernah diputar).
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = [  # (model, augmentasi) -> label, warna, marker
    (("YOLOv12", "tidak"), "YOLOv12 (non-augmentasi)", SLOT[0], "o"),
    (("YOLOv12", "ya"), "YOLOv12 (augmentasi)", SLOT[1], "s"),
    (("YOLOv8", "ya"), "YOLOv8 (augmentasi)", SLOT[2], "^"),
]
CLASS_COLOR = {"kertas_tisu": SLOT[0], "logam_kaleng_kaca": SLOT[1], "plastik": SLOT[2], "puntung_rokok": SLOT[3]}
GT_COLOR, PRED_COLOR = "#00c853", "#ff1744"  # diminta: GT hijau, prediksi merah (dibedakan juga oleh gaya garis)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def label(model, varian, aug):
    return f"{model}{varian}" + (" aug" if aug == "ya" else "")


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}", flush=True)


def footnote(fig, text):
    fig.text(0.01, -0.01, text, ha="left", va="top", fontsize=8, color=INK2)


def load_overall():
    ov = pd.read_csv(OUT / "test_metrics_overall.csv", dtype={"varian": str})
    # mAP 3 kelas: puntung_rokok hanya 2 instance di test full dan 0 di test clean,
    # jadi full dan clean hanya sebanding pada 3 kelas.
    ov["m50"], ov["m5095"] = ov["mAP50_3kelas"], ov["mAP50_95_3kelas"]
    ov["nama"] = [label(*r) for r in ov[["model", "varian", "augmentasi"]].itertuples(index=False)]
    return ov


def model_order(ov):
    keys = ov[["model", "varian", "augmentasi"]].drop_duplicates()
    rank = {g[0]: i for i, g in enumerate(GROUPS)}
    return sorted(keys.itertuples(index=False, name=None),
                  key=lambda k: (rank.get((k[0], k[2]), 9), VARIANTS.index(k[1]) if k[1] in VARIANTS else 9))


# ---------------------------------------------------------------- 1. mAP per varian
def fig_map(ov):
    n_img = ov.groupby("test_set").n_images.first()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
    w = 0.26
    for r, (col, ylab) in enumerate([("m50", "mAP@50"), ("m5095", "mAP@50-95")]):
        for c, ts in enumerate(["full", "clean"]):
            ax = axes[r, c]
            d = ov[ov.test_set == ts]
            best = d[col].max()
            for gi, ((model, aug), glabel, color, _) in enumerate(GROUPS):
                for vi, v in enumerate(VARIANTS):
                    row = d[(d.model == model) & (d.augmentasi == aug) & (d.varian == v)]
                    if not len(row):
                        continue
                    y = row[col].iloc[0]
                    x = vi + (gi - 1) * w
                    ax.bar(x, y, w, color=color, edgecolor="white", linewidth=0.8)
                    if np.isclose(y, best):
                        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 2), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks(range(len(VARIANTS)), [f"varian {v}" for v in VARIANTS])
            ax.grid(axis="x", visible=False)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 0:
                ax.set_title(f"Test {ts} ({n_img[ts]} citra)")
        axes[r, 0].set_ylim(0, ov[col].max() * 1.15)
    fig.legend(handles=[Patch(color=g[2], label=g[1]) for g in GROUPS], loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Akurasi deteksi pada data uji per varian model", y=1.07, fontsize=12, fontweight="bold")
    fig.tight_layout()
    footnote(fig, "mAP dihitung atas 3 kelas (kertas/tisu, logam/kaleng/kaca, plastik) agar test full dan clean "
                  "sebanding;\npuntung rokok hanya 2 instance di test full dan 0 di test clean. "
                  "YOLOv8 hanya dilatih varian n, s, m. Angka tebal = nilai tertinggi per panel.")
    save(fig, "01_map_per_varian.png")


# ---------------------------------------------------------------- 2. selisih full vs clean
def fig_delta(ov):
    keys = model_order(ov)
    piv = ov.set_index(["model", "varian", "augmentasi", "test_set"])
    d50 = [piv.loc[k + ("clean",), "m50"] - piv.loc[k + ("full",), "m50"] for k in keys]
    d95 = [piv.loc[k + ("clean",), "m5095"] - piv.loc[k + ("full",), "m5095"] for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w / 2, d50, w, color=SLOT[0], edgecolor="white", linewidth=0.8, label="Δ mAP@50")
    ax.bar(x + w / 2, d95, w, color=SLOT[1], edgecolor="white", linewidth=0.8, label="Δ mAP@50-95")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axhline(np.mean(d50), color=SLOT[0], linewidth=1.2, linestyle="--",
               label=f"rata-rata Δ mAP@50 = {np.mean(d50):+.3f}")
    ax.axhline(np.mean(d95), color=SLOT[1], linewidth=1.2, linestyle=":",
               label=f"rata-rata Δ mAP@50-95 = {np.mean(d95):+.3f}")
    ax.set_xticks(x, [label(*k) for k in keys], rotation=35, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Selisih mAP (clean − full)")
    # pemisah antar kelompok model
    for i in range(1, len(keys)):
        if (keys[i][0], keys[i][2]) != (keys[i - 1][0], keys[i - 1][2]):
            ax.axvline(i - 0.5, color=GRID, linewidth=1)
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.set_title("Dampak overlap scene antar split: penurunan mAP dari test full ke test clean",
                 fontweight="bold")
    fig.tight_layout()
    footnote(fig, "Test clean = test full tanpa citra yang satu grup scene dengan citra di train/valid "
                  "(audit D0). Nilai negatif = mAP lebih rendah\npada data yang benar-benar baru, "
                  "indikasi test full terlalu optimis. mAP 3 kelas.")
    save(fig, "02_selisih_full_vs_clean.png")


# ---------------------------------------------------------------- 3. trade-off
def pareto(xs, ys):
    """Titik yang tidak didominasi (lebih cepat DAN lebih akurat), urut menurut x."""
    o = np.argsort(xs)
    front, best = [], -np.inf
    for i in o:
        if ys[i] > best:
            front.append(i)
            best = ys[i]
    return front


def repel_labels(xs, ys, texts, char_w=0.33, h=0.0075, iters=300):
    """Geser posisi y label (satuan data) sampai kotak label tidak saling tumpang tindih.
    char_w = lebar satu karakter dalam ms, h = tinggi label dalam mAP (disetel untuk figsize ini)."""
    ly = np.array(ys, float)
    x0 = np.asarray(xs) + 0.6
    x1 = x0 + np.array([len(t) for t in texts]) * char_w
    for _ in range(iters):
        moved = False
        for i in range(len(ly)):
            for j in range(i + 1, len(ly)):
                if x0[i] < x1[j] and x0[j] < x1[i] and abs(ly[i] - ly[j]) < h:
                    push = (h - abs(ly[i] - ly[j])) / 2 + 1e-4
                    s = 1 if (ly[i], ys[i]) >= (ly[j], ys[j]) else -1
                    ly[i] += s * push
                    ly[j] -= s * push
                    moved = True
            for k in range(len(ly)):  # jangan menimpa marker titik lain
                if k != i and x0[i] - 0.4 < xs[k] < x1[i] and abs(ly[i] - ys[k]) < h * 0.8:
                    ly[i] += (h * 0.8 - abs(ly[i] - ys[k]) + 1e-4) * (1 if ly[i] >= ys[k] else -1)
                    moved = True
        if not moved:
            break
    return ly


def fig_tradeoff(ov):
    # waktu inferensi diukur pada test full (548 citra, lebih stabil); dipakai untuk kedua panel
    ms = ov[ov.test_set == "full"].set_index(["model", "varian", "augmentasi"]).ms_per_img
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, ts in zip(axes, ["full", "clean"]):
        d = ov[ov.test_set == ts].copy()
        d["ms"] = [ms.loc[k] for k in d[["model", "varian", "augmentasi"]].itertuples(index=False, name=None)]
        xs, ys = d.ms.to_numpy(), d.m5095.to_numpy()
        f = pareto(xs, ys)
        ax.step(xs[f], ys[f], where="post", color=INK2, linewidth=1, linestyle="--", zorder=1)
        for (model, aug), glabel, color, marker in GROUPS:
            g = d[(d.model == model) & (d.augmentasi == aug)]
            ax.scatter(g.ms, g.m5095, s=60, color=color, marker=marker, edgecolor="white", linewidth=1.2,
                       zorder=3, label=glabel)
        ly = repel_labels(xs, ys, list(d.nama))
        for i in range(len(xs)):
            moved = abs(ly[i] - ys[i]) > 0.002
            ax.annotate(d.nama.iloc[i], (xs[i], ys[i]), xytext=(xs[i] + 0.6, ly[i]), textcoords="data",
                        fontsize=7.5, va="center", color=INK, fontweight="bold" if i in f else "normal",
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5, shrinkA=0, shrinkB=4)
                        if moved else None)
        ax.set_title(f"Test {ts}")
        ax.set_xlabel("Waktu inferensi (ms/citra)")
    axes[0].set_ylabel("mAP@50-95")
    lo = ov.m5095.min() - 0.02
    hi = ov.m5095.max() + 0.02
    for ax in axes:
        ax.set_ylim(lo, hi)
        ax.set_xlim(right=ms.max() * 1.18)
    handles = [Line2D([], [], color=g[2], marker=g[3], linestyle="", markersize=8, label=g[1]) for g in GROUPS]
    handles.append(Line2D([], [], color=INK2, linestyle="--", label="Batas Pareto (tebal = di batas)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Trade-off akurasi vs kecepatan inferensi", y=1.09, fontsize=12, fontweight="bold")
    fig.tight_layout()
    footnote(fig, "Waktu = inference saja (tanpa pre/post-processing), batch 16, imgsz 640, GPU RTX 5070, "
                  "diukur pada test full. Model di kiri-atas lebih baik.\nmAP@50-95 3 kelas.")
    save(fig, "03_tradeoff_akurasi_vs_waktu.png")


# ---------------------------------------------------------------- 4. kurva training
def fig_training(runs):
    panels = [("train/box_loss", "Box loss (train)"), ("train/cls_loss", "Cls loss (train)"),
              ("metrics/mAP50(B)", "mAP@50 (validasi)"), ("metrics/mAP50-95(B)", "mAP@50-95 (validasi)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    found = []
    for rec in runs:
        if rec["model"] != "YOLOv12":
            continue
        f = rec["best_pt"].parent.parent / "results.csv"
        if not f.exists():
            print(f"  [SKIP] {f} tidak ada")
            continue
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        color = SLOT[VARIANTS.index(rec["varian"])]
        ls = "-" if rec["augmentasi"] == "ya" else "--"
        for ax, (col, _) in zip(axes.flat, panels):
            if col in df:
                ax.plot(df.epoch, df[col], color=color, linestyle=ls, linewidth=1.4)
        found.append(rec)
    for ax, (_, title) in zip(axes.flat, panels):
        ax.set_title(title)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    handles = [Line2D([], [], color=SLOT[i], linewidth=2, label=f"YOLOv12{v}") for i, v in enumerate(VARIANTS)
               if any(r["varian"] == v for r in found)]
    handles += [Line2D([], [], color=INK2, linestyle="-", label="augmentasi"),
                Line2D([], [], color=INK2, linestyle="--", label="non-augmentasi")]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Kurva training YOLOv12", y=1.07, fontsize=12, fontweight="bold")
    fig.tight_layout()
    footnote(fig, "Loss train augmentasi dihitung pada data train yang berbeda (diaugmentasi), sehingga nilainya "
                  "tidak sebanding langsung dengan non-augmentasi.\nmAP validasi diukur pada split valid "
                  "masing-masing dataset, 4 kelas.")
    save(fig, "04_kurva_training_yolov12.png")


# ---------------------------------------------------------------- 5. kurva PR
def pr_curve(gts, preds, c):
    conf, tp, n_gt = [], [], 0
    for g, p in zip(gts, preds):
        t, _ = eval02.ap_records(g, p)
        k = p[:, 0] == c
        conf.append(p[k, 5])
        tp.append(t[k])
        n_gt += int((g[:, 0] == c).sum())
    conf, tp = np.concatenate(conf), np.concatenate(tp)
    if n_gt == 0:
        return None
    o = np.argsort(-conf, kind="stable")
    tpc, fpc = np.cumsum(tp[o]), np.cumsum(~tp[o])
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-16)
    env = np.flip(np.maximum.accumulate(np.flip(prec)))  # presisi terinterpolasi (seperti AP)
    return rec, env, eval02.ap50(conf, tp, n_gt), n_gt


def fig_pr(runs, sets, names, ts):
    imgs = sets[ts][0]
    pair = [r for (m, a) in [("YOLOv12", "ya"), ("YOLOv8", "ya")] for r in runs
            if r["model"] == m and r["augmentasi"] == a and r["varian"] == FOCUS]
    if len(pair) < 2:
        print(f"  [SKIP] pasangan YOLOv12{FOCUS}/YOLOv8{FOCUS} aug tidak lengkap")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, rec in zip(axes, pair):
        preds, gts = eval02.predict(rec, ts, imgs)
        grid = np.linspace(0, 1, 101)
        curves, aps = [], []
        for c, n in enumerate(names):
            r = pr_curve(gts, preds, c)
            if r is None:
                continue
            rc, pc, ap, n_gt = r
            ax.plot(rc, pc, color=CLASS_COLOR.get(n, INK2), linewidth=1.6,
                    label=f"{NAMES_ID.get(n, n)} (n={n_gt}) AP50 {ap:.3f}")
            curves.append(np.interp(grid, np.r_[0, rc], np.r_[pc[0] if len(pc) else 0, pc], right=0))
            aps.append(ap)
        ax.plot(grid, np.mean(curves, 0), color=INK, linewidth=2.4,
                label=f"Semua kelas  mAP50 {np.mean(aps):.3f}")
        ax.set_title(f"{label(rec['model'], rec['varian'], rec['augmentasi'])}")
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=8)
    axes[0].set_ylabel("Presisi")
    fig.suptitle(f"Kurva presisi-recall per kelas, test {ts} ({len(imgs)} citra)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    footnote(fig, "IoU ≥ 0.5, conf ≥ 0.001 (sama dengan evaluasi 01). Kurva = presisi terinterpolasi "
                  "(maksimum presisi pada recall ≥ r), luas di bawahnya = AP50.")
    save(fig, f"05_kurva_pr_v12{FOCUS}_vs_v8{FOCUS}_{ts}.png")


# ---------------------------------------------------------------- 6-7. contoh deteksi
def source(name):
    if name[0].isdigit():
        return "TACO"
    if name.lower().startswith("batch"):
        return "UAVVaste"
    return "Lapangan"


def draw(ax, img_path, gt, pr, names, max_labels=99):
    im = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    ax.imshow(im)
    for c, x1, y1, x2, y2 in gt:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=GT_COLOR, linewidth=2.2))
    for c, x1, y1, x2, y2, conf in pr:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=PRED_COLOR,
                                   linewidth=1.3, linestyle=(0, (4, 2))))
        if len(pr) > max_labels:
            continue
        n = names[int(c)]
        ax.text(x1, y1 - 3, f"{NAMES_ID.get(n, n)} {conf:.2f}", fontsize=5.5, color="white", va="bottom",
                bbox=dict(facecolor=PRED_COLOR, edgecolor="none", pad=0.6, alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def detection_legend(fig, y):
    fig.legend(handles=[Line2D([], [], color=GT_COLOR, linewidth=2.2, label="Ground truth (hijau, garis utuh)"),
                        Line2D([], [], color=PRED_COLOR, linewidth=1.3, linestyle=(0, (4, 2)),
                               label=f"Prediksi (merah, garis putus, conf ≥ {CM_CONF})")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, y))


def per_image(gts, preds, nc):
    rows = []
    for i, (g, p) in enumerate(zip(gts, preds)):
        cm, hit = eval02.cm_image(g, p, nc)
        d = np.diag(cm)[:nc].sum()
        fn = cm[:nc, :].sum() - d
        fp = cm[:, :nc].sum() - d
        rows.append(dict(i=i, n_gt=len(g), tp=int(d), fn=int(fn), fp=int(fp),
                         fn_miss=int(cm[:nc, nc].sum()), fp_bg=int(cm[nc, :nc].sum()), cm=cm, hit=hit))
    return pd.DataFrame(rows)


def focus_run(runs):
    r = [r for r in runs if r["model"] == "YOLOv12" and r["augmentasi"] == "ya" and r["varian"] == FOCUS]
    return r[0] if r else None


def fig_correct(runs, sets, names):
    rec = focus_run(runs)
    if rec is None:
        print(f"  [SKIP] YOLOv12{FOCUS} aug tidak ada")
        return
    imgs = sets["clean"][0]
    preds, gts = eval02.predict(rec, "clean", imgs)
    df = per_image(gts, preds, len(names))
    ok = df[(df.n_gt > 0) & (df.fn == 0) & (df.fp == 0)].copy()
    ok["src"] = [source(imgs[i].name) for i in ok.i]
    ok["kelas"] = [frozenset(gts[i][:, 0].astype(int)) for i in ok.i]
    ok = ok.sample(frac=1, random_state=SEED).sort_values("n_gt", ascending=False, kind="stable")
    # round-robin antar sumber data agar contoh beragam, objek terbanyak lebih dulu
    by_src = {s: list(g.i) for s, g in ok.groupby("src", sort=False)}
    pick = []
    while len(pick) < 9 and any(by_src.values()):
        for s in list(by_src):
            if by_src[s] and len(pick) < 9:
                pick.append(by_src[s].pop(0))
    print(f"  {len(ok)} citra clean terdeteksi sempurna; dipilih {len(pick)}")
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 10.2))
    for ax, i in zip(axes.flat, pick):
        p = preds[i][preds[i][:, 5] >= CM_CONF]
        draw(ax, imgs[i], gts[i], p, names)
        ax.set_title(f"{source(imgs[i].name)} · {len(gts[i])} objek", fontsize=9)
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    detection_legend(fig, 1.005)
    fig.suptitle(f"Contoh deteksi benar {label(rec['model'], rec['varian'], rec['augmentasi'])} pada test clean",
                 y=1.035, fontsize=12, fontweight="bold")
    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    footnote(fig, "Dipilih dari citra tanpa FN dan FP (IoU > 0.5, kelas benar), diurutkan dari jumlah objek "
                  "terbanyak dan diambil bergiliran per sumber data.")
    save(fig, "06_contoh_deteksi_benar.png")


def guess_causes(img_path, gt, hit, cm, names):
    """Heuristik sederhana dugaan penyebab. Hanya untuk objek yang gagal (FN); bukan diagnosis pasti."""
    nc = len(names)
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(float)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    green = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 40)
    miss = gt[hit == 0]
    out = []
    if len(miss):
        areas = (miss[:, 3] - miss[:, 1]) * (miss[:, 4] - miss[:, 2])
        veg, dark, camo = [], [], []
        for c, x1, y1, x2, y2 in miss:
            a, b, c2, d = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, W)), int(min(y2, H))
            if c2 <= a or d <= b:
                continue
            veg.append(green[b:d, a:c2].mean())
            dark.append(gray[b:d, a:c2].mean())
            pw, ph = (c2 - a) // 2 + 4, (d - b) // 2 + 4
            A, B, C, D = max(a - pw, 0), max(b - ph, 0), min(c2 + pw, W), min(d + ph, H)
            ring = np.ones((D - B, C - A), bool)
            ring[b - B:d - B, a - A:c2 - A] = False
            inner = lab[b:d, a:c2].reshape(-1, 3).mean(0)
            outer = lab[B:D, A:C][ring].mean(0) if ring.any() else inner
            camo.append(np.linalg.norm(inner - outer))
        if np.median(areas) < 32 ** 2:
            out.append("objek kecil")
        if veg and np.mean(veg) > 0.25:
            out.append("oklusi vegetasi")
        if (gray > 240).mean() > 0.08 and dark and np.mean(dark) < gray.mean() - 30:
            out.append("backlight")
        if camo and np.median(camo) < 12:
            out.append("warna menyerupai latar")
    wrong = cm[:nc, :nc].copy()
    np.fill_diagonal(wrong, 0)
    if wrong.sum() and wrong.sum() >= cm[:nc, nc].sum():
        t, p = np.unravel_index(wrong.argmax(), wrong.shape)
        out.append(f"salah kelas ({NAMES_ID.get(names[t], names[t])} → {NAMES_ID.get(names[p], names[p])})")
    if cm[nc, :nc].sum() > max(cm[:nc, :].sum() - np.trace(cm[:nc, :nc]), 0):
        out.append("banyak deteksi palsu di latar")
    if len(gt) > 5:
        out.append("objek padat")
    return out[:3] or ["tidak ada pola jelas"]


def fig_failures(runs, sets, names):
    rec = focus_run(runs)
    if rec is None:
        return
    imgs = sets["clean"][0]
    preds, gts = eval02.predict(rec, "clean", imgs)
    df = per_image(gts, preds, len(names))
    df["err"] = df.fn + df.fp
    worst = df.sort_values(["err", "fn"], ascending=False, kind="stable").head(6)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8.4))
    for ax, (_, r) in zip(axes.flat, worst.iterrows()):
        i = r.i
        p = preds[i][preds[i][:, 5] >= CM_CONF]
        draw(ax, imgs[i], gts[i], p, names, max_labels=8)
        causes = guess_causes(imgs[i], gts[i], r.hit, r.cm, names)
        ax.set_title(f"{source(imgs[i].name)} · GT {r.n_gt} · TP {r.tp} · FN {r.fn} · FP {r.fp}", fontsize=9)
        ax.text(0.5, -0.03, textwrap.fill("Dugaan: " + "; ".join(causes), 46), transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color=INK)
    for ax in axes.flat[len(worst):]:
        ax.axis("off")
    fig.suptitle(f"Kegagalan terparah {label(rec['model'], rec['varian'], rec['augmentasi'])} pada test clean",
                 y=1.0, fontsize=12, fontweight="bold")
    detection_legend(fig, 0.975)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.08, wspace=0.06, hspace=0.32)
    footnote(fig, "Diurutkan dari FN + FP terbanyak (conf ≥ 0.25, IoU > 0.5). Dugaan penyebab dari heuristik "
                  "otomatis pada objek yang gagal: kecil = median luas < 32² px;\nvegetasi = > 25% piksel hijau "
                  "di dalam box; backlight = > 8% piksel sangat terang dan objek ≥ 30 level lebih gelap dari "
                  "rata-rata citra;\nwarna menyerupai latar = ΔE (Lab) box vs sekelilingnya < 12. Perlu "
                  "dikonfirmasi secara visual. Label kelas disembunyikan bila > 8 prediksi dalam satu citra.")
    save(fig, "07_contoh_kegagalan.png")


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    import yaml

    names = yaml.safe_load((ROOT / "TA-1" / "data.yaml").read_text())["names"]
    runs, _ = eval01.inventory()
    runs = [r for r in runs if r["best_pt"].exists()]
    ov = load_overall()
    sets = {s: eval02.load_test_set(s) for s in ("full", "clean")}

    jobs = [("1. mAP per varian", lambda: fig_map(ov)),
            ("2. selisih full vs clean", lambda: fig_delta(ov)),
            ("3. trade-off akurasi vs waktu", lambda: fig_tradeoff(ov)),
            ("4. kurva training YOLOv12", lambda: fig_training(runs)),
            ("5. kurva PR (clean)", lambda: fig_pr(runs, sets, names, "clean")),
            ("5. kurva PR (full)", lambda: fig_pr(runs, sets, names, "full")),
            ("6. contoh deteksi benar", lambda: fig_correct(runs, sets, names)),
            ("7. contoh kegagalan", lambda: fig_failures(runs, sets, names))]
    failed = []
    for title, job in jobs:
        print(title, flush=True)
        try:
            job()
        except Exception as e:
            traceback.print_exc()
            failed.append((title, f"{type(e).__name__}: {e}"))
            plt.close("all")
    print(f"Selesai dalam {(time.time() - t0) / 60:.1f} menit. Figure di {FIG.relative_to(ROOT)}")
    for title, msg in failed:
        print(f"  [GAGAL] {title}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

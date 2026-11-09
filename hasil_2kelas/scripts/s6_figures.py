"""LANGKAH 6: salinan results.csv, kurva training, akurasi vs kecepatan, 8 contoh prediksi. PNG 300 dpi, lebar satu kolom IEEE."""
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from hk_common import COL_W, FAMILIES, MODELS, NAMES, OUT, PAPER, RUNS, SCALES, best_epoch, family, label, paper_style, read_results, size_bin, source_of  # noqa: E402
from s5_error import best_models, load_test, match_image  # noqa: E402

# Warna mengikuti entitas, sama dengan gambar lain di project; bentuk marker sebagai pembeda kedua untuk cetak hitam putih
FAM_STYLE = {"YOLOv8": ("#eb6834", "s"), "YOLOv12": ("#2a78d6", "o"), "YOLO26": ("#1baf7a", "^")}
C_CLS = ["#1baf7a", "#eda100"]  # lightweight litter, metal and glass
C_ERR = "#e34948"
MISS_FX = [pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()]
CATS = [("correct", "correct, both classes"), ("wrong_class", "wrong class"), ("small_missed", "small object missed"),
        ("false_positive", "false positive")]


def fig_curves(best, path):
    fig, axs = plt.subplots(3, 1, figsize=(COL_W, COL_W * 1.45), sharex=True)
    panels = [("Box loss", "train/box_loss", "val/box_loss"), ("Cls loss", "train/cls_loss", "val/cls_loss"),
              ("Val mAP@50-95", "metrics/mAP50-95(B)", None)]
    for ax, (title, k1, k2) in zip(axs, panels):
        for f, m in best.items():
            d = read_results(RUNS / f"{m}_2kelas" / "results.csv")
            c = FAM_STYLE[f][0]
            ax.plot(d["epoch"], d[k1], color=c, lw=1.0)
            if k2:
                ax.plot(d["epoch"], d[k2], color=c, lw=1.0, ls=(0, (3, 1.5)))
            else:
                be = best_epoch(d)
                ax.plot(be, d.loc[d.epoch == be, k1].iloc[0], marker=FAM_STYLE[f][1], ms=5, color=c, mec="white", mew=0.6, zorder=4)
        ax.set_ylabel(title)
        ax.grid(True)
        ax.set_axisbelow(True)
    axs[-1].set_xlabel("Epoch")
    h = [Line2D([], [], color=FAM_STYLE[f][0], lw=1.2, marker=FAM_STYLE[f][1], ms=4, label=label(m)) for f, m in best.items()]
    h += [Line2D([], [], color="#52514e", lw=1.0, label="train"), Line2D([], [], color="#52514e", lw=1.0, ls=(0, (3, 1.5)), label="val")]
    axs[0].legend(handles=h, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.8)
    fig.align_ylabels(axs)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_acc_speed(path):
    T = pd.read_csv(OUT / "metrics_test.csv")
    S = pd.read_csv(OUT / "speed.csv")
    S = S[S.conf == 0.001]
    D = T.merge(S[["model", "total"]], on="model")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.74))
    offsets = {("YOLOv8", "n"): (5, -2), ("YOLOv8", "s"): (-8, 2), ("YOLOv8", "m"): (-3, 5), ("YOLOv8", "l"): (-1, 5), ("YOLOv8", "x"): (5, -3),
               ("YOLOv12", "n"): (5, -2), ("YOLOv12", "s"): (-1, -10), ("YOLOv12", "m"): (4, -9), ("YOLOv12", "l"): (-9, -2), ("YOLOv12", "x"): (-9, -6),
               ("YOLO26", "n"): (5, -2), ("YOLO26", "s"): (5, -5), ("YOLO26", "m"): (-8, 4), ("YOLO26", "l"): (-2, 5), ("YOLO26", "x"): (5, 0)}
    for f, (c, mk) in FAM_STYLE.items():
        t = D[D.family == f].set_index("skala").loc[SCALES]
        ax.plot(t["total"], t["mAP50_95"] * 100, color=c, lw=1.0, marker=mk, ms=5, mec="white", mew=0.6, label=f, zorder=3)
        for s, r in t.iterrows():
            ax.annotate(s, (r["total"], r["mAP50_95"] * 100), xytext=offsets.get((f, s), (3, 3)), textcoords="offset points",
                        fontsize=8, color="#0b0b0b")
    ax.set_xlim(5, 23)
    ax.set_xlabel("Total latency (ms/image, batch 1)")
    ax.set_ylabel("Test mAP@50-95 (%)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def pick_samples(best):
    per = {}
    files = None
    for f, m in best.items():
        files, imgs = load_test(m)
        per[f] = (imgs, [match_image(im) for im in imgs])
    overlap = pd.read_csv(OUT / "tumpang_tindih_train.csv")
    stats = pd.read_csv(OUT / "dataset_stats.csv")
    thr = int(stats[stats.item == "ambang_inlier_ORB_maks_kontrol_acak"].nilai.iloc[0])
    leak = set(overlap[(overlap.split == "test") & (overlap.inlier_nama_sama > thr)].file)
    rows = []
    for i, fname in enumerate(files):
        im0 = per["YOLOv8"][0][i]
        gc = im0["gt_cls"]
        wh = im0["gt_box"][:, 2:] - im0["gt_box"][:, :2]
        small = np.array([size_bin(w * h) == "small" for w, h in wh], bool)
        r = dict(i=i, file=fname, sumber=source_of(fname), n_gt=len(gc), n_kelas=len(set(gc.tolist())), overlap=fname in leak)
        for f in FAM_STYLE:
            gs, ps, keep = per[f][1][i]
            conf = per[f][0][i]["pr_conf"][keep]
            r[f"tp_{f}"] = int((gs == 1).sum())
            r[f"wrong_{f}"] = int((gs == 2).sum())
            r[f"miss_{f}"] = int((gs == 0).sum())
            r[f"miss_small_{f}"] = int(((gs == 0) & small).sum())
            r[f"fp_{f}"] = int((ps == 0).sum())
            r[f"fpconf_{f}"] = float(conf[ps == 0].max()) if (ps == 0).any() else 0.0
        rows.append(r)
    S = pd.DataFrame(rows)
    fams = list(FAM_STYLE)
    S["n_all_ok"] = sum(((S[f"wrong_{f}"] == 0) & (S[f"miss_{f}"] == 0) & (S[f"fp_{f}"] == 0)).astype(int) for f in fams)
    S["n_wrong"] = sum((S[f"wrong_{f}"] > 0).astype(int) for f in fams)
    S["n_small"] = sum((S[f"miss_small_{f}"] > 0).astype(int) for f in fams)
    S["n_fp"] = sum((S[f"fp_{f}"] > 0).astype(int) for f in fams)
    S["sum_wrong"] = sum(S[f"wrong_{f}"] for f in fams)
    S["sum_small"] = sum(S[f"miss_small_{f}"] for f in fams)
    S["max_fpconf"] = S[[f"fpconf_{f}" for f in fams]].min(axis=1)
    pools = {
        "correct": S[(S.n_all_ok == 3) & (S.n_gt >= 1)].sort_values(["n_kelas", "n_gt"], ascending=False),
        "wrong_class": S[(S.n_wrong >= 2) & (S.n_gt <= 6)].sort_values(["n_wrong", "sum_wrong", "n_gt"], ascending=[False, False, True]),
        "small_missed": S[(S.n_small == 3) & (S.n_gt <= 10)].sort_values(["sum_small", "n_gt"], ascending=[False, True]),
        "false_positive": S[(S.n_fp == 3) & (S.n_gt == 0)].sort_values("max_fpconf", ascending=False),
    }
    chosen, used = [], set()
    for cat, _ in CATS:
        pool = pools[cat]
        print(cat, "kandidat:", len(pool), "bukan tumpang tindih:", int((~pool.overlap).sum()))
        taken = 0
        src_taken = set()
        # Utamakan gambar yang sumbernya tidak ada di train, lalu sumber yang berbeda dalam satu kategori
        for sub, only_new_src in ((pool[~pool.overlap], True), (pool[~pool.overlap], False), (pool, True), (pool, False)):
            for _, r in sub.iterrows():
                if taken == 2:
                    break
                if r.i in used or (only_new_src and r.sumber in src_taken):
                    continue
                chosen.append((cat, r))
                used.add(r.i)
                src_taken.add(r.sumber)
                taken += 1
        if taken < 2:
            print(f"[PERINGATAN] kategori {cat} hanya dapat {taken} gambar")
    return per, chosen


def crop_window(cat, im, per, i, W, H):
    """Jendela persegi di sekitar objek penting supaya objek kecil masih terlihat di lebar satu kolom."""
    boxes = [im["gt_box_ori"]] if len(im["gt_cls"]) else []
    if cat == "false_positive":
        for f in FAM_STYLE:
            imf, m = per[f][0][i], per[f][1][i]
            gs, ps, keep = m
            boxes.append(imf["pr_box_ori"][keep][ps == 0])
    if not boxes or not sum(len(b) for b in boxes):
        return 0, 0, W, H
    b = np.concatenate(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(max(x2 - x1, y2 - y1) + 60, 200), min(W, H))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0, y0 = np.clip(cx - side / 2, 0, W - side), np.clip(cy - side / 2, 0, H - side)
    return x0, y0, x0 + side, y0 + side


def draw_panel(ax, img, win, gt=None, pred=None):
    ax.imshow(img)
    ax.set_xlim(win[0], win[2])
    ax.set_ylim(win[3], win[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if gt is not None:
        for (x1, y1, x2, y2), c in zip(*gt):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec=C_CLS[c]))
    if pred is not None:
        im, (gs, ps, keep) = pred
        for k, s in zip(keep, ps):
            x1, y1, x2, y2 = im["pr_box_ori"][k]
            col = C_CLS[im["pr_cls"][k]] if s == 1 else C_ERR
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec=col))
        for (x1, y1, x2, y2), s in zip(im["gt_box_ori"], gs):
            if s == 0:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=0.9, ec="white", path_effects=MISS_FX))


def legend_handles():
    return [Line2D([], [], color=C_CLS[0], lw=1.4, label=PAPER[NAMES[0]]), Line2D([], [], color=C_CLS[1], lw=1.4, label=PAPER[NAMES[1]]),
            Line2D([], [], color=C_ERR, lw=1.4, label="FP / wrong class"),
            Line2D([], [], color="white", lw=1.4, path_effects=MISS_FX, label="missed ground truth")]


def fig_samples(best, per, chosen, outdir):
    fams = list(FAM_STYLE)
    n = len(chosen)
    cols = ["Ground truth"] + [label(best[f]) for f in fams]
    fig, axs = plt.subplots(n, 4, figsize=(COL_W, COL_W / 4 * n + 0.42))
    list_rows = []
    letters = "abcdefgh"
    for row, (cat, r) in enumerate(chosen):
        i = int(r.i)
        im0 = per[fams[0]][0][i]
        img = Image.open(im0["im_file"]).convert("RGB")
        win = crop_window(cat, im0, per, i, *img.size)
        panels = [dict(gt=(im0["gt_box_ori"], im0["gt_cls"]))] + [dict(pred=(per[f][0][i], per[f][1][i])) for f in fams]
        for col, kw in enumerate(panels):
            draw_panel(axs[row, col], img, win, **kw)
            if row == 0:
                axs[row, col].set_title(cols[col], fontsize=7.5, pad=2)
        axs[row, 0].set_ylabel(f"({letters[row]})", fontsize=8, rotation=0, labelpad=7, va="center", color="#0b0b0b")
        # satu PNG per gambar, untuk dipakai terpisah
        f2, a2 = plt.subplots(1, 4, figsize=(COL_W, COL_W / 4 + 0.2))
        for col, kw in enumerate(panels):
            draw_panel(a2[col], img, win, **kw)
            a2[col].set_title(cols[col], fontsize=7.5, pad=2)
        f2.tight_layout(pad=0.1, w_pad=0.2)
        f2.savefig(outdir / f"sample_{row + 1}_{cat}.png")
        plt.close(f2)
        lr = dict(no=row + 1, huruf=letters[row], kategori=cat, file=r.file, sumber=r.sumber, n_gt=r.n_gt,
                  sumber_juga_di_train=bool(r.overlap), crop=win != (0, 0, *img.size))
        for f in fams:
            lr.update({f"{label(best[f])}_{k}": int(r[f"{k}_{f}"]) for k in ("tp", "wrong", "miss", "fp")})
        list_rows.append(lr)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, frameon=False, fontsize=7, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(pad=0.1, w_pad=0.15, h_pad=0.15, rect=(0, 0.075, 1, 1))
    fig.savefig(outdir / "samples_grid.png")
    plt.close(fig)
    pd.DataFrame(list_rows).to_csv(outdir / "samples.csv", index=False)
    print(pd.DataFrame(list_rows).to_string(index=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paper_style()
    for d in ("figures", "curves", "samples"):
        (OUT / d).mkdir(exist_ok=True)
    for m in MODELS:
        shutil.copy2(RUNS / f"{m}_2kelas" / "results.csv", OUT / "curves" / f"{m}_2kelas_results.csv")
    best = best_models()
    fig_curves(best, OUT / "figures" / "fig_kurva_training_terbaik.png")
    fig_acc_speed(OUT / "figures" / "fig_akurasi_vs_kecepatan.png")
    per, chosen = pick_samples(best)
    fig_samples(best, per, chosen, OUT / "samples")
    print("[SELESAI s6]")


if __name__ == "__main__":
    main()

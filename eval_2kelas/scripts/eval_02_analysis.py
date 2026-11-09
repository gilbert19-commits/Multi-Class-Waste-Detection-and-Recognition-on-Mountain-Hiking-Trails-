"""
EVALUASI 2 KELAS — analisis (CPU), memakai cache prediksi dari eval_01_infer.py.

1. Inventaris run (results.csv, args.yaml, best.pt train_args, train_2kelas.log).
2. Tabel metrik test penuh dari Ultralytics + cek ulang mAP dari matriks TP validator.
3. Confusion matrix (conf >= 0.25, IoU > 0.5; pencocokan seperti Ultralytics ConfusionMatrix,
   fungsi greedy/cm_image sama dengan scripts/02_error_analysis.py) angka mentah + PNG.
4. Bootstrap berpasangan 1000x level citra (seed 0) YOLOv12 vs YOLOv8 per skala.
5. Analisis error (recall per sumber dan per ukuran objek pada input 512) model terbaik v12 dan v8.
6. Gambar: mAP50 vs waktu inference, kurva training, contoh prediksi.
7. hasil_eval_2kelas.md, salinan args.yaml/results.csv/README, zip eval_2kelas.zip.
"""
import json
import pickle
import re
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset-2kelas"
RUNS = ROOT / "runs_2kelas"
OUT = ROOT / "eval_2kelas"
CACHE = OUT / "_cache"
TABLES = OUT / "tables"
FIG = OUT / "figures"
CMDIR = OUT / "confusion_matrix"
SEED = 0
N_BOOT = 1000
CM_CONF, CM_IOU = 0.25, 0.5
SIZE_BINS = [("kecil (<32²)", 0, 32 ** 2), ("sedang (32²–96²)", 32 ** 2, 96 ** 2), ("besar (>96²)", 96 ** 2, np.inf)]
SOURCE_CLEAN_RE = re.compile(r"(_jpe?g\.rf\.[a-f0-9]+|_png\.rf\.[a-f0-9]+|\.rf\.[a-f0-9]+).*$", re.IGNORECASE)
SCALES = list("nsmlx")
MODELS = [f"yolov12{v}" for v in SCALES] + [f"yolov8{v}" for v in SCALES]
# palet acuan dataviz (slot kategorikal 1-2) + ink
C_V12, C_V8 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C_CLS = ["#1baf7a", "#eda100"]  # kelas 0, kelas 1 (contoh prediksi)
C_ERR = "#e34948"
SHORT = {"kertas_plastik_puntungRokok": "KPP", "logam_kaleng_kaca": "LKK"}
BLUES = LinearSegmentedColormap.from_list("ramp_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def label(m):
    return ("YOLOv12" if m.startswith("yolov12") else "YOLOv8") + m[-1]


def fam(m):
    return "YOLOv12" if m.startswith("yolov12") else "YOLOv8"


def fmt(x):
    """Presisi penuh (repr float Python) untuk tabel md."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "TIDAK TERSEDIA"
    if isinstance(x, (bool, np.bool_)):
        return "ya" if x else "tidak"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return out


def source_of(fname):
    src = SOURCE_CLEAN_RE.sub("", Path(fname).stem)
    if re.match(r"^(batch|BATCH)_", src):
        return "UAVVaste"
    if re.fullmatch(r"\d+", src):
        return "TACO"
    if re.match(r"^(WhatsApp-Video|IMG_|GOPR)", src):
        return "Burangrang"
    return "Unknown"


def paper_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
                         "grid.linewidth": 0.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


# ---------------------------------------------------------------- 1. inventaris run
def read_results(p):
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    return d


def training_seconds(d):
    """results.csv 'time' = detik kumulatif per sesi; reset ke ~0 saat resume. Jumlahkan akhir tiap sesi."""
    t = d["time"].to_numpy()
    total, prev = 0.0, 0.0
    for x in t:
        if x < prev:  # sesi baru (resume)
            total += prev
        prev = x
    return total + prev, int((np.diff(t) < 0).sum()) + 1


def inventory():
    import torch

    log = (ROOT / "train_2kelas.log").read_text(encoding="utf-8", errors="replace")
    opt_lines = sorted(set(re.findall(r"optimizer:\S*\s+(\w+\(lr=[\d.]+, momentum=[\d.]+\))", log)))
    rows = []
    for m in MODELS:
        rd = RUNS / f"{m}_2kelas"
        d = read_results(rd / "results.csv")
        a = yaml.safe_load((rd / "args.yaml").read_text())
        ck = torch.load(rd / "weights" / "best.pt", map_location="cpu", weights_only=False)
        batch = int(ck["train_args"]["batch"])
        nbs = int(a.get("nbs", 64))
        acc = max(round(nbs / batch), 1)
        secs, n_sess = training_seconds(d)
        ep_run = int(d["epoch"].iloc[-1])
        best_ep = int(d.loc[d["metrics/mAP50-95(B)"].idxmax(), "epoch"])
        rows.append(dict(model=label(m), run=rd.name, epoch_dijalankan=ep_run, epoch_terbaik=best_ep,
                         early_stopping_terpicu=ep_run < int(a["epochs"]), batch_dipakai=batch,
                         akumulasi_gradien=acc, lama_training_jam=secs / 3600, jumlah_sesi=n_sess,
                         epochs=a["epochs"], patience=a["patience"], imgsz=a["imgsz"], optimizer=a["optimizer"],
                         lr0=a["lr0"], momentum=a["momentum"], weight_decay=a["weight_decay"], seed=a["seed"],
                         workers_args_yaml=a["workers"],
                         weight_decay_efektif=a["weight_decay"] * batch * acc / nbs,
                         best_val_mAP50=float(d.loc[d["metrics/mAP50-95(B)"].idxmax(), "metrics/mAP50(B)"]),
                         best_val_mAP50_95=float(d["metrics/mAP50-95(B)"].max())))
    return pd.DataFrame(rows), opt_lines


# ---------------------------------------------------------------- matching (sama dengan scripts/02)
def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def greedy(iou, thr):
    """Seperti Ultralytics ConfusionMatrix.process_batch: IoU > thr, urut IoU, unik per prediksi lalu per GT."""
    gi, pi = np.nonzero(iou > thr)
    if not len(gi):
        return np.zeros((0, 2), int)
    m = np.stack([gi, pi], 1)
    if len(m) > 1:
        m = m[np.argsort(-iou[gi, pi], kind="stable")]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[np.argsort(-iou[m[:, 0], m[:, 1]], kind="stable")]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
    return m


def cm_image(im, nc):
    """cm[true, pred] dengan indeks nc = latar. gt_state: 1 benar, 0 terlewat, 2 salah kelas.
    pr_state (untuk prediksi conf >= CM_CONF): 1 TP, 0 FP latar, 2 FP salah kelas."""
    keep = im["pr_conf"] >= CM_CONF
    pc, pb = im["pr_cls"][keep], im["pr_box"][keep]
    gc, gb = im["gt_cls"], im["gt_box"]
    cm = np.zeros((nc + 1, nc + 1), int)
    gs = np.zeros(len(gc), int)
    ps = np.zeros(len(pc), int)
    m = greedy(iou_matrix(gb, pb), CM_IOU)
    for g, p in m:
        cm[gc[g], pc[p]] += 1
        gs[g] = ps[p] = 1 if gc[g] == pc[p] else 2
    for g in np.nonzero(gs == 0)[0]:
        cm[gc[g], nc] += 1
    for p in np.nonzero(ps == 0)[0]:
        cm[nc, pc[p]] += 1
    return cm, gs, ps, np.nonzero(keep)[0]


# ---------------------------------------------------------------- AP (sama dengan Ultralytics ap_per_class/compute_ap)
def ap_curve(tpc, fpc, n_gt):
    """tpc/fpc: (n, k) kumulatif. Return AP (k,) 101 titik interpolasi, sama dengan
    ultralytics.utils.metrics.compute_ap 8.4 (sentinel: presisi jatuh ke 0 tepat setelah recall maksimum)."""
    x = np.linspace(0, 1, 101)
    rec = tpc / (n_gt + 1e-16)
    prec = tpc / np.maximum(tpc + fpc, 1e-16)
    out = np.zeros(tpc.shape[1])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    for j in range(tpc.shape[1]):
        mrec = np.concatenate([[0.0], rec[:, j], [rec[-1, j]], [1.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1.0], prec[:, j], [0.0], [0.0]]))))
        out[j] = trapz(np.interp(x, mrec, mpre), x)
    return out


class APData:
    """Prediksi semua citra diurutkan per kelas sekali; AP bisa dihitung ulang dengan bobot citra (bootstrap)."""

    def __init__(self, imgs, nc):
        self.nc = nc
        self.ngt = np.stack([np.bincount(im["gt_cls"], minlength=nc) for im in imgs])  # (n_img, nc)
        img = np.concatenate([np.full(len(im["pr_cls"]), i) for i, im in enumerate(imgs)])
        cls = np.concatenate([im["pr_cls"] for im in imgs])
        conf = np.concatenate([im["pr_conf"] for im in imgs])
        tp = np.concatenate([im["tp"] for im in imgs]).astype(float)
        self.per_cls = []
        for c in range(nc):
            s = np.nonzero(cls == c)[0]
            o = s[np.argsort(-conf[s], kind="stable")]
            self.per_cls.append((img[o], tp[o]))

    def ap(self, w=None):
        """Return AP (nc, 10). w: bobot per citra (None = semua 1)."""
        res = np.full((self.nc, 10), np.nan)
        for c, (img, tp) in enumerate(self.per_cls):
            n_gt = self.ngt[:, c].sum() if w is None else w @ self.ngt[:, c]
            if n_gt <= 0:
                continue
            if not len(img):
                res[c] = 0.0
                continue
            wp = np.ones(len(img)) if w is None else w[img].astype(float)
            tpc = np.cumsum(tp * wp[:, None], 0)
            fpc = np.cumsum((1 - tp) * wp[:, None], 0)
            res[c] = ap_curve(tpc, fpc, n_gt)
        return res


def summarize_ap(ap):
    return dict(mAP50=np.nanmean(ap[:, 0]), mAP50_95=np.nanmean(ap.mean(1)), mAP75=np.nanmean(ap[:, 5]))


# ---------------------------------------------------------------- gambar
def plot_cm(cm, names, title, path, star):
    labels = [n.replace("_", "_\n", 1) if len(n) > 20 else n for n in names] + ["latar\n(background)"]
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    vmax = max(cm[:-1, :-1].max(), 1)
    ax.imshow(np.minimum(cm, vmax), cmap=BLUES, vmin=0, vmax=vmax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j == cm.shape[0] - 1:
                ax.text(j, i, "–", ha="center", va="center", color=INK2)
                continue
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > vmax * 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Ground truth")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title + ("\n[mAP@50 tertinggi]" if star else ""))
    fig.text(0.99, -0.07, f"conf ≥ {CM_CONF}, IoU > {CM_IOU}. Baris latar = FP, kolom latar = FN terlewat.",
             ha="right", va="top", fontsize=5.5, color=INK2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_map_vs_ms(tab, path):
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for f, c, mk in (("YOLOv12", C_V12, "o"), ("YOLOv8", C_V8, "s")):
        t = tab[tab.keluarga == f].set_index("skala").loc[SCALES]
        ax.plot(t["ms_inference"], t["mAP50"], color=c, lw=1.0, marker=mk, ms=4.2, mec="white", mew=0.6,
                label=f, zorder=3)
        for s, r in t.iterrows():
            # label di sisi yang tidak dilalui garis keluarga lain
            off = (-7, 1) if f == "YOLOv8" and s in "ns" else (4, -8) if f == "YOLOv8" else (3, 3)
            if f == "YOLOv12" and s == "n":
                off = (4, -2)
            ax.annotate(s, (r["ms_inference"], r["mAP50"]), xytext=off, textcoords="offset points",
                        fontsize=6, color=INK2)
    ax.set_xlabel("Waktu inference (ms/citra, batch 16, RTX 5070)")
    ax.set_ylabel("mAP@50 (test)")
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=600)
    plt.close(fig)


def plot_training(models, path):
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.4), sharex=True)
    panels = [("Box loss", "train/box_loss", "val/box_loss"), ("Cls loss", "train/cls_loss", "val/cls_loss"),
              ("Val mAP@50", "metrics/mAP50(B)", None), ("Val mAP@50-95", "metrics/mAP50-95(B)", None)]
    for ax, (title, k1, k2) in zip(axs.flat, panels):
        for m in models:
            d = read_results(RUNS / f"{m}_2kelas" / "results.csv")
            c = C_V12 if m.startswith("yolov12") else C_V8
            if k2:
                ax.plot(d["epoch"], d[k1], color=c, lw=1.0, label=f"{label(m)} train")
                ax.plot(d["epoch"], d[k2], color=c, lw=1.0, ls="--", label=f"{label(m)} val")
            else:
                ax.plot(d["epoch"], d[k1], color=c, lw=1.0, label=label(m))
                be = int(d.loc[d["metrics/mAP50-95(B)"].idxmax(), "epoch"])
                ax.plot(be, d.loc[d.epoch == be, k1].iloc[0], marker="o", ms=4, color=c, mec="white", mew=0.6)
        ax.set_title(title)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=2 if k2 else 1)
    for ax in axs[1]:
        ax.set_xlabel("Epoch")
    fig.text(0.99, -0.01, "Titik = epoch terbaik (fitness mAP@50-95, best.pt).", ha="right", va="top",
             fontsize=5.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def draw_pair(axL, axR, im, gs, ps, keep_idx, names, title):
    img = Image.open(im["im_file"]).convert("RGB")
    for ax in (axL, axR):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for (x1, y1, x2, y2), c, s in zip(im["gt_box_ori"], im["gt_cls"], gs):
        ok = s == 1
        axL.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0 if ok else 1.3,
                                ec=C_CLS[c] if ok else C_ERR, ls="-" if ok else "--"))
    for k, s in zip(keep_idx, ps):
        x1, y1, x2, y2 = im["pr_box_ori"][k]
        c = im["pr_cls"][k]
        col = C_CLS[c] if s == 1 else C_ERR
        axR.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec=col))
        W_ = img.size[0]
        ha = "right" if x1 > W_ * 0.8 else "left"
        axR.text(x2 if ha == "right" else x1, y1 - 2, f"{SHORT.get(names[c], c)} {im['pr_conf'][k]:.2f}",
                 fontsize=4.5, color="white", va="bottom", ha=ha, clip_on=True,
                 bbox=dict(fc=col, ec="none", pad=0.6))
    axL.set_title(f"GT — {title}", fontsize=6)
    axR.set_title("Prediksi (conf ≥ 0.25)", fontsize=6)


# ---------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for d in (TABLES, FIG, CMDIR):
        d.mkdir(parents=True, exist_ok=True)
    paper_style()
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    nc = len(names)
    raw = json.loads((TABLES / "eval_ultralytics_raw.json").read_text())
    models = [m for m in MODELS if m in raw]
    L = []  # baris md

    # ---- 1. inventaris
    inv, opt_lines = inventory()
    inv.to_csv(TABLES / "01_inventaris_run.csv", index=False)

    # ---- data per citra
    caches = {m: pickle.loads((CACHE / f"{m}_2kelas.pkl").read_bytes()) for m in models}
    files = sorted(caches[models[0]]["images"])
    for m in models:
        if sorted(caches[m]["images"]) != files:
            raise RuntimeError(f"himpunan citra {m} berbeda")
    test_imgs = sorted(p.name for p in (DATA_DIR / "test" / "images").iterdir())
    dropped = sorted(set(test_imgs) - set(files))
    from ultralytics.data.utils import load_dataset_cache_file

    lc = load_dataset_cache_file(OUT / "_testset" / "labels.cache")
    corrupt_msgs = [x for x in lc.get("msgs", []) if "corrupt" in x.lower()]
    imgs = {m: [caches[m]["images"][f] for f in files] for m in models}
    n_img = len(files)
    src = np.array([source_of(f) for f in files])

    # ---- 2. metrik + cek ulang
    rows, cls_rows, check = [], [], []
    apd = {}
    for m in models:
        r = raw[m]
        sp = pd.DataFrame(r["speeds"])
        med = sp.median()
        tot = med["preprocess"] + med["inference"] + med["postprocess"]
        rows.append(dict(model=label(m), keluarga=fam(m), skala=m[-1], params=r["params"],
                         params_M=r["params"] / 1e6, GFLOPs_512=r["GFLOPs_512"],
                         mAP50=r["mAP50"], mAP50_95=r["mAP50_95"], mAP75=r["mAP75"], precision=r["precision"],
                         recall=r["recall"], ms_preprocess=med["preprocess"], ms_inference=med["inference"],
                         ms_postprocess=med["postprocess"], ms_total=tot, FPS_inference=1000 / med["inference"],
                         FPS_total=1000 / tot, ms_inference_min=sp["inference"].min(),
                         ms_inference_max=sp["inference"].max(), peak_mem_alloc_MiB=r["peak_mem_alloc_MiB"],
                         peak_mem_reserved_MiB=r["peak_mem_reserved_MiB"], n_citra=r["n_images_scored"]))
        for cname, c in r["per_class"].items():
            cls_rows.append(dict(model=label(m), kelas=cname, citra=c["images"], instance=c["instances"],
                                 precision=c["precision"], recall=c["recall"], AP50=c["AP50"],
                                 AP50_95=c["AP50_95"], AP75=c["AP75"]))
        apd[m] = APData(imgs[m], nc)
        own = summarize_ap(apd[m].ap())
        check.append(dict(model=label(m), **{f"{k}_ultralytics": r[k] for k in own},
                          **{f"{k}_hitung_ulang": v for k, v in own.items()},
                          max_selisih_abs=max(abs(r[k] - v) for k, v in own.items())))
    ov = pd.DataFrame(rows)
    pc = pd.DataFrame(cls_rows)
    ck = pd.DataFrame(check)
    ov.to_csv(TABLES / "02_metrik_test_penuh.csv", index=False)
    pc.to_csv(TABLES / "02_metrik_per_kelas_test_penuh.csv", index=False)
    ck.to_csv(TABLES / "02_cek_ulang_mAP.csv", index=False)
    best_m50 = ov.loc[ov.mAP50.idxmax(), "model"]
    best_fam = {f: ov[ov.keluarga == f].sort_values(["mAP50", "mAP50_95"], ascending=False).iloc[0]["model"]
                for f in ("YOLOv12", "YOLOv8")}
    lab2m = {label(m): m for m in models}

    # ---- 3. confusion matrix + TP/FP/FN per citra
    per = {}
    cm_rows, err_rows = [], []
    for m in models:
        cms, gss, pss, kis = [], [], [], []
        for im in imgs[m]:
            cm_i, gs, ps, ki = cm_image(im, nc)
            cms.append(cm_i), gss.append(gs), pss.append(ps), kis.append(ki)
        cms = np.stack(cms)  # (n_img, nc+1, nc+1)
        per[m] = dict(cm=cms, gs=gss, ps=pss, ki=kis)
        cm = cms.sum(0)
        lab = label(m)
        cmdf = pd.DataFrame(cm, index=[f"GT {n}" for n in names] + ["GT latar"],
                            columns=[f"pred {n}" for n in names] + ["pred latar"])
        cmdf.to_csv(CMDIR / f"cm_{m}_test_penuh.csv")
        plot_cm(cm, names, f"{lab} — test penuh ({n_img} citra)", CMDIR / f"cm_{m}_test_penuh.png",
                star=lab == best_m50)
        for c, cname in enumerate(names):
            tp = cm[c, c]
            fn_miss, fp_bg = cm[c, nc], cm[nc, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            err_rows.append(dict(model=lab, kelas=cname, GT=int(cm[c, :].sum()), TP=int(tp), FN=int(fn),
                                 FN_terlewat=int(fn_miss), FN_salah_kelas=int(fn - fn_miss), FP=int(fp),
                                 FP_latar=int(fp_bg), FP_salah_kelas=int(fp - fp_bg),
                                 precision=tp / (tp + fp) if tp + fp else np.nan,
                                 recall=tp / (tp + fn) if tp + fn else np.nan))
        cm_rows.append(dict(model=lab, **{f"{a}->{b}": int(cm[i, j])
                                          for i, a in enumerate(list(names) + ["latar"])
                                          for j, b in enumerate(list(names) + ["latar"])}))
    errs = pd.DataFrame(err_rows)
    errs.to_csv(TABLES / "03_tp_fp_fn_per_kelas_conf025.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(TABLES / "03_confusion_matrix_semua_model.csv", index=False)

    # ---- 4. bootstrap
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_img, size=(N_BOOT, n_img))
    W = np.stack([np.bincount(ix, minlength=n_img) for ix in idx]).astype(float)
    boot_ap = {}
    for m in models:
        print(f"[bootstrap AP] {label(m)}", flush=True)
        boot_ap[m] = np.stack([apd[m].ap(w) for w in W])  # (N_BOOT, nc, 10)

    def comps(m):
        cms = per[m]["cm"]
        out = {}
        for c, cname in enumerate(names):
            tp = cms[:, c, c]
            out[(cname, "FN")] = cms[:, c, :].sum(1) - tp
            out[(cname, "FN_terlewat")] = cms[:, c, nc]
            out[(cname, "FN_salah_kelas")] = out[(cname, "FN")] - cms[:, c, nc]
            out[(cname, "FP")] = cms[:, :, c].sum(1) - tp
            out[(cname, "FP_latar")] = cms[:, nc, c]
            out[(cname, "FP_salah_kelas")] = out[(cname, "FP")] - cms[:, nc, c]
        for k in ("FN", "FN_terlewat", "FN_salah_kelas", "FP", "FP_latar", "FP_salah_kelas"):
            out[("TOTAL", k)] = sum(out[(n, k)] for n in names)
        return out

    def verdict(lo, hi, lower_better):
        if lo <= 0 <= hi:
            return "setara (CI memuat 0)"
        better = hi < 0 if lower_better else lo > 0
        return "YOLOv12 lebih baik" if better else "YOLOv12 lebih buruk"

    boot_rows = []
    for s in SCALES:
        a, b = f"yolov12{s}", f"yolov8{s}"
        if a not in models or b not in models:
            continue
        for metric, fn in (("mAP50", lambda x: np.nanmean(x[..., 0], -1)),
                           ("mAP50_95", lambda x: np.nanmean(x.mean(-1), -1))):
            va, vb = fn(apd[a].ap()), fn(apd[b].ap())
            dist = fn(boot_ap[a]) - fn(boot_ap[b])
            lo, hi = np.percentile(dist, [2.5, 97.5])
            boot_rows.append(dict(skala=s, kelas="semua", metrik=metric, YOLOv12=float(va), YOLOv8=float(vb),
                                  delta=float(va - vb), ci_lo=lo, ci_hi=hi, kesimpulan=verdict(lo, hi, False)))
        ca, cb = comps(a), comps(b)
        for (cname, k) in ca:
            xa, xb = ca[(cname, k)], cb[(cname, k)]
            dist = W @ (xa - xb)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            boot_rows.append(dict(skala=s, kelas=cname, metrik=k, YOLOv12=int(xa.sum()), YOLOv8=int(xb.sum()),
                                  delta=int(xa.sum() - xb.sum()), ci_lo=lo, ci_hi=hi,
                                  kesimpulan=verdict(lo, hi, True)))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(TABLES / "04_bootstrap_v12_vs_v8.csv", index=False)

    # ---- 5. analisis error model terbaik per keluarga
    src_rows, size_rows, size_cls_rows = [], [], []
    for f, lab in best_fam.items():
        m = lab2m[lab]
        recs = []
        for i, im in enumerate(imgs[m]):
            wh = im["gt_box"][:, 2:] - im["gt_box"][:, :2]
            for g in range(len(im["gt_cls"])):
                recs.append(dict(sumber=src[i], kelas=names[im["gt_cls"][g]], area=wh[g, 0] * wh[g, 1],
                                 state=per[m]["gs"][i][g]))
        R = pd.DataFrame(recs)
        R["ukuran"] = pd.cut(R["area"], [b[1] for b in SIZE_BINS] + [np.inf], right=False,
                             labels=[b[0] for b in SIZE_BINS])

        def agg(g):
            n = len(g)
            return pd.Series(dict(GT=n, TP=int((g.state == 1).sum()), FN_terlewat=int((g.state == 0).sum()),
                                  FN_salah_kelas=int((g.state == 2).sum()),
                                  recall=(g.state == 1).mean() if n else np.nan))

        for key, rows_, by in (("sumber", src_rows, ["sumber"]), ("ukuran", size_rows, ["ukuran"]),
                               ("ukuran_kelas", size_cls_rows, ["ukuran", "kelas"])):
            t = R.groupby(by, observed=False).apply(agg, include_groups=False).reset_index()
            t.insert(0, "model", lab)
            if key == "sumber":
                ni = pd.Series(src).value_counts()
                t.insert(2, "citra", t["sumber"].map(ni).fillna(0).astype(int))
            rows_.append(t)
    src_t = pd.concat(src_rows, ignore_index=True)
    size_t = pd.concat(size_rows, ignore_index=True)
    size_cls_t = pd.concat(size_cls_rows, ignore_index=True)
    src_t.to_csv(TABLES / "05_recall_per_sumber.csv", index=False)
    size_t.to_csv(TABLES / "05_recall_per_ukuran.csv", index=False)
    size_cls_t.to_csv(TABLES / "05_recall_per_ukuran_kelas.csv", index=False)

    # ---- 6. gambar
    plot_map_vs_ms(ov, FIG / "fig_map50_vs_inference_ms.png")
    plot_training([lab2m[best_fam["YOLOv12"]], lab2m[best_fam["YOLOv8"]]], FIG / "fig_kurva_training_terbaik.png")
    bm = lab2m[best_m50]
    st = []
    for i, im in enumerate(imgs[bm]):
        gs, ps = per[bm]["gs"][i], per[bm]["ps"][i]
        st.append(dict(i=i, file=files[i], sumber=src[i], n_gt=len(gs), miss=int((gs == 0).sum()),
                       mis=int((gs == 2).sum()), fpbg=int((ps == 0).sum()), tp=int((gs == 1).sum())))
    S = pd.DataFrame(st)
    chosen = []

    def pick(df, kind, k=1):
        for _, r in df.iterrows():
            if r.i not in [c[0] for c in chosen]:
                chosen.append((r.i, kind))
                k -= 1
                if not k:
                    return

    good = S[(S.n_gt >= 3) & (S.miss == 0) & (S.mis == 0) & (S.fpbg == 0)]
    pick(good.sort_values("n_gt", ascending=False).drop_duplicates("sumber"), "benar semua", 2)
    pick(S[(S.n_gt <= 30) & (S.miss > 0)].sort_values(["miss", "n_gt"], ascending=False), "objek terlewat (FN)")
    pick(S[S.mis > 0].sort_values("mis", ascending=False), "salah kelas")
    pick(S[S.fpbg > 0].sort_values("fpbg", ascending=False), "deteksi palsu di latar (FP)")
    fig, axs = plt.subplots(len(chosen), 2, figsize=(5.2, 2.6 * len(chosen)))
    ex_rows = []
    for row, (i, kind) in enumerate(chosen):
        r = S.loc[S.i == i].iloc[0]
        draw_pair(axs[row, 0], axs[row, 1], imgs[bm][i], per[bm]["gs"][i], per[bm]["ps"][i], per[bm]["ki"][i],
                  names, f"{kind} ({r.sumber})")
        ex_rows.append(dict(contoh=row + 1, jenis=kind, file=r.file, sumber=r.sumber, GT=r.n_gt, TP=r.tp,
                            FN_terlewat=r.miss, FN_salah_kelas=r.mis, FP_latar=r.fpbg))
    fig.text(0.5, -0.005, f"{best_m50}. Kotak hijau = {names[0]} (KPP), kuning = {names[1]} (LKK). "
             "Merah: GT putus-putus = terlewat/salah kelas; prediksi merah = FP (latar atau salah kelas). "
             f"Pencocokan IoU > {CM_IOU}.", ha="center", va="top", fontsize=5.2, color=INK2, wrap=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_contoh_prediksi.png", dpi=300)
    plt.close(fig)
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(TABLES / "06_contoh_prediksi.csv", index=False)

    # ---- 7. salinan
    cp = OUT / "salinan"
    for m in models:
        d = cp / "runs" / f"{m}_2kelas"
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("args.yaml", "results.csv"):
            shutil.copy2(RUNS / f"{m}_2kelas" / fn, d / fn)
    for f in (DATA_DIR / "README.roboflow.txt", DATA_DIR / "data.yaml", ROOT / "waktu_training_2kelas.txt"):
        shutil.copy2(f, cp / f.name)
    sdir = OUT / "scripts"
    sdir.mkdir(exist_ok=True)
    for f in ("eval_01_infer.py", "eval_01b_memori.py", "eval_02_analysis.py"):
        shutil.copy2(ROOT / "scripts_2kelas" / f, sdir / f)

    # ---- md
    T = "tables/"
    L += ["# Evaluasi dataset 2 kelas — test split penuh", "",
          "Dataset berdiri sendiri; tidak dibandingkan dengan run atau hasil lama.", "",
          "## Pengaturan evaluasi", "",
          f"- Bobot: `best.pt` tiap run di `runs_2kelas/`. Data: `dataset-2kelas/test` ({len(test_imgs)} citra), "
          "di-mirror dengan hardlink ke `eval_2kelas/_testset/` agar dataset asli tidak tersentuh.",
          "- `model.val(split='test', imgsz=512, conf=0.001, iou=0.6, batch=16, device=0, half=False, seed=0, "
          f"deterministic=True)`. GPU: {raw[models[0]]['device']}.",
          f"- Citra yang benar-benar dinilai: **{n_img}**. Dibuang Ultralytics (label campuran poligon+box, "
          f"ditandai corrupt): {len(dropped)} citra" + (": " + ", ".join(f"`{x}`" for x in dropped) if dropped else "") + ".",
          f"- Pesan labels.cache terkait: {len(corrupt_msgs)} (lihat `_testset/labels.cache`).",
          "- Waktu per citra dari `validator.speed` (ms/citra, batch 16, FP32), median dari 3 kali val. "
          "FPS_inference = 1000 / ms_inference (forward saja); FPS_total = 1000 / (pre + inference + post).",
          "- Puncak memori GPU = `torch.cuda.max_memory_allocated` / `max_memory_reserved` selama satu kali val, "
          "diukur di proses Python baru per model (`scripts/eval_01b_memori.py`, termasuk bobot model). Angka dari "
          "proses bersama (10 model berurutan) ikut menghitung sisa alokasi model sebelumnya, disimpan sebagai "
          "`*_proses_bersama` di `tables/eval_ultralytics_raw.json` dan tidak dipakai.",
          "- Params dan GFLOPs pada 512×512 dari `ultralytics.utils.torch_utils.get_num_params/get_flops`, "
          "model sebelum di-fuse.", ""]

    L += ["## 1. Run training", "", f"Sumber: `{T}01_inventaris_run.csv`, `salinan/runs/*/args.yaml`, "
          "`salinan/runs/*/results.csv`, `best.pt` (train_args), `train_2kelas.log`.", ""]
    L += md_table(inv[["model", "epoch_dijalankan", "epoch_terbaik", "early_stopping_terpicu", "batch_dipakai",
                       "lama_training_jam", "jumlah_sesi", "epochs", "patience", "imgsz", "optimizer", "lr0",
                       "momentum", "weight_decay", "seed"]])
    L += ["", f"- Optimizer yang benar-benar dipakai (log Ultralytics, `optimizer=auto`): "
          f"{', '.join(f'`{x}`' for x in opt_lines)} untuk semua run; lr0=0.01 dan momentum=0.937 di args.yaml "
          "diabaikan oleh mode auto.",
          "- weight_decay efektif = weight_decay × batch × akumulasi / nbs(64): " +
          ", ".join(f"{r.model} {fmt(r.weight_decay_efektif)}" for r in inv.itertuples()) + ".",
          "- Epoch terbaik = epoch dengan val mAP@50-95 tertinggi (fitness Ultralytics, = best.pt).",
          "- Lama training = jumlah kolom `time` results.csv per sesi (train + val tiap epoch); tidak termasuk "
          "epoch parsial yang hilang saat jeda dan validasi akhir best.pt.",
          "- Catatan run (dari `salinan/waktu_training_2kelas.txt`): patience 100 untuk YOLOv12n/s/m, 50 untuk "
          "lainnya (YOLOv12l: 100 pada epoch 1–46, lalu 50 setelah resume, state early stop dipulihkan). "
          "Batch YOLOv12 dan YOLOv8m/l/x dari AutoBatch; YOLOv8n/s batch tetap 16. workers 4, kecuali YOLOv8s (0) "
          "dan YOLOv8n epoch 1–67 (0). Jeda/resume: YOLOv12l setelah epoch 46; YOLOv8n setelah epoch 67 dan 74.",
          ""]

    L += ["## 2. Metrik test penuh", "", f"Sumber: `{T}02_metrik_test_penuh.csv`, "
          f"`{T}eval_ultralytics_raw.json`. ★ = mAP@50 tertinggi ({best_m50}).", ""]
    o2 = ov.copy()
    o2["model"] = [f"{x} ★" if x == best_m50 else x for x in o2.model]
    L += md_table(o2[["model", "mAP50", "mAP50_95", "mAP75", "precision", "recall", "n_citra"]])
    L += ["", "### Per kelas", "", f"Sumber: `{T}02_metrik_per_kelas_test_penuh.csv`.", ""]
    L += md_table(pc)
    L += ["", "### Kompleksitas dan kecepatan", "", f"Sumber: `{T}02_metrik_test_penuh.csv`.", ""]
    L += md_table(ov[["model", "params", "GFLOPs_512", "ms_preprocess", "ms_inference", "ms_postprocess",
                      "ms_total", "FPS_inference", "FPS_total", "ms_inference_min", "ms_inference_max",
                      "peak_mem_alloc_MiB", "peak_mem_reserved_MiB"]])
    L += ["", "Waktu bervariasi antar-pengulangan (lihat kolom min/max); selisih kecil antar model tidak bermakna.",
          "", "### Cek ulang mAP", "", f"mAP dihitung ulang dari matriks TP validator (`{T}02_cek_ulang_mAP.csv`), "
          f"selisih absolut maksimum terhadap Ultralytics: {fmt(float(ck.max_selisih_abs.max()))}. "
          "Ini memastikan prediksi yang dipakai untuk bootstrap sama dengan yang dinilai Ultralytics.", ""]

    L += ["## 3. Confusion matrix test penuh", "",
          f"conf ≥ {CM_CONF}, IoU > {CM_IOU} (pencocokan seperti Ultralytics ConfusionMatrix). Angka mentah: "
          f"`confusion_matrix/cm_<model>_test_penuh.csv` dan `{T}03_confusion_matrix_semua_model.csv`; PNG: "
          f"`confusion_matrix/cm_<model>_test_penuh.png`. ★ = mAP@50 tertinggi ({best_m50}). "
          "Format kolom `a->b` = GT a diprediksi b.", ""]
    cmt = pd.DataFrame(cm_rows)
    cmt["model"] = [f"{x} ★" if x == best_m50 else x for x in cmt.model]
    L += md_table(cmt)
    L += ["", f"TP/FP/FN per kelas (`{T}03_tp_fp_fn_per_kelas_conf025.csv`). FN = terlewat + salah kelas; "
          "FP = latar + salah kelas.", ""]
    L += md_table(errs)

    L += ["", "## 4. Bootstrap berpasangan YOLOv12 vs YOLOv8", "",
          f"{N_BOOT}× resampling citra dengan pengembalian (seed {SEED}), citra yang sama untuk kedua model. "
          "Δ = YOLOv12 − YOLOv8; CI 95% = persentil 2,5–97,5. mAP dari matriks TP validator (conf ≥ 0.001); "
          f"FN/FP pada conf ≥ {CM_CONF}, IoU > {CM_IOU}. Untuk mAP, Δ positif = YOLOv12 lebih baik; "
          "untuk FN/FP, Δ negatif = YOLOv12 lebih baik. Kalau CI memuat 0, dicatat setara. "
          f"Sumber: `{T}04_bootstrap_v12_vs_v8.csv`.", ""]
    for s in SCALES:
        b_ = boot[boot.skala == s]
        if not len(b_):
            continue
        L += [f"### YOLOv12{s} vs YOLOv8{s} ({n_img} citra)", ""]
        L += md_table(b_[b_.kelas == "semua"].drop(columns=["skala", "kelas"]))
        L += [""]
        t = b_[(b_.kelas != "semua") & b_.metrik.isin(["FN", "FP"])]
        L += md_table(t.drop(columns=["skala"]))
        L += ["", "Rincian FN terlewat vs salah kelas dan FP latar vs salah kelas:", ""]
        t = b_[(b_.kelas != "semua") & ~b_.metrik.isin(["FN", "FP"])]
        L += md_table(t.drop(columns=["skala"])) + [""]

    L += ["## 5. Analisis error model terbaik per keluarga", "",
          f"Model terbaik = mAP@50 test tertinggi dalam keluarganya: YOLOv12 → {best_fam['YOLOv12']}, "
          f"YOLOv8 → {best_fam['YOLOv8']}. Recall = TP / GT pada conf ≥ {CM_CONF}, IoU > {CM_IOU}, kelas benar. "
          "Sumber dari pola nama file (sama dengan analisis sebelumnya): `batch_*`/`BATCH_*` = UAVVaste, nama "
          "numerik = TACO, `WhatsApp-Video*`/`IMG_*`/`GOPR*` = Burangrang, lainnya Unknown. Ukuran = luas bbox GT "
          "dalam piksel pada input 512 (letterbox), bin COCO.", "",
          f"### Per sumber (`{T}05_recall_per_sumber.csv`)", ""]
    L += md_table(src_t)
    L += ["", f"### Per ukuran objek (`{T}05_recall_per_ukuran.csv`)", ""]
    L += md_table(size_t)
    L += ["", f"### Per ukuran dan kelas (`{T}05_recall_per_ukuran_kelas.csv`)", ""]
    L += md_table(size_cls_t)

    L += ["", "## 6. Gambar", "",
          "- `figures/fig_map50_vs_inference_ms.png`: mAP@50 test vs ms inference (forward, batch 16), 600 dpi, "
          "lebar 3,45 inci, font serif.",
          f"- `figures/fig_kurva_training_terbaik.png`: kurva training {best_fam['YOLOv12']} dan "
          f"{best_fam['YOLOv8']} (box loss, cls loss train/val; val mAP@50; val mAP@50-95), 600 dpi.",
          f"- `figures/fig_contoh_prediksi.png`: {len(chosen)} contoh {best_m50} (GT kiri, prediksi kanan), "
          f"daftar di `{T}06_contoh_prediksi.csv`:", ""]
    L += md_table(ex)
    L += ["", "## Isi folder", "",
          "- `tables/` semua CSV; `confusion_matrix/` CSV + PNG per model; `figures/` PNG; `val/` keluaran "
          "Ultralytics; `salinan/` args.yaml dan results.csv tiap run, README.roboflow.txt, data.yaml, "
          "waktu_training_2kelas.txt; `scripts/` script evaluasi; `eval_01.log` log inferensi.",
          "- `_cache/` (prediksi per citra, pickle) dan `_testset/` (hardlink test split) tidak dimasukkan ke zip.", ""]
    (OUT / "hasil_eval_2kelas.md").write_text("\n".join(L), encoding="utf-8")

    # ---- zip
    zf = ROOT / "eval_2kelas.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            rel = f.relative_to(OUT)
            if f.is_file() and rel.parts[0] not in ("_cache", "_testset"):
                z.write(f, Path("eval_2kelas") / rel)
    print(f"[ZIP] {zf} ({zf.stat().st_size / 2 ** 20:.1f} MiB)")

    # ---- ringkas
    print(ov[["model", "params", "GFLOPs_512", "ms_inference", "mAP50", "mAP50_95", "mAP75"]].to_string(index=False))
    print("cek ulang max selisih:", ck.max_selisih_abs.max())
    print("terbaik:", best_m50, best_fam)


if __name__ == "__main__":
    main()

"""Error analysis beyond mAP: confusion matrices, TP/FP/FN, YOLOv12s-aug vs YOLOv8s-aug
head-to-head with paired image-level bootstrap, confused class pairs, size/density
stratification, and a dataset-wide scan for label files Ultralytics rejects.

Read-only w.r.t. weights and dataset. Reuses the inventory and test subsets of
01_evaluate_test.py (results/test_sets/{full,clean}).

Matching rules (mirror Ultralytics):
  confusion matrix : conf >= 0.25, class-agnostic greedy matching on IoU > 0.5
                     (highest IoU first, one-to-one); unmatched GT -> FN vs background,
                     unmatched prediction -> FP vs background.
  AP50             : conf >= 0.001, class-aware greedy matching on IoU >= 0.5,
                     101-point interpolated AP.

Usage:
    python scripts/02_error_analysis.py
"""
import importlib.util
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CM_DIR = OUT / "confusion_matrix"
CACHE = OUT / "_preds"

SEED = 0
IMGSZ = 640
CM_CONF, CM_IOU, AP_IOU = 0.25, 0.5, 0.5
N_BOOT = 1000
HEAD = ("YOLOv12", "s", "ya"), ("YOLOv8", "s", "ya")
NAMES_ID = {"kertas_tisu": "Kertas/Tisu", "logam_kaleng_kaca": "Logam/Kaleng/Kaca",
            "plastik": "Plastik", "puntung_rokok": "Puntung Rokok"}
SIZE_BINS = [("small (<32²)", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large (>96²)", 96 ** 2, np.inf)]
DENSITY_BINS = [("1-2", 1, 2), ("3-5", 3, 5), (">5", 6, 10 ** 9)]

spec = importlib.util.spec_from_file_location("eval01", ROOT / "scripts" / "01_evaluate_test.py")
eval01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval01)
PRED_ARGS = eval01.EVAL_ARGS  # identical to 01_evaluate_test.py


# ---------------------------------------------------------------- labels
def parse_label_file(path):
    """Return (rows, is_mixed). rows = [(cls, x1, y1, x2, y2)] in pixels (polygons -> bbox)."""
    if not path.exists():
        return [], False
    lines = [x.split() for x in path.read_text(encoding="utf-8").strip().splitlines() if x.strip()]
    mixed = any(len(x) > 6 for x in lines) and any(len(x) == 5 for x in lines)
    rows = []
    for x in lines:
        v = np.array(x[1:], dtype=float)
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            pts = v[: len(v) // 2 * 2].reshape(-1, 2)
            (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
        rows.append((int(x[0]), x1 * IMGSZ, y1 * IMGSZ, x2 * IMGSZ, y2 * IMGSZ))
    return rows, mixed


def load_test_set(name):
    d = OUT / "test_sets" / name
    imgs, gts, rejected = [], [], []
    for p in sorted((d / "images").iterdir()):
        rows, mixed = parse_label_file(d / "labels" / (p.stem + ".txt"))
        if mixed:  # Ultralytics drops these images entirely; do the same for consistency
            rejected.append(p.name)
            continue
        imgs.append(p)
        gts.append(np.array(rows, dtype=float).reshape(-1, 5))
    return imgs, gts, rejected


# ---------------------------------------------------------------- predictions
def predict(rec, set_name, imgs):
    """Capture predictions and ground truth from inside model.val(), so they are exactly what
    01_evaluate_test.py scored (same letterbox, multi-label NMS, batch). Returns (preds, gts)
    aligned with imgs; preds rows = (cls, x1, y1, x2, y2, conf), gts rows = (cls, x1, y1, x2, y2)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{rec['run']}_{set_name}.pkl"
    if f.exists():
        cached = pickle.loads(f.read_bytes())
        if cached["files"] == [p.name for p in imgs] and cached["args"] == PRED_ARGS:
            return cached["preds"], cached["gts"]
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops
    from ultralytics.utils.torch_utils import init_seeds

    store = {}
    orig = DetectionValidator.update_metrics

    def hooked(self, preds, batch):
        for si, pred in enumerate(preds):
            pb = self._prepare_batch(si, batch)
            p = self.scale_preds(pred, pb)
            g = ops.scale_boxes(pb["imgsz"], pb["bboxes"].clone(), pb["ori_shape"], ratio_pad=pb["ratio_pad"])
            store[Path(pb["im_file"]).name] = (
                torch.cat([p["cls"][:, None], p["bboxes"], p["conf"][:, None]], 1).cpu().numpy().reshape(-1, 6),
                torch.cat([pb["cls"].reshape(-1, 1), g.reshape(-1, 4)], 1).cpu().numpy().reshape(-1, 5))
        return orig(self, preds, batch)

    model = YOLO(str(rec["best_pt"]))
    DetectionValidator.update_metrics = hooked
    try:
        init_seeds(SEED, deterministic=True)
        model.val(data=str(OUT / "test_sets" / f"test_{set_name}.yaml"), project=str(OUT / "_val_runs"),
                  name=f"{rec['run']}_{set_name}_err", exist_ok=True, **PRED_ARGS)
    finally:
        DetectionValidator.update_metrics = orig
        del model
        torch.cuda.empty_cache()
    missing = [p.name for p in imgs if p.name not in store]
    if missing:
        raise RuntimeError(f"{len(missing)} citra tidak diprediksi, mis. {missing[0]}")
    preds = [store[p.name][0] for p in imgs]
    gts = [store[p.name][1] for p in imgs]
    f.write_bytes(pickle.dumps(dict(files=[p.name for p in imgs], args=PRED_ARGS, preds=preds, gts=gts)))
    return preds, gts


# ---------------------------------------------------------------- matching
def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def greedy(iou, thr, cm_style):
    """One-to-one matches (gt_idx, pred_idx), replicating Ultralytics exactly:
    ConfusionMatrix.process_batch (IoU > thr, re-sorted by IoU before de-duplicating GT) and
    DetectionValidator.match_predictions (IoU >= thr, no re-sort, so among the predictions whose
    best GT is g, the one with the lowest index = highest confidence wins)."""
    gi, pi = np.nonzero(iou > thr if cm_style else iou >= thr)
    if not len(gi):
        return np.zeros((0, 2), int)
    m = np.stack([gi, pi], 1)
    if len(m) > 1:
        m = m[np.argsort(-iou[gi, pi], kind="stable")]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        if cm_style:
            m = m[np.argsort(-iou[m[:, 0], m[:, 1]], kind="stable")]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
    return m


def cm_image(gt, pr, nc):
    """Per-image confusion counts. cm[true, pred], index nc = background.
    Also returns per-GT outcome: 1 = correct, 0 = missed or wrong class."""
    pr = pr[pr[:, 5] >= CM_CONF]
    cm = np.zeros((nc + 1, nc + 1), int)
    hit = np.zeros(len(gt), int)
    m = greedy(iou_matrix(gt[:, 1:5], pr[:, 1:5]), CM_IOU, cm_style=True)
    mg, mp = set(m[:, 0]), set(m[:, 1])
    for g, p in m:
        tc, pc = int(gt[g, 0]), int(pr[p, 0])
        cm[tc, pc] += 1
        hit[g] = tc == pc
    for g in range(len(gt)):
        if g not in mg:
            cm[int(gt[g, 0]), nc] += 1
    for p in range(len(pr)):
        if p not in mp:
            cm[nc, int(pr[p, 0])] += 1
    return cm, hit


def ap_records(gt, pr):
    """Class-aware AP50 matching. Returns pred tp flags and index of matched GT (-1 if none)."""
    iou = iou_matrix(gt[:, 1:5], pr[:, 1:5]) * (gt[:, :1] == pr[None, :, 0])
    m = greedy(iou, AP_IOU, cm_style=False)
    tp = np.zeros(len(pr), bool)
    gidx = -np.ones(len(pr), int)
    tp[m[:, 1]] = True
    gidx[m[:, 1]] = m[:, 0]
    return tp, gidx


def ap50(conf, tp, n_gt, w=None):
    """101-point interpolated AP; optional per-prediction weights (for bootstrap)."""
    if n_gt <= 0:
        return np.nan
    if not len(conf):
        return 0.0
    w = np.ones(len(conf)) if w is None else w
    o = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[o] * w[o])
    fpc = np.cumsum(~tp[o] * w[o])
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-16)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1.0], prec, [0.0]]))))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


# ---------------------------------------------------------------- per model / set
def analyse(gts, preds, nc):
    """Everything later steps need, kept per image so it can be bootstrapped/stratified."""
    n = len(gts)
    cm = np.zeros((nc + 1, nc + 1), int)
    tp_i, fn_i, fp_i = np.zeros((n, nc)), np.zeros((n, nc)), np.zeros((n, nc))
    gt_hit, ap_rows = [], []
    for i, (g, p) in enumerate(zip(gts, preds)):
        c, hit = cm_image(g, p, nc)
        cm += c
        d = np.diag(c)[:nc]
        tp_i[i] = d
        fn_i[i] = c[:nc, :].sum(1) - d
        fp_i[i] = c[:, :nc].sum(0) - d
        gt_hit.append(hit)
        tp, gidx = ap_records(g, p)
        ap_rows.append((p, tp, gidx))
    return dict(cm=cm, tp=tp_i, fn=fn_i, fp=fp_i, gt_hit=gt_hit, ap=ap_rows)


def map50(gts, res, nc, img_w=None, gt_keep=None, pred_keep=None):
    """mean AP50 over classes with GT. img_w: per-image weights (bootstrap / image subset).
    gt_keep / pred_keep: per-image boolean masks for object-level strata; predictions
    matched to a GT outside the stratum are ignored, as in COCO area ranges."""
    n = len(gts)
    img_w = np.ones(n) if img_w is None else img_w
    if gt_keep is None and pred_keep is None:  # fast path, used by the bootstrap
        if "flat" not in res:
            res["flat"] = (np.concatenate([np.full(len(p), i) for i, (p, _, _) in enumerate(res["ap"])]),
                           np.concatenate([p[:, 0] for p, _, _ in res["ap"]]),
                           np.concatenate([p[:, 5] for p, _, _ in res["ap"]]),
                           np.concatenate([t for _, t, _ in res["ap"]]),
                           np.stack([np.bincount(g[:, 0].astype(int), minlength=nc) for g in gts]))
        img, cls, conf, tp, ngt = res["flat"]
        aps = {}
        for c in range(nc):
            n_gt = img_w @ ngt[:, c]
            if n_gt > 0:
                s = (cls == c) & (img_w[img] > 0)
                aps[c] = ap50(conf[s], tp[s], n_gt, img_w[img[s]])
        return aps
    aps = {}
    for c in range(nc):
        conf, tp, w, n_gt = [], [], [], 0.0
        for i in range(n):
            if img_w[i] == 0:
                continue
            g = gts[i]
            gk = np.ones(len(g), bool) if gt_keep is None else gt_keep[i]
            p, t, gidx = res["ap"][i]
            pk = np.ones(len(p), bool) if pred_keep is None else pred_keep[i]
            n_gt += img_w[i] * np.sum((g[:, 0] == c) & gk)
            sel = p[:, 0] == c
            matched_out = (gidx >= 0) & ~gk[np.clip(gidx, 0, None)] if len(g) else np.zeros(len(p), bool)
            keep = sel & np.where(gidx >= 0, ~matched_out, pk)
            conf.append(p[keep, 5]); tp.append(t[keep]); w.append(np.full(keep.sum(), img_w[i]))
        if n_gt > 0:
            aps[c] = ap50(np.concatenate(conf), np.concatenate(tp), n_gt, np.concatenate(w))
    return aps


# ---------------------------------------------------------------- plots
def plot_cm(cm, names, title, path):
    labels = [NAMES_ID.get(n, n) for n in names] + ["Latar\n(background)"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = max(cm[:-1, :-1].max(), 1)
    ax.imshow(np.minimum(cm, vmax), cmap="Blues", vmin=0, vmax=vmax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j == cm.shape[0] - 1:
                ax.text(j, i, "–", ha="center", va="center", color="grey", fontsize=10)
                continue
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    color="white" if v > vmax * 0.55 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Sebenarnya (ground truth)", fontsize=11)
    ax.set_title(title, fontsize=11)
    fig.text(0.99, 0.01, f"conf ≥ {CM_CONF}, IoU > {CM_IOU}. Baris latar = FP, kolom latar = FN.",
             ha="right", va="bottom", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- label scan
def scan_mixed_labels():
    rows = []
    for ds in ("TA-1", "augmented"):
        for split in ("train", "valid", "test"):
            ldir = ROOT / ds / split / "labels"
            if not ldir.exists():
                continue
            for f in sorted(ldir.glob("*.txt")):
                lines = [x.split() for x in f.read_text(encoding="utf-8").strip().splitlines() if x.strip()]
                n_box = sum(len(x) == 5 for x in lines)
                n_seg = sum(len(x) > 6 for x in lines)
                rows.append(dict(dataset=ds, split=split, file=f.name, n_box=n_box, n_seg=n_seg,
                                 mixed=n_box > 0 and n_seg > 0,
                                 classes=",".join(sorted({x[0] for x in lines}))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers
def label(model, varian, aug):
    return f"{model}{varian}" + (" aug" if aug == "ya" else "")


def fname(model, varian, aug):
    return f"{model}{varian}" + ("_aug" if aug == "ya" else "")


def verdict(lo, hi, lower_is_better=True):
    if lo <= 0 <= hi:
        return "setara"
    better = hi < 0 if lower_is_better else lo > 0
    return "YOLOv12 lebih baik" if better else "YOLOv12 lebih buruk"


def f3(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return out


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    CM_DIR.mkdir(parents=True, exist_ok=True)
    runs, _ = eval01.inventory()
    if not (OUT / "test_sets" / "clean").exists():
        eval01.build_test_sets(ROOT / "TA-1" / "data.yaml")
    import yaml

    names = yaml.safe_load((ROOT / "TA-1" / "data.yaml").read_text())["names"]
    nc = len(names)
    sets = {s: load_test_set(s) for s in ("full", "clean")}
    for s, (imgs, gts, rej) in sets.items():
        print(f"Test {s}: {len(imgs)} citra dianalisis, {sum(len(g) for g in gts)} objek"
              + (f", {len(rej)} ditolak (label campuran): {', '.join(rej)}" if rej else ""))

    # reference mAP50 from 01 to sanity-check the re-implemented matching
    ref = pd.read_csv(OUT / "test_metrics_overall.csv") if (OUT / "test_metrics_overall.csv").exists() else None

    results, rows, skipped = {}, [], []
    for i, rec in enumerate(runs, 1):
        key = (rec["model"], rec["varian"], rec["augmentasi"])
        print(f"[{i}/{len(runs)}] {label(*key)}", flush=True)
        try:
            for s, (imgs, gts, rej) in sets.items():
                preds, gts_val = predict(rec, s, imgs)
                n_own, n_val = sum(len(g) for g in gts), sum(len(g) for g in gts_val)
                if n_own != n_val:
                    raise RuntimeError(f"jumlah GT berbeda: label {n_own} vs validator {n_val}")
                gts = gts_val  # use exactly the GT boxes Ultralytics scored
                sets[s] = (imgs, gts, rej)
                res = analyse(gts, preds, nc)
                results[key + (s,)] = res
                plot_cm(res["cm"], names, f"{label(*key)} — test {s} ({len(imgs)} citra)",
                        CM_DIR / f"{fname(*key)}_{s}.png")
                aps = map50(gts, res, nc)
                for c, n in enumerate(names):
                    tp, fn, fp = res["tp"][:, c].sum(), res["fn"][:, c].sum(), res["fp"][:, c].sum()
                    rows.append(dict(model=rec["model"], varian=rec["varian"], augmentasi=rec["augmentasi"],
                                     test_set=s, kelas=n, gt=int(tp + fn), TP=int(tp), FP=int(fp), FN=int(fn),
                                     FN_terlewat=int(res["cm"][c, nc]),
                                     FN_salah_kelas=int(fn - res["cm"][c, nc]),
                                     FP_latar=int(res["cm"][nc, c]),
                                     FP_salah_kelas=int(fp - res["cm"][nc, c]),
                                     precision=tp / (tp + fp) if tp + fp else np.nan,
                                     recall=tp / (tp + fn) if tp + fn else np.nan,
                                     AP50=aps.get(c, np.nan)))
                m = np.nanmean(list(aps.values()))
                msg = f"      {s:5s}: mAP50 {m:.3f}"
                if ref is not None:
                    r = ref[(ref.model == key[0]) & (ref.varian == key[1]) & (ref.augmentasi == key[2])
                            & (ref.test_set == s)]
                    if len(r):
                        msg += f" (01_evaluate_test: {r.mAP50.iloc[0]:.3f})"
                print(msg, flush=True)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}".splitlines()[0][:200]
            print(f"      [SKIP] {msg}", flush=True)
            skipped.append((rec["run"], msg))

    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / "fp_fn_per_class.csv", index=False)

    L = ["# Analisis Error: YOLOv12 vs YOLOv8", "",
         f"Prediksi pada imgsz={IMGSZ}, NMS IoU=0.6. Confusion matrix dan TP/FP/FN memakai conf ≥ {CM_CONF} "
         f"dan IoU > {CM_IOU} (pencocokan seperti Ultralytics ConfusionMatrix). AP50 memakai conf ≥ 0.001.", "",
         f"- Test full: {len(sets['full'][0])} citra (citra berlabel campuran dibuang, sama seperti Ultralytics).",
         f"- Test clean: {len(sets['clean'][0])} citra. puntung_rokok tidak punya instance di test clean.",
         "- FN = objek yang tidak terdeteksi (terlewat) + objek yang terdeteksi dengan kelas salah.",
         "- FP = deteksi pada latar + deteksi dengan kelas salah.", ""]

    # ------------------------------------------------ 3. head-to-head
    a_key, b_key = HEAD
    have = all(k + (s,) in results for k in HEAD for s in sets)
    rng = np.random.default_rng(SEED)
    boot_rows = []
    if have:
        A, B = label(*a_key), label(*b_key)
        L += [f"## 1. Head-to-head {A} vs {B}", "",
              "Δ = YOLOv12 − YOLOv8. Untuk FN dan FP, Δ negatif berarti YOLOv12 lebih baik.", ""]
        for s, (imgs, gts, _) in sets.items():
            ra, rb = results[a_key + (s,)], results[b_key + (s,)]
            n = len(gts)
            idx = rng.integers(0, n, size=(N_BOOT, n))
            w_all = np.stack([np.bincount(ix, minlength=n) for ix in idx])  # (N_BOOT, n)
            tab = []
            for c, cname in enumerate(names + ["TOTAL"]):
                cs = slice(None) if cname == "TOTAL" else c
                gt_n = int((ra["tp"][:, cs].sum() + ra["fn"][:, cs].sum()))
                if gt_n == 0 and cname != "TOTAL":
                    tab.append({"Kelas": cname, "GT": 0, "FN v12": "–", "FN v8": "–", "ΔFN": "–", "ΔFN %": "–",
                                "95% CI ΔFN": "–", "FP v12": int(ra["fp"][:, c].sum()), "FP v8": int(rb["fp"][:, c].sum()),
                                "ΔFP": int(ra["fp"][:, c].sum() - rb["fp"][:, c].sum()), "95% CI ΔFP": "–",
                                "Kesimpulan FN": "tidak ada GT", "Kesimpulan FP": "–"})
                    continue
                fa = ra["fn"][:, cs].reshape(n, -1).sum(1); fb = rb["fn"][:, cs].reshape(n, -1).sum(1)
                pa = ra["fp"][:, cs].reshape(n, -1).sum(1); pb = rb["fp"][:, cs].reshape(n, -1).sum(1)
                dfn = fa.sum() - fb.sum(); dfp = pa.sum() - pb.sum()
                bfn = w_all @ (fa - fb); bfp = w_all @ (pa - pb)
                ci_fn = np.percentile(bfn, [2.5, 97.5]); ci_fp = np.percentile(bfp, [2.5, 97.5])
                tab.append({"Kelas": cname, "GT": gt_n, "FN v12": int(fa.sum()), "FN v8": int(fb.sum()),
                            "ΔFN": f"{int(dfn):+d}",
                            "ΔFN %": f"{100 * dfn / fb.sum():+.1f}%" if fb.sum() else "–",
                            "95% CI ΔFN": f"[{ci_fn[0]:+.0f}, {ci_fn[1]:+.0f}]",
                            "FP v12": int(pa.sum()), "FP v8": int(pb.sum()), "ΔFP": f"{int(dfp):+d}",
                            "95% CI ΔFP": f"[{ci_fp[0]:+.0f}, {ci_fp[1]:+.0f}]",
                            "Kesimpulan FN": verdict(*ci_fn), "Kesimpulan FP": verdict(*ci_fp)})
                boot_rows.append(dict(test_set=s, kelas=cname, metrik="FN", v12=int(fa.sum()), v8=int(fb.sum()),
                                      delta=int(dfn), ci_lo=ci_fn[0], ci_hi=ci_fn[1], kesimpulan=verdict(*ci_fn)))
                boot_rows.append(dict(test_set=s, kelas=cname, metrik="FP", v12=int(pa.sum()), v8=int(pb.sum()),
                                      delta=int(dfp), ci_lo=ci_fp[0], ci_hi=ci_fp[1], kesimpulan=verdict(*ci_fp)))
            # recall and mAP50 difference CI
            ma, mb = map50(gts, ra, nc), map50(gts, rb, nc)
            bm = []
            for w in w_all:
                xa, xb = map50(gts, ra, nc, img_w=w), map50(gts, rb, nc, img_w=w)
                common = [c for c in xa if c in xb and not np.isnan(xa[c]) and not np.isnan(xb[c])]
                bm.append(np.mean([xa[c] for c in common]) - np.mean([xb[c] for c in common]))
            ci_m = np.nanpercentile(bm, [2.5, 97.5])
            dm = np.mean(list(ma.values())) - np.mean(list(mb.values()))
            boot_rows.append(dict(test_set=s, kelas="mAP50", metrik="mAP50", v12=np.mean(list(ma.values())),
                                  v8=np.mean(list(mb.values())), delta=dm, ci_lo=ci_m[0], ci_hi=ci_m[1],
                                  kesimpulan=verdict(*ci_m, lower_is_better=False)))
            L += [f"### Test {s} ({n} citra, bootstrap {N_BOOT}× berpasangan pada level citra)", ""]
            L += md_table(pd.DataFrame(tab))
            L += ["", f"mAP50 ({len(ma)} kelas): {A} {np.mean(list(ma.values())):.3f}, {B} "
                      f"{np.mean(list(mb.values())):.3f}, Δ {dm:+.3f}, 95% CI [{ci_m[0]:+.3f}, {ci_m[1]:+.3f}].", ""]
            # breakdown of FN: missed vs wrong class
            sub = pc[(pc.test_set == s) & (pc.augmentasi == "ya") & (pc.varian == "s")
                     & pc.model.isin(["YOLOv12", "YOLOv8"]) & (pc["gt"] > 0)]
            br = sub.pivot_table(index="kelas", columns="model",
                                 values=["FN_terlewat", "FN_salah_kelas", "FP_latar", "FP_salah_kelas"],
                                 aggfunc="sum")
            br.columns = [f"{a} {b}" for a, b in br.columns]
            br = br.reset_index()
            L += ["Rincian FN (terlewat vs salah kelas) dan FP (latar vs salah kelas):", ""]
            L += md_table(br) + [""]
        L += ["Kesimpulan per kelas ditentukan oleh CI 95%: kalau CI memuat 0, selisihnya masih dalam "
              "rentang noise dan dicatat **setara**.", ""]
        pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_head_to_head.csv", index=False)

    # ------------------------------------------------ 4. confused pairs
    L += ["## 2. Pasangan kelas yang paling sering tertukar", "",
          "Jumlah objek dengan kelas sebenarnya A yang diprediksi sebagai B (conf ≥ 0.25, IoU > 0.5). "
          "Kolom 'semua model' = jumlah dari semua model yang dievaluasi.", ""]
    for s in sets:
        pair = defaultdict(lambda: defaultdict(int))
        for k, res in results.items():
            if k[3] != s:
                continue
            for t in range(nc):
                for p in range(nc):
                    if t != p:
                        pair[(t, p)]["all"] += res["cm"][t, p]
                        if k[:3] in HEAD:
                            pair[(t, p)][k[0]] += res["cm"][t, p]
        und = defaultdict(lambda: defaultdict(int))
        for (t, p), v in pair.items():
            k = tuple(sorted((t, p)))
            for m, x in v.items():
                und[k][m] += x
                und[k][f"{m}_{t}->{p}"] = x
        tbl = []
        for (a, b), v in sorted(und.items(), key=lambda kv: -kv[1]["all"]):
            if v["all"] == 0:
                continue
            tbl.append({"Pasangan (A ↔ B)": f"{names[a]} ↔ {names[b]}", "Semua model": v["all"],
                        "A→B": v[f"all_{a}->{b}"], "B→A": v[f"all_{b}->{a}"],
                        "YOLOv12s aug": v["YOLOv12"], "YOLOv8s aug": v["YOLOv8"]})
        df = pd.DataFrame(tbl)
        L += [f"### Test {s}", ""] + (md_table(df) if len(df) else ["Tidak ada kelas tertukar."]) + [""]

    # ------------------------------------------------ 5. stratification (clean)
    if have:
        imgs, gts, _ = sets["clean"]
        L += ["## 3. Stratifikasi pada test clean", "",
              "Recall pada conf ≥ 0.25 (objek dihitung benar bila terdeteksi dengan kelas benar). "
              "mAP50 = rata-rata AP50 kelas yang punya objek di kelompok tsb. Untuk strata ukuran, deteksi "
              "yang cocok dengan objek di luar strata diabaikan, dan deteksi yang tidak cocok dihitung FP "
              "hanya bila ukurannya sendiri masuk strata (konvensi COCO area range).", ""]
        area = lambda b: (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
        st_rows = []
        for title, bins, kind in (("Ukuran objek (luas bbox pada 640×640)", SIZE_BINS, "size"),
                                  ("Kepadatan objek per citra", DENSITY_BINS, "density")):
            tbl = []
            for bname, lo, hi in bins:
                row = {"Kelompok": bname}
                for key in HEAD:
                    res = results[key + ("clean",)]
                    if kind == "size":
                        gk = [(area(g) >= lo) & (area(g) < hi) for g in gts]
                        pk = [(area(p[0]) >= lo) & (area(p[0]) < hi) for p in res["ap"]]
                        hits = np.concatenate([h[k] for h, k in zip(res["gt_hit"], gk)])
                        aps = map50(gts, res, nc, gt_keep=gk, pred_keep=pk)
                        n_img = sum(k.any() for k in gk)
                    else:
                        cnt = np.array([len(g) for g in gts])
                        w = ((cnt >= lo) & (cnt <= hi)).astype(float)
                        hits = np.concatenate([h for h, x in zip(res["gt_hit"], w) if x])
                        aps = map50(gts, res, nc, img_w=w)
                        n_img = int(w.sum())
                    tag = "v12" if key[0] == "YOLOv12" else "v8"
                    row["Citra"] = n_img
                    row["Objek"] = len(hits)
                    row[f"Recall {tag}"] = f3(hits.mean() if len(hits) else np.nan)
                    row[f"mAP50 {tag}"] = f3(np.nanmean(list(aps.values())) if aps else np.nan)
                    st_rows.append(dict(strata=kind, kelompok=bname, model=label(*key), citra=n_img,
                                        objek=len(hits), recall=hits.mean() if len(hits) else np.nan,
                                        mAP50=np.nanmean(list(aps.values())) if aps else np.nan,
                                        **{f"AP50_{names[c]}": v for c, v in aps.items()}))
                tbl.append(row)
            df = pd.DataFrame(tbl)[["Kelompok", "Citra", "Objek", "Recall v12", "Recall v8", "mAP50 v12", "mAP50 v8"]]
            L += [f"### {title}", ""] + md_table(df) + [""]
        pd.DataFrame(st_rows).to_csv(OUT / "stratifikasi_clean.csv", index=False)
        L += ["Kelompok dengan sedikit objek (lihat kolom Objek) punya ketidakpastian besar; "
              "selisih kecil di sana jangan dibaca sebagai perbedaan nyata.", ""]

    # ------------------------------------------------ 6. mixed label scan
    scan = scan_mixed_labels()
    scan.to_csv(OUT / "label_campuran_scan.csv", index=False)
    mixed = scan[scan.mixed]
    L += ["## 4. Cek label campuran (poligon + box dalam satu file)", "",
          "Aturan Ultralytics: kalau satu file label berisi baris poligon (>6 kolom) dan baris box "
          "(5 kolom), seluruh citra dibuang dari training/evaluasi, beserta semua anotasinya.", ""]
    summ = scan.groupby(["dataset", "split"]).agg(file_label=("file", "size"),
                                                   file_campuran=("mixed", "sum")).reset_index()
    lost = mixed.assign(obj=mixed.n_box + mixed.n_seg).groupby(["dataset", "split"])["obj"].sum()
    summ["objek_terbuang"] = [int(lost.get((d, s), 0)) for d, s in zip(summ.dataset, summ.split)]
    L += md_table(summ) + [""]
    if len(mixed):
        L += ["File bermasalah:", ""]
        L += md_table(mixed[["dataset", "split", "file", "n_box", "n_seg", "classes"]]) + [""]
        tr = mixed[mixed.split == "train"]
        if len(tr):
            L += [f"Ada {len(tr)} file di split train ({', '.join(sorted(tr.dataset.unique()))}). Citra-citra "
                  "ini **tidak pernah dipakai saat training**, beserta seluruh anotasinya.", ""]
        else:
            L += ["Tidak ada file bermasalah di split train, jadi tidak ada anotasi training yang terbuang.", ""]
    else:
        L += ["Tidak ditemukan file label campuran.", ""]

    # ------------------------------------------------ conclusion
    if have:
        bt = pd.DataFrame(boot_rows)
        cl = bt[(bt.test_set == "clean") & (bt.metrik == "FN")]
        tot = cl[cl.kelas == "TOTAL"].iloc[0]
        per = cl[cl.kelas != "TOTAL"]
        better = per[per.kesimpulan == "YOLOv12 lebih baik"].kelas.tolist()
        worse = per[per.kesimpulan == "YOLOv12 lebih buruk"].kelas.tolist()
        lean_v8 = per[per.delta > 0].kelas.tolist()
        lean_v12 = per[per.delta < 0].kelas.tolist()
        supported = tot.kesimpulan == "YOLOv12 lebih baik"
        txt = (f"Pada test clean, {label(*a_key)} menghasilkan {int(tot.v12)} FN dan {label(*b_key)} {int(tot.v8)} FN "
               f"(Δ {int(tot.delta):+d}, 95% CI [{tot.ci_lo:+.0f}, {tot.ci_hi:+.0f}]). ")
        if supported:
            txt += "CI tidak memuat 0, sehingga hipotesis bahwa YOLOv12 menghasilkan FN lebih sedikit **terdukung** pada data ini. "
        elif tot.kesimpulan == "YOLOv12 lebih buruk":
            txt += "CI berada di atas 0: YOLOv12 justru menghasilkan FN **lebih banyak**, sehingga hipotesis **tidak terdukung**. "
        else:
            txt += ("CI memuat 0, sehingga perbedaan jumlah FN masih dalam rentang noise dan hipotesis bahwa "
                    "YOLOv12 menghasilkan FN lebih sedikit **tidak terdukung** oleh data ini. ")
        if better:
            txt += f"Secara per kelas, YOLOv12 lebih baik secara bermakna pada {', '.join(better)}. "
        if worse:
            txt += f"YOLOv8 lebih unggul secara bermakna pada {', '.join(worse)}. "
        if not better and not worse:
            txt += ("Tidak ada kelas dengan selisih FN yang bermakna. Secara arah saja (tanpa signifikansi), "
                    f"YOLOv12 punya FN lebih sedikit pada {', '.join(lean_v12) or '-'} dan YOLOv8 punya FN lebih "
                    f"sedikit pada {', '.join(lean_v8) or '-'}. ")
        txt += ("Kesimpulan ini terbatas pada pasangan varian s augmented, satu seed training, dan test clean "
                f"berukuran {len(sets['clean'][0])} citra tanpa instance puntung_rokok.")
        L += ["## 5. Kesimpulan", "", txt, ""]
    if skipped:
        L += ["## Run yang dilewati", ""] + [f"- {r}: {m}" for r, m in skipped] + [""]

    (OUT / "analisis_error.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nSelesai dalam {(time.time() - t0) / 60:.1f} menit. Output: results/analisis_error.md, "
          f"results/fp_fn_per_class.csv, results/confusion_matrix/ ({len(list(CM_DIR.glob('*.png')))} PNG)")
    if skipped:
        print("Dilewati:", *[f"{r}: {m}" for r, m in skipped], sep="\n  ")


if __name__ == "__main__":
    sys.exit(main())

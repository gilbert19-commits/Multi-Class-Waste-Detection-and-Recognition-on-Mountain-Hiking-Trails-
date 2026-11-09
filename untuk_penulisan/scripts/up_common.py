"""Fungsi bersama script untuk_penulisan. Hanya membaca runs_2kelas/, dataset-2kelas/, dan hasil_2kelas/.

Empat versi mAP (lihat metrics_for_versions):
1 standar: AP per kelas lalu dirata-rata, sama dengan Ultralytics.
2 pooled class-aware: TP sama dengan versi 1, tetapi semua prediksi dan objek digabung dalam satu kurva PR.
3 class-agnostic: semua kelas jadi satu, NMS ulang tanpa melihat kelas (IoU 0.7), lalu dicocokkan ulang.
4 class-agnostic tanpa NMS ulang: semua kelas jadi satu, langsung dicocokkan ulang.
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from ultralytics.utils.metrics import ap_per_class, box_iou

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hasil_2kelas" / "scripts"))
from hk_common import COL_W, DATA_DIR, EVAL_ARGS, FAMILIES, MIRROR, MODELS, NAMES, PAPER, RUNS, SCALES, SEED, base_name, family, label, paper_style, source_of  # noqa: E402,F401

HK = ROOT / "hasil_2kelas"
HK_CACHE = HK / "_cache"
OUT = ROOT / "untuk_penulisan"
CACHE = OUT / "_cache"
NMS_IOU = 0.7
IOUV = torch.linspace(0.5, 0.95, 10).tolist()  # float32 dibulatkan sama dengan validator
N_BOOT = 1000
N_WORKERS = 6
METRICS = ("mAP50", "mAP75", "mAP50_95")
VERSIONS = ("v1", "v2", "v3", "v4")


def source_bukti(fname):
    """Aturan usulan dari pencocokan nama dengan anotasi TACO dan UAVVaste (b3_bukti_sumber.py). Burangrang: nama yang tidak ada di keduanya."""
    b = base_name(fname)
    if re.fullmatch(r"\d+", b) or re.fullmatch(r"IMG_\d+", b):
        return "TACO"
    if b.startswith("WhatsApp-Video") or re.fullmatch(r"IMG_\d+_JPG", b):
        return "Burangrang"
    return "UAVVaste"


def match_tp(gt_cls, gt_box, pr_cls, pr_box):
    """Salinan BaseValidator.match_predictions (use_scipy=False). Hasil: bool (n_pred, 10)."""
    correct = np.zeros((len(pr_cls), len(IOUV)), dtype=bool)
    if not len(gt_cls) or not len(pr_cls):
        return correct
    iou = box_iou(torch.from_numpy(gt_box), torch.from_numpy(pr_box)).numpy()
    iou = iou * (gt_cls[:, None] == pr_cls)
    for i, thr in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def agnostic_keep(pr_box, pr_conf):
    """Indeks prediksi yang lolos NMS tanpa melihat kelas. Sama dengan non_max_suppression(agnostic=True)."""
    if not len(pr_conf):
        return np.zeros(0, dtype=int)
    return torchvision.ops.nms(torch.from_numpy(pr_box), torch.from_numpy(pr_conf), NMS_IOU).numpy()


def image_versions(gt_cls, gt_box, pr_cls, pr_conf, pr_box):
    """Rekaman satu gambar untuk keempat versi. TP versi 2 sama dengan versi 1."""
    z_gt, z_pr = np.zeros_like(gt_cls), np.zeros_like(pr_cls)
    keep = agnostic_keep(pr_box, pr_conf)
    return dict(gt_cls=gt_cls, pr_cls=pr_cls, conf=pr_conf, tp=match_tp(gt_cls, gt_box, pr_cls, pr_box), keep3=keep,
                tp3=match_tp(z_gt, gt_box, z_pr[keep], pr_box[keep]), tp4=match_tp(z_gt, gt_box, z_pr, pr_box))


def arrays(imgs, version):
    """(tp, conf, cls, target) gabungan beberapa gambar untuk satu versi."""
    target = np.concatenate([x["gt_cls"] for x in imgs])
    conf = np.concatenate([x["conf"] for x in imgs])
    if version == "v1":
        return np.concatenate([x["tp"] for x in imgs]), conf, np.concatenate([x["pr_cls"] for x in imgs]), target
    if version == "v2":
        tp = np.concatenate([x["tp"] for x in imgs])
    elif version == "v3":
        tp = np.concatenate([x["tp3"] for x in imgs])
        conf = np.concatenate([x["conf"][x["keep3"]] for x in imgs])
    else:
        tp = np.concatenate([x["tp4"] for x in imgs])
    return tp, conf, np.zeros(len(conf), dtype=int), np.zeros(len(target), dtype=int)


def ap_matrix(imgs, version):
    """AP (nc, 10) dari ap_per_class Ultralytics. Versi 1: nc = 2, baris kelas tanpa GT = NaN. Versi lain: nc = 1."""
    tp, conf, cls, target = arrays(imgs, version)
    nc = 2 if version == "v1" else 1
    out = np.full((nc, 10), np.nan)
    if len(target):
        r = ap_per_class(tp, conf, cls, target)
        out[r[6].astype(int)] = r[5]
    return out


def summarize(ap):
    if np.isnan(ap).all():
        return dict(mAP50=np.nan, mAP75=np.nan, mAP50_95=np.nan)
    return dict(mAP50=float(np.nanmean(ap[:, 0])), mAP75=float(np.nanmean(ap[:, 5])), mAP50_95=float(np.nanmean(ap.mean(1))))


METRIC_FN = {"mAP50": lambda a: np.nanmean(a[..., 0], -1), "mAP75": lambda a: np.nanmean(a[..., 5], -1),
             "mAP50_95": lambda a: np.nanmean(a.mean(-1), -1)}


def load_preds(m):
    d = pickle.loads((HK_CACHE / f"{m}_2kelas_test.pkl").read_bytes())["images"]
    return {f: d[f] for f in sorted(d)}


def load_versions(m):
    return pickle.loads((CACHE / f"versi_{m}.pkl").read_bytes())


def best_std():
    """Model terbaik tiap family menurut mAP@50-95 standar di 550 citra, sama dengan hasil_2kelas."""
    T = pd.read_csv(HK / "metrics_test.csv")
    lab2m = {label(m): m for m in MODELS}
    return {f: lab2m[T[T.family == f].sort_values("mAP50_95", ascending=False).iloc[0]["model"]] for f in FAMILIES}


def scale_pairs():
    """15 pasangan per skala, urutannya sama dengan bootstrap hasil_2kelas."""
    out = []
    for s in SCALES:
        out += [(s, f"yolo26{s}", f"yolov12{s}"), (s, f"yolo26{s}", f"yolov8{s}"), (s, f"yolov12{s}", f"yolov8{s}")]
    return out


def best_pairs(best):
    return [(best["YOLO26"], best["YOLOv12"]), (best["YOLO26"], best["YOLOv8"]), (best["YOLOv12"], best["YOLOv8"])]


def boot_p(d):
    """p dua sisi dari distribusi bootstrap: (mentah, dengan koreksi +1). Yang mentah bisa 0 kalau tidak ada resample melewati nol."""
    b = len(d)
    le, ge = int((d <= 0).sum()), int((d >= 0).sum())
    return min(1.0, 2 * min(le, ge) / b), min(1.0, 2 * (min(le, ge) + 1) / (b + 1))


def holm(p):
    """p terkoreksi Holm (step-down, monoton, maksimal 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pair_row(dist, va, vb):
    """Kolom CI dan p untuk distribusi selisih dalam fraksi; hasil dalam poin persen."""
    d = dist * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_raw, p_p1 = boot_p(d)
    cross = bool(lo <= 0 <= hi)
    return dict(A_persen=va * 100, B_persen=vb * 100, delta_poin_persen=(va - vb) * 100, ci95_bawah=lo, ci95_atas=hi,
                ci_melewati_nol=cross, p_dua_sisi=p_raw, p_dua_sisi_plus1=p_p1)

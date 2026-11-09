"""
EVALUASI 2 KELAS — langkah GPU (test split penuh, tanpa test clean).

Untuk setiap run runs_2kelas/<model>_2kelas yang selesai:
- best.pt, imgsz 512, conf 0.001, NMS IoU 0.6, batch 16, device 0, seed 0, deterministic.
- metrik Ultralytics (mAP50, mAP50-95, mAP75, P, R; AP50/AP50-95/AP75/P/R per kelas, citra & instance per kelas).
- params dan GFLOPs pada 512x512 (dihitung sebelum val, sebelum layer di-fuse).
- waktu per citra dari validator (preprocess / inference / postprocess, batch 16), 3 kali val -> median.
- puncak memori GPU (torch.cuda.max_memory_allocated / reserved) selama val pertama.
- prediksi dan GT per citra diambil langsung dari DetectionValidator.update_metrics (sama seperti
  scripts/02_error_analysis.py), termasuk matriks TP 10 IoU yang dipakai Ultralytics, untuk analisis lanjutan.

Dataset tidak diubah: test split di-mirror dengan hardlink ke eval_2kelas/_testset/ supaya labels.cache
Ultralytics ditulis di sana, bukan di dataset-2kelas/test/.

Output: eval_2kelas/_cache/<run>.pkl, eval_2kelas/tables/eval_ultralytics_raw.json, eval_2kelas/eval_01.log
"""
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset-2kelas"
RUNS = ROOT / "runs_2kelas"
OUT = ROOT / "eval_2kelas"
TESTSET = OUT / "_testset"
CACHE = OUT / "_cache"
TABLES = OUT / "tables"
SEED = 0
EVAL_ARGS = dict(split="test", imgsz=512, conf=0.001, iou=0.6, batch=16, device=0, half=False,
                 plots=False, save_json=False, verbose=True, deterministic=True, seed=SEED)
N_SPEED = 3  # jumlah val untuk median waktu inference
MODELS = [f"yolov12{v}" for v in "nsmlx"] + [f"yolov8{v}" for v in "nsmlx"]


def build_testset():
    """Mirror dataset-2kelas/test dengan hardlink (fallback: copy). Tidak menyentuh dataset asli."""
    src = DATA_DIR / "test"
    for sub in ("images", "labels"):
        d = TESTSET / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted((src / sub).iterdir()):
            t = d / f.name
            if not t.exists():
                try:
                    os.link(f, t)
                except OSError:
                    shutil.copy2(f, t)
    names = yaml.safe_load((DATA_DIR / "data.yaml").read_text())["names"]
    y = TESTSET / "data_test.yaml"
    y.write_text(yaml.safe_dump(dict(path=TESTSET.as_posix(), train="images", val="images", test="images",
                                     nc=len(names), names=names), allow_unicode=True, sort_keys=False))
    return y, names


def is_done(run_dir):
    import torch

    last = run_dir / "weights" / "last.pt"
    return last.exists() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", 0) == -1


def evaluate(model_name, data_yaml, names):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops
    from ultralytics.utils.torch_utils import get_flops, get_num_params, init_seeds

    run = f"{model_name}_2kelas"
    best = RUNS / run / "weights" / "best.pt"
    model = YOLO(str(best))
    mn = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if mn != list(names):
        raise RuntimeError(f"nama kelas model {mn} != data.yaml {names}")
    params = int(get_num_params(model.model))
    gflops = float(get_flops(model.model, imgsz=EVAL_ARGS["imgsz"]))

    store = {}
    orig = DetectionValidator.update_metrics

    def hooked(self, preds, batch):
        for si, pred in enumerate(preds):
            pb = self._prepare_batch(si, batch)
            p = {k: v.clone() for k, v in pred.items()}
            tp = self._process_batch(p, pb)["tp"]
            sp = self.scale_preds(p, pb)
            g_ori = ops.scale_boxes(pb["imgsz"], pb["bboxes"].clone(), pb["ori_shape"], ratio_pad=pb["ratio_pad"])
            store[Path(pb["im_file"]).name] = dict(
                im_file=str(pb["im_file"]), ori_shape=tuple(int(x) for x in pb["ori_shape"]),
                imgsz=tuple(int(x) for x in pb["imgsz"]),
                gt_cls=pb["cls"].cpu().numpy().reshape(-1).astype(int),
                gt_box=pb["bboxes"].cpu().numpy().reshape(-1, 4),           # xyxy, ruang input (letterbox 512)
                gt_box_ori=g_ori.cpu().numpy().reshape(-1, 4),              # xyxy, piksel citra asli
                pr_cls=p["cls"].cpu().numpy().reshape(-1).astype(int),
                pr_conf=p["conf"].cpu().numpy().reshape(-1),
                pr_box=p["bboxes"].cpu().numpy().reshape(-1, 4),            # xyxy, ruang input
                pr_box_ori=sp["bboxes"].cpu().numpy().reshape(-1, 4),       # xyxy, piksel citra asli
                tp=np.asarray(tp, dtype=bool).reshape(-1, 10))              # TP pada IoU 0.50:0.05:0.95
        return orig(self, preds, batch)

    speeds = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    DetectionValidator.update_metrics = hooked
    try:
        init_seeds(SEED, deterministic=True)
        m = model.val(data=str(data_yaml), project=str(OUT / "val"), name=run, exist_ok=True, **EVAL_ARGS)
    finally:
        DetectionValidator.update_metrics = orig
    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
    peak_res = torch.cuda.max_memory_reserved() / 2 ** 20
    speeds.append(dict(m.speed))
    for _ in range(N_SPEED - 1):
        init_seeds(SEED, deterministic=True)
        mm = model.val(data=str(data_yaml), project=str(OUT / "val"), name=f"{run}_speed", exist_ok=True,
                       **{**EVAL_ARGS, "verbose": False})
        speeds.append(dict(mm.speed))

    box = m.box
    present = [int(c) for c in m.ap_class_index]
    per_class = {}
    for c, cname in enumerate(names):
        r = dict(images=int(m.nt_per_image[c]), instances=int(m.nt_per_class[c]),
                 precision=None, recall=None, AP50=None, AP50_95=None, AP75=None)
        if c in present:
            i = present.index(c)
            p_, r_, ap50, ap = box.class_result(i)
            r.update(precision=float(p_), recall=float(r_), AP50=float(ap50), AP50_95=float(ap),
                     AP75=float(box.all_ap[i, 5]))
        per_class[cname] = r
    res = dict(model=model_name, run=run, best_pt=str(best), params=params, GFLOPs_512=gflops,
               mAP50=float(box.map50), mAP50_95=float(box.map), mAP75=float(box.map75),
               precision=float(box.mp), recall=float(box.mr), per_class=per_class,
               n_images_scored=len(store), speeds=speeds, peak_mem_alloc_MiB=peak_alloc,
               peak_mem_reserved_MiB=peak_res, device=torch.cuda.get_device_name(0))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{run}.pkl").write_bytes(pickle.dumps(dict(args=EVAL_ARGS, names=list(names), images=store)))
    del model
    torch.cuda.empty_cache()
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    data_yaml, names = build_testset()
    import torch
    import ultralytics

    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    raw_f = TABLES / "eval_ultralytics_raw.json"
    allres = json.loads(raw_f.read_text()) if raw_f.exists() else {}
    t0 = time.time()
    for mname in MODELS:
        rd = RUNS / f"{mname}_2kelas"
        if not rd.exists() or not is_done(rd):
            print(f"[SKIP] {mname}: run tidak ada / belum selesai")
            continue
        if mname in allres and (CACHE / f"{mname}_2kelas.pkl").exists():
            print(f"[SKIP] {mname}: sudah dievaluasi")
            continue
        print(f"[EVAL] {mname} ({time.time() - t0:.0f}s)", flush=True)
        r = evaluate(mname, data_yaml, names)
        allres[mname] = r
        raw_f.write_text(json.dumps(allres, indent=1))
        sp = np.median([s["inference"] for s in r["speeds"]])
        print(f"       mAP50 {r['mAP50']:.5f} mAP50-95 {r['mAP50_95']:.5f} mAP75 {r['mAP75']:.5f} | "
              f"{r['n_images_scored']} citra | inference {sp:.2f} ms/citra | peak {r['peak_mem_alloc_MiB']:.0f} MiB",
              flush=True)
    print(f"[SELESAI] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

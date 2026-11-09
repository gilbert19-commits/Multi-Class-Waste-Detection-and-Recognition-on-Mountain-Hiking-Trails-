"""LANGKAH 2 (GPU): evaluasi seragam best.pt di test dan valid, sekaligus menyimpan prediksi per gambar (Langkah 4).

Ultralytics membuang gambar yang labelnya mencampur poligon dan box (dianggap corrupt): 2 di test, 3 di valid. Supaya
evaluasi benar-benar memakai 550 dan 595 gambar, split di-mirror dengan hardlink ke hasil_2kelas/_data/, dan HANYA label
gambar campuran itu ditulis ulang (file baru, bukan hardlink) dengan poligon diubah ke kotak pembungkus, sama seperti yang
dilakukan Ultralytics untuk gambar yang seluruh labelnya poligon. dataset-2kelas/ tidak disentuh. Angka tanpa gambar
tersebut (548 / 592) dihitung ulang di s4_bootstrap.py untuk cross-check.

Confusion matrix: validator 8.4 memakai conf eksplisit dari argumen (0.001) untuk confusion matrix, jadi nilainya
diset ke 0.25 (default Ultralytics) di hook; ambang IoU memakai default process_batch (0.45).
"""
import copy
import json
import os
import pickle
import sys
import time

import numpy as np
import yaml

from hk_common import CACHE, DATA_DIR, EVAL_ARGS, MIRROR, MODELS, NAMES, OUT, RUNS, SEED, VALDIR, label

CM_CONF = 0.25
SPLIT_DIR = {"test": "test", "val": "valid"}
EXPECT = {"test": 550, "val": 595}


def build_mirror():
    fixed = []
    for sp in SPLIT_DIR.values():
        for sub in ("images", "labels"):
            (MIRROR / sp / sub).mkdir(parents=True, exist_ok=True)
        for f in sorted((DATA_DIR / sp / "images").iterdir()):
            t = MIRROR / sp / "images" / f.name
            if not t.exists():
                os.link(f, t)
        for f in sorted((DATA_DIR / sp / "labels").iterdir()):
            if f.suffix != ".txt":
                continue
            rows = [r.split() for r in f.read_text().splitlines() if r.strip()]
            t = MIRROR / sp / "labels" / f.name
            kinds = {len(r) == 5 for r in rows}
            if len(kinds) > 1:
                out = []
                for r in rows:
                    if len(r) == 5:
                        out.append(" ".join(r))
                        continue
                    xy = np.array(r[1:], float).reshape(-1, 2)
                    x1, y1, x2, y2 = xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()
                    out.append(f"{r[0]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
                t.write_text("\n".join(out) + "\n")
                fixed.append(dict(split=sp, file=f.name, baris=len(rows), baris_box=sum(len(r) == 5 for r in rows),
                                  baris_poligon=sum(len(r) != 5 for r in rows)))
            elif not t.exists():
                os.link(f, t)
    y = MIRROR / "data_eval.yaml"
    y.write_text(yaml.safe_dump(dict(path=MIRROR.as_posix(), train="valid/images", val="valid/images",
                                     test="test/images", nc=len(NAMES), names=NAMES), allow_unicode=True, sort_keys=False))
    import pandas as pd

    pd.DataFrame(fixed).to_csv(OUT / "label_campuran_diperbaiki.csv", index=False)
    return y


def evaluate(model_name, data_yaml, split):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import nms, ops
    from ultralytics.utils.torch_utils import get_flops, get_num_params, init_seeds

    run = f"{model_name}_2kelas"
    model = YOLO(str(RUNS / run / "weights" / "best.pt"))
    if list(model.names.values()) != NAMES:
        raise RuntimeError(f"nama kelas {run}: {model.names}")
    fused = copy.deepcopy(model.model).fuse(verbose=False)
    cplx = dict(params_tanpa_fuse=int(get_num_params(model.model)), GFLOPs_512_tanpa_fuse=float(get_flops(model.model, 512)),
                params_fused=int(get_num_params(fused)), GFLOPs_512_fused=float(get_flops(fused, 512)))
    del fused

    store, info = {}, {}
    orig_up, orig_nms = DetectionValidator.update_metrics, nms.non_max_suppression

    def spy_nms(prediction, *a, **k):
        # Panggilan dari AutoBackend.warmup (tensor acak 1x84x16) tidak membawa argumen end2end, jadi tidak dicatat
        if "end2end" in k:
            p0 = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
            info.setdefault("nms_end2end_arg", bool(k["end2end"]))
            info.setdefault("bentuk_output_model", list(p0.shape))
        return orig_nms(prediction, *a, **k)

    def hooked(self, preds, batch):
        info["validator"] = self
        self.confusion_matrix_conf = CM_CONF
        for si, pred in enumerate(preds):
            pb = self._prepare_batch(si, batch)
            p = {k: v.clone() for k, v in pred.items()}
            tp = self._process_batch(p, pb)["tp"]
            sp = self.scale_preds(p, pb)
            g_ori = ops.scale_boxes(pb["imgsz"], pb["bboxes"].clone(), pb["ori_shape"], ratio_pad=pb["ratio_pad"])
            store[os.path.basename(pb["im_file"])] = dict(
                im_file=str(pb["im_file"]), ori_shape=tuple(int(x) for x in pb["ori_shape"]),
                imgsz=tuple(int(x) for x in pb["imgsz"]),
                gt_cls=pb["cls"].cpu().numpy().reshape(-1).astype(int),
                gt_box=pb["bboxes"].cpu().numpy().reshape(-1, 4),       # xyxy, ruang input 512
                gt_box_ori=g_ori.cpu().numpy().reshape(-1, 4),          # xyxy, piksel gambar asli
                pr_cls=p["cls"].cpu().numpy().reshape(-1).astype(int),
                pr_conf=p["conf"].cpu().numpy().reshape(-1),
                pr_box=p["bboxes"].cpu().numpy().reshape(-1, 4),
                pr_box_ori=sp["bboxes"].cpu().numpy().reshape(-1, 4),
                tp=np.asarray(tp, dtype=bool).reshape(-1, 10))
        return orig_up(self, preds, batch)

    DetectionValidator.update_metrics = hooked
    nms.non_max_suppression = spy_nms
    try:
        init_seeds(SEED, deterministic=True)
        m = model.val(data=str(data_yaml), split=split, project=str(VALDIR / split), name=run, exist_ok=True,
                      plots=split == "test", **EVAL_ARGS)
    finally:
        DetectionValidator.update_metrics, nms.non_max_suppression = orig_up, orig_nms
    if len(store) != EXPECT[split]:
        raise RuntimeError(f"{run} {split}: {len(store)} gambar dinilai, seharusnya {EXPECT[split]}")

    box = m.box
    present = [int(c) for c in m.ap_class_index]
    per_class = {}
    for c, cname in enumerate(NAMES):
        i = present.index(c)
        p_, r_, ap50, ap = box.class_result(i)
        per_class[cname] = dict(images=int(m.nt_per_image[c]), instances=int(m.nt_per_class[c]), precision=float(p_),
                                recall=float(r_), AP50=float(ap50), AP75=float(box.all_ap[i, 5]), AP50_95=float(ap))
    v = info.pop("validator")
    res = dict(model=model_name, split=split, n_images=len(store), **cplx, mAP50=float(box.map50), mAP75=float(box.map75),
               mAP50_95=float(box.map), precision=float(box.mp), recall=float(box.mr), per_class=per_class,
               speed_batch16=dict(m.speed), end2end_validator=bool(v.end2end), end2end_model=bool(getattr(model.model, "end2end", False)),
               **info)
    if split == "test":
        cm = v.confusion_matrix.matrix.copy()  # baris = prediksi, kolom = ground truth, indeks nc = background
        n_pred = np.bincount(np.concatenate([x["pr_cls"][x["pr_conf"] > CM_CONF] for x in store.values()]), minlength=2)
        n_gt = np.bincount(np.concatenate([x["gt_cls"] for x in store.values()]), minlength=2)
        if not (cm[:2].sum(1) == n_pred).all() or not (cm[:, :2].sum(0) == n_gt).all():
            raise RuntimeError(f"{run}: confusion matrix tidak cocok dengan prediksi conf>{CM_CONF}: {cm} {n_pred} {n_gt}")
        res["confusion_matrix"] = cm.tolist()
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{run}_{split}.pkl").write_bytes(pickle.dumps(dict(args=EVAL_ARGS, names=NAMES, images=store)))
    del model
    torch.cuda.empty_cache()
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data_yaml = build_mirror()
    raw_f = OUT / "_eval_raw.json"
    allres = json.loads(raw_f.read_text()) if raw_f.exists() else {}
    t0 = time.time()
    for split in ("test", "val"):
        for m in MODELS:
            key = f"{m}|{split}"
            if key in allres and (CACHE / f"{m}_2kelas_{split}.pkl").exists():
                continue
            r = evaluate(m, data_yaml, split)
            allres[key] = r
            raw_f.write_text(json.dumps(allres, indent=1))
            print(f"[{time.time() - t0:6.0f}s] {split:4s} {label(m):9s} n={r['n_images']} mAP50={r['mAP50']:.5f} "
                  f"mAP50-95={r['mAP50_95']:.5f} mAP75={r['mAP75']:.5f} e2e={r['end2end_validator']} "
                  f"nms_arg={r.get('nms_end2end_arg')} output={r.get('bentuk_output_model')} params={r['params_fused']}",
                  flush=True)
    print("[SELESAI s2]", flush=True)


if __name__ == "__main__":
    main()

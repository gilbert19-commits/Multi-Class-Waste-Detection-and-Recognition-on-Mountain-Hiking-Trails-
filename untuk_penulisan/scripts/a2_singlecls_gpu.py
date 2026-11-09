"""VERIFIKASI b (GPU): model.val(single_cls=True) dengan setting evaluasi hasil_2kelas, untuk cek silang versi 3.

Prediksi per gambar yang keluar dari validator ikut disimpan supaya selisihnya bisa dijelaskan.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

from up_common import CACHE, EVAL_ARGS, MIRROR, MODELS, OUT, RUNS, SEED, label

VALDIR = OUT / "_val_singlecls"


def evaluate(m):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.torch_utils import init_seeds

    run = f"{m}_2kelas"
    model = YOLO(str(RUNS / run / "weights" / "best.pt"))
    store = {}
    orig_up = DetectionValidator.update_metrics

    def hooked(self, preds, batch):
        for si, pred in enumerate(preds):
            pb = self._prepare_batch(si, batch)
            store[os.path.basename(pb["im_file"])] = dict(conf=pred["conf"].cpu().numpy().reshape(-1), box=pred["bboxes"].cpu().numpy().reshape(-1, 4),
                                                          gt_cls=pb["cls"].cpu().numpy().reshape(-1).astype(int))
        return orig_up(self, preds, batch)

    DetectionValidator.update_metrics = hooked
    try:
        init_seeds(SEED, deterministic=True)
        r = model.val(data=str(MIRROR / "data_eval.yaml"), split="test", single_cls=True, project=str(VALDIR), name=run, exist_ok=True,
                      plots=False, **EVAL_ARGS)
    finally:
        DetectionValidator.update_metrics = orig_up
    if len(store) != 550:
        raise RuntimeError(f"{run}: {len(store)} gambar dinilai, seharusnya 550")
    n_obj = int(sum(len(x["gt_cls"]) for x in store.values()))
    n_cls = sorted({int(c) for x in store.values() for c in x["gt_cls"]})
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"singlecls_{m}.pkl").write_bytes(pickle.dumps(store))
    res = dict(model=label(m), n_gambar=len(store), n_objek=n_obj, kelas_gt=n_cls, end2end=bool(getattr(model.model, "end2end", False)),
               mAP50=float(r.box.map50), mAP75=float(r.box.map75), mAP50_95=float(r.box.map),
               n_prediksi=int(sum(len(x["conf"]) for x in store.values())))
    del model
    torch.cuda.empty_cache()
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw_f = OUT / "_singlecls_raw.json"
    OUT.mkdir(exist_ok=True)
    allres = json.loads(raw_f.read_text()) if raw_f.exists() else {}
    t0 = time.time()
    for m in MODELS:
        if m in allres and (CACHE / f"singlecls_{m}.pkl").exists():
            continue
        allres[m] = evaluate(m)
        raw_f.write_text(json.dumps(allres, indent=1))
        print(f"[{time.time() - t0:6.0f}s] {allres[m]}", flush=True)
    print("[SELESAI a2]", flush=True)


if __name__ == "__main__":
    main()

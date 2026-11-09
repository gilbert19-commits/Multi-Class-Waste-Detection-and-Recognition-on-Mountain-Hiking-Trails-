"""LANGKAH 3 (GPU): kecepatan semua model dalam satu proses. batch=1, imgsz=512, half=False, 550 gambar test, 3 ulangan.

Waktu dari validator (Profile dengan torch.cuda.synchronize) dibagi jumlah gambar. Warmup: AutoBackend.warmup dijalankan
10 iterasi per val. Ulangan bergiliran antar model (round robin) supaya perubahan kondisi GPU tidak menumpuk di satu model.
Dua mode: conf=0.001 (setting Langkah 2, utama) dan conf=0.25 (default prediksi Ultralytics), karena beban NMS bergantung
pada conf sedangkan permintaan tidak menyebut nilainya. Preprocess validator = pindah ke GPU + normalisasi; baca JPEG dan
letterbox terjadi di dataloader dan tidak terukur.
"""
import json
import sys
import time

import pandas as pd

from hk_common import MIRROR, MODELS, OUT, RUNS, VALDIR, label

N_REP, N_WARM = 3, 10
CONFS = (0.001, 0.25)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import AutoBackend

    orig = AutoBackend.warmup

    def warmup10(self, imgsz=(1, 3, 640, 640)):
        for _ in range(N_WARM):
            orig(self, imgsz)

    AutoBackend.warmup = warmup10
    rows = []
    t0 = time.time()
    for conf in CONFS:
        for rep in range(1, N_REP + 1):
            for m in MODELS:
                model = YOLO(str(RUNS / f"{m}_2kelas" / "weights" / "best.pt"))
                r = model.val(data=str(MIRROR / "data_eval.yaml"), split="test", imgsz=512, batch=1, conf=conf, iou=0.7,
                              half=False, augment=False, device=0, workers=4, plots=False, verbose=False,
                              project=str(VALDIR / "speed"), name="tmp", exist_ok=True)
                s = dict(r.speed)
                rows.append(dict(model=label(m), conf=conf, ulangan=rep, preprocess=s["preprocess"], inference=s["inference"],
                                 postprocess=s["postprocess"], n_images=int(r.seen) if hasattr(r, "seen") else None))
                del model, r
                torch.cuda.empty_cache()
                print(f"[{time.time() - t0:6.0f}s] conf={conf} ulangan {rep} {label(m):9s} pre={s['preprocess']:.3f} "
                      f"inf={s['inference']:.3f} post={s['postprocess']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "speed_ulangan.csv", index=False)
    d = pd.DataFrame(rows)
    d["total"] = d.preprocess + d.inference + d.postprocess
    d.to_csv(OUT / "speed_ulangan.csv", index=False)
    g = d.groupby(["model", "conf"], sort=False)
    out = g[["preprocess", "inference", "postprocess", "total"]].mean().reset_index()
    out["FPS"] = 1000 / out.total
    out["total_min"] = g.total.min().values
    out["total_max"] = g.total.max().values
    out["gpu"] = torch.cuda.get_device_name(0)
    out["batch"], out["imgsz"], out["half"], out["warmup_iterasi"], out["n_ulangan"] = 1, 512, False, N_WARM, N_REP
    out.to_csv(OUT / "speed.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(json.dumps({"selesai_detik": time.time() - t0}))
    print("[SELESAI s3]", flush=True)


if __name__ == "__main__":
    main()

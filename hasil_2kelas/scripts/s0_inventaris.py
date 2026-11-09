"""LANGKAH 0: inventaris run, setting tiap run, perbedaan setting, environment. Berhenti kalau ada run belum selesai."""
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import yaml

from hk_common import MODELS, OUT, ROOT, RUNS, best_epoch, family, label, read_results, simulate_stop, training_seconds

LOG = ROOT / "train_2kelas.log"
TIME = ROOT / "waktu_training_2kelas.txt"
KEYS = ["data", "epochs", "patience", "imgsz", "batch", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "mosaic", "close_mosaic", "mixup", "cutmix", "copy_paste",
        "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "scale", "translate", "degrees", "shear", "perspective", "seed",
        "deterministic", "amp", "pretrained", "cos_lr", "nbs", "workers", "cache", "rect", "iou", "max_det", "box",
        "cls", "dfl"]
TARGET = dict(epochs=200, patience=50, imgsz=512)
RE_OPT = re.compile(r"optimizer:\S*\s+(\w+)\(lr=([\d.e-]+), momentum=([\d.e-]+)\) with parameter groups "
                    r"(\d+) weight\(decay=[\d.e-]+\), (\d+) weight\(decay=([\d.e-]+)\), (\d+) bias")
RE_TRAIN = re.compile(r"\[INFO\] Train (\S+) dari awal \((\S+)\)")
RE_RESUME = re.compile(r"\[INFO\] Resume (\S+) dari")
RE_TRANSF = re.compile(r"Transferred (\d+/\d+) items from pretrained weights")
RE_AUTO = re.compile(r"AutoBatch:.*?[Uu]sing batch-size (\d+)")
RE_SESSION = re.compile(r"(\S+) \| awal training \(epoch (\d+)\) \| args\.batch=(-?\d+) \| batch dipakai=(\d+) \| "
                        r"imgsz efektif=(\d+)(?: \| workers=(\d+))?(?: \| epochs=(\d+))?(?: \| patience=(\d+))?")


def parse_log():
    ev = defaultdict(lambda: defaultdict(list))
    cur = None
    with open(LOG, "rb") as f:
        for raw in f:
            if not any(k in raw for k in (b"[INFO] Train", b"[INFO] Resume", b"optimizer:", b"Transferred", b"AutoBatch")):
                continue
            line = raw.decode("utf-8", "replace")
            m = RE_TRAIN.search(line)
            if m:
                cur = m.group(1)
                ev[cur].clear()  # run dibatalkan dengan nama sama: pakai awal yang terakhir
                ev[cur]["weight_awal"].append(m.group(2))
                continue
            m = RE_RESUME.search(line)
            if m:
                cur = m.group(1)
                continue
            if cur is None:
                continue
            m = RE_OPT.search(line)
            if m:
                ev[cur]["optimizer"].append(m.groups())
            m = RE_TRANSF.search(line)
            if m:
                ev[cur]["transfer"].append(m.group(1))
            m = RE_AUTO.search(line)
            if m:
                ev[cur]["autobatch"].append(m.group(1))
    return ev


def scan_all_args():
    """Semua args.yaml di project selain 15 run utama, dipakai untuk tabel run yang dikecualikan."""
    rows = []
    for r, ds, fs in os.walk(ROOT):
        ds[:] = [d for d in ds if d not in (".git", "hasil_2kelas", "_testset", "_cache")]
        if "args.yaml" not in fs:
            continue
        p = Path(r)
        rel = p.relative_to(ROOT).as_posix()
        if p.parent == RUNS and p.name.removesuffix("_2kelas") in MODELS:
            continue
        a = yaml.safe_load((p / "args.yaml").read_text(encoding="utf-8", errors="replace"))
        data = str(a.get("data"))
        res = p / "results.csv"
        n_ep = len(read_results(res)) if res.exists() else None
        if rel.startswith("eval_2kelas/"):
            why = "salinan args.yaml dari run utama, bukan run"
        elif "dataset-2kelas" in data:
            why = "dataset 2 kelas, run dibatalkan dan digantikan runs_2kelas/yolov12s_2kelas"
        elif "TA-4" in data:
            why = "dataset 4 kelas (TA-4/data.yaml, nc=4)"
        else:
            why = "bukan dataset_2class (lihat kolom data)"
        rows.append(dict(path=rel, data=data, model=Path(str(a.get("model"))).name, epochs=a.get("epochs"),
                         epoch_tercatat=n_ep, imgsz=a.get("imgsz"), ada_best_pt=(p / "weights" / "best.pt").exists(),
                         alasan_dikecualikan=why))
    return pd.DataFrame(rows).sort_values("path")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    belum = []
    cks = {}
    for m in MODELS:
        rd = RUNS / f"{m}_2kelas"
        last, best = rd / "weights" / "last.pt", rd / "weights" / "best.pt"
        if not best.exists() or not last.exists():
            belum.append(f"{m}: bobot tidak ada")
            continue
        if torch.load(last, map_location="cpu", weights_only=False).get("epoch", 0) != -1:
            belum.append(f"{m}: last.pt belum final")
        cks[m] = torch.load(best, map_location="cpu", weights_only=False)
    if belum:
        print("[BERHENTI] run belum selesai:", belum)
        sys.exit(1)

    ev = parse_log()
    sessions = defaultdict(list)
    for line in TIME.read_text(encoding="utf-8", errors="replace").splitlines():
        g = RE_SESSION.search(line)
        if g:
            if g.group(2) == "1":  # mulai dari awal: abaikan run dibatalkan dengan nama sama
                sessions[g.group(1)].clear()
            sessions[g.group(1)].append(g.groups()[1:])

    inv, sett = [], []
    for m in MODELS:
        name = f"{m}_2kelas"
        rd = RUNS / name
        a = yaml.safe_load((rd / "args.yaml").read_text())
        if "dataset-2kelas" not in str(a["data"]):
            print(f"[BERHENTI] {name} tidak memakai dataset 2 kelas: {a['data']}")
            sys.exit(1)
        d = read_results(rd / "results.csv")
        secs, n_sess = training_seconds(d)
        ep, be = int(d["epoch"].iloc[-1]), best_epoch(d)
        ck = cks[m]
        batch = int(ck["train_args"]["batch"])
        nbs = int(a.get("nbs", 64))
        acc = max(round(nbs / batch), 1)
        e = ev.get(name, {})
        opt = sorted(set(e.get("optimizer", [])))
        ses = sessions.get(name, [])

        def per_sesi(i):
            return ";".join(f"e{s[0]}:{s[i] if s[i] is not None else 'TIDAK TERCATAT'}" for s in ses)

        inv.append(dict(family=family(m), skala=m[-1], model=label(m), path=rd.relative_to(ROOT).as_posix(),
                        epoch_dijalankan=ep, best_epoch=be, early_stopping=ep < int(a["epochs"]),
                        lama_training_jam=secs / 3600, jumlah_sesi=n_sess,
                        val_mAP50_best=float(d.loc[d.epoch == be, "metrics/mAP50(B)"].iloc[0]),
                        val_mAP50_95_best=float(d.loc[d.epoch == be, "metrics/mAP50-95(B)"].iloc[0]),
                        berhenti_jika_patience_50=simulate_stop(d, 50), berhenti_jika_patience_100=simulate_stop(d, 100),
                        versi_ultralytics_ckpt=ck.get("version")))
        row = dict(family=family(m), skala=m[-1], model=label(m))
        row.update({k: a.get(k) for k in KEYS})
        row["data"] = Path(str(a["data"])).parent.name + "/" + Path(str(a["data"])).name
        row.update(batch_args_yaml=a.get("batch"), batch=batch, batch_per_sesi=per_sesi(2), akumulasi_gradien=acc,
                   weight_decay_efektif=a["weight_decay"] * batch * acc / nbs,
                   patience_per_sesi=per_sesi(6), workers_per_sesi=per_sesi(4),
                   optimizer_sebenarnya="; ".join(f"{o[0]}(lr={o[1]}, momentum={o[2]})" for o in
                                                  sorted({x[:3] for x in opt})) or "TIDAK TERSEDIA",
                   weight_decay_log=";".join(sorted({o[5] for o in opt})) or "TIDAK TERSEDIA",
                   weight_awal=(e.get("weight_awal") or ["TIDAK TERSEDIA"])[0],
                   transfer_pretrained=";".join(sorted(set(e.get("transfer", [])))) or "TIDAK TERSEDIA",
                   autobatch_log=";".join(e.get("autobatch", [])) or "tidak ada",
                   warmup_bias_lr_ckpt=ck["train_args"].get("warmup_bias_lr"))  # args.yaml ditulis sebelum build_optimizer
        beda = [f"{k}={a.get(k)} (target {v})" for k, v in TARGET.items() if a.get(k) != v]
        row["menyimpang_dari_target"] = "; ".join(beda) or "tidak"
        sett.append(row)
    inv = pd.DataFrame(inv)
    sett = pd.DataFrame(sett)
    inv.to_csv(OUT / "runs_inventory.csv", index=False)
    sett.to_csv(OUT / "settings_all_runs.csv", index=False)

    # Semua kunci args.yaml yang nilainya tidak seragam, dipindai otomatis supaya tidak ada yang terlewat
    allargs = {m: yaml.safe_load((RUNS / f"{m}_2kelas" / "args.yaml").read_text()) for m in MODELS}
    skip = {"model", "name", "save_dir", "project"}
    rows = []
    for k in sorted({k for a in allargs.values() for k in a} - skip):
        vals = defaultdict(list)
        for m, a in allargs.items():
            vals[str(a.get(k))].append(label(m))
        if len(vals) > 1:
            rows.append(dict(sumber="args.yaml", setting=k, nilai_dan_model=" | ".join(
                f"{v}: {','.join(ms)}" for v, ms in sorted(vals.items()))))
    for k in ("batch", "akumulasi_gradien", "weight_decay_efektif", "patience_per_sesi", "workers_per_sesi",
              "optimizer_sebenarnya", "transfer_pretrained"):
        vals = defaultdict(list)
        for _, r in sett.iterrows():
            vals[str(r[k])].append(r["model"])
        if len(vals) > 1:
            rows.append(dict(sumber="log/best.pt", setting=k, nilai_dan_model=" | ".join(
                f"{v}: {','.join(ms)}" for v, ms in sorted(vals.items()))))
    pd.DataFrame(rows).to_csv(OUT / "settings_berbeda.csv", index=False)
    scan_all_args().to_csv(OUT / "runs_dikecualikan.csv", index=False)

    import ultralytics

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    (OUT / "environment.txt").write_text("\n".join([
        f"ultralytics: {ultralytics.__version__}", f"torch: {torch.__version__}",
        f"CUDA (torch.version.cuda): {torch.version.cuda}", f"cudnn: {torch.backends.cudnn.version()}",
        f"GPU: {torch.cuda.get_device_name(0)}", f"nvidia-smi: {smi}",
        f"python: {platform.python_version()} ({sys.executable})", f"OS: {platform.platform()}",
        f"versi ultralytics di checkpoint best.pt: {sorted({str(c.get('version')) for c in cks.values()})}", ""]),
        encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 100)
    print(inv[["model", "epoch_dijalankan", "best_epoch", "early_stopping", "lama_training_jam", "jumlah_sesi",
               "berhenti_jika_patience_50", "berhenti_jika_patience_100"]].to_string(index=False))
    print(sett[["model", "batch", "akumulasi_gradien", "batch_per_sesi", "patience_per_sesi", "workers_per_sesi",
                "optimizer_sebenarnya", "weight_awal", "transfer_pretrained", "autobatch_log",
                "menyimpang_dari_target"]].to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print(scan_all_args()[["path", "epochs", "epoch_tercatat", "ada_best_pt", "alasan_dikecualikan"]].to_string(index=False))
    print((OUT / "environment.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

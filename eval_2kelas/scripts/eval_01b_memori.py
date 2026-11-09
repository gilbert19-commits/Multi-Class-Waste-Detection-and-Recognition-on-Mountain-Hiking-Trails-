"""
Puncak memori GPU saat inference, diukur ulang di proses Python baru per model (eval_01_infer.py
mengevaluasi 10 model dalam satu proses sehingga sisa alokasi model sebelumnya bisa ikut terhitung).
Satu kali model.val dengan argumen yang sama dengan eval_01_infer.py. Hasil ditulis ke
eval_2kelas/tables/eval_ultralytics_raw.json: peak_mem_alloc_MiB / peak_mem_reserved_MiB (proses terpisah);
nilai lama disimpan sebagai *_proses_bersama.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "eval_2kelas" / "tables" / "eval_ultralytics_raw.json"


def child(model_name):
    import torch
    from ultralytics import YOLO

    sys.path.insert(0, str(Path(__file__).parent))
    from eval_01_infer import EVAL_ARGS, OUT, RUNS, TESTSET

    model = YOLO(str(RUNS / f"{model_name}_2kelas" / "weights" / "best.pt"))
    torch.cuda.reset_peak_memory_stats()
    model.val(data=str(TESTSET / "data_test.yaml"), project=str(OUT / "val"), name=f"{model_name}_2kelas_mem",
              exist_ok=True, **{**EVAL_ARGS, "verbose": False})
    print("RESULT " + json.dumps(dict(alloc=torch.cuda.max_memory_allocated() / 2 ** 20,
                                      reserved=torch.cuda.max_memory_reserved() / 2 ** 20)))


def main():
    raw = json.loads(RAW.read_text())
    for m in raw:
        p = subprocess.run([sys.executable, __file__, m], capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(f"[GAGAL] {m}\n{p.stderr[-2000:]}")
            continue
        r = json.loads(line[0][7:])
        raw[m].setdefault("peak_mem_alloc_MiB_proses_bersama", raw[m]["peak_mem_alloc_MiB"])
        raw[m].setdefault("peak_mem_reserved_MiB_proses_bersama", raw[m]["peak_mem_reserved_MiB"])
        raw[m]["peak_mem_alloc_MiB"], raw[m]["peak_mem_reserved_MiB"] = r["alloc"], r["reserved"]
        print(f"{m}: alloc {r['alloc']:.1f} MiB, reserved {r['reserved']:.1f} MiB "
              f"(proses bersama: {raw[m]['peak_mem_alloc_MiB_proses_bersama']:.1f})", flush=True)
        RAW.write_text(json.dumps(raw, indent=1))


if __name__ == "__main__":
    child(sys.argv[1]) if len(sys.argv) > 1 else main()

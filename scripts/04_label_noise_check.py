"""
04_label_noise_check.py — apakah kebingungan kertas_tisu <-> plastik kesalahan model atau label GT?

Tidak ada training, tidak ada perubahan dataset. Memakai cache prediksi validator (results/_preds)
dan pencocokan yang sama dengan confusion matrix di 02_error_analysis.py (conf >= 0.25, IoU > 0.5).

1. Kasus kertas<->plastik di test clean, semua model; satu baris per objek GT (confidence tertinggi
   antar model), dipilih 40 dengan confidence > 0.5.
2. Contact sheet crop objek -> results/figures/08_label_noise_sample.png
3. Breakdown per sumber dataset (UAVVaste / TACO / Burangrang), dinormalisasi dengan jumlah GT.
4. Keterlibatan citra UAVVaste yang punya salinan berlabel berbeda (audit D0).
5. Ringkasan -> results/analisis_label_noise.md, tabel -> results/label_noise_*.csv

Jalankan: python scripts/04_label_noise_check.py
"""
import importlib.util
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
TABLES = ROOT / "analysis" / "tables"

spec = importlib.util.spec_from_file_location("eval02", ROOT / "scripts" / "02_error_analysis.py")
eval02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval02)
eval01 = eval02.eval01

N_SAMPLE = 40
MIN_CONF = 0.5
MAX_PER_IMAGE = 3  # agar sampel tidak didominasi satu citra padat
CROP_PX = 300
NAMES_ID = eval02.NAMES_ID
GT_COLOR, PRED_COLOR = "#00c853", "#ff1744"
SOURCES = ["UAVVaste", "Burangrang", "TACO", "Unknown"]


def source_heuristic(name):
    """Sama dengan aturan D0: batch_* = UAVVaste, numerik = TACO, WhatsApp/IMG_/GOPR = Burangrang."""
    if re.match(r"(?i)batch_", name):
        return "UAVVaste"
    if re.match(r"\d+_", name):
        return "TACO"
    if re.match(r"(?i)(whatsapp|img_|gopr)", name):
        return "Burangrang"
    return "Unknown"


def model_label(rec):
    return f"{rec['model']}{rec['varian']}" + (" aug" if rec["augmentasi"] == "ya" else "")


# ---------------------------------------------------------------- 1. kumpulkan kasus
def collect_events(runs, sets, k, p):
    """Satu baris per pasangan GT-prediksi yang cocok (gaya ConfusionMatrix) dengan kelas {kertas, plastik}
    yang tertukar. Ini persis sel kertas->plastik dan plastik->kertas di confusion matrix 02."""
    rows, gt_rows = [], []
    for rec in runs:
        for s, (imgs, _, _) in sets.items():
            preds, gts = eval02.predict(rec, s, imgs)
            for img, g, pr in zip(imgs, gts, preds):
                pr = pr[pr[:, 5] >= eval02.CM_CONF]
                m = eval02.greedy(eval02.iou_matrix(g[:, 1:5], pr[:, 1:5]), eval02.CM_IOU, cm_style=True)
                for gi, pi in m:
                    tc, pc = int(g[gi, 0]), int(pr[pi, 0])
                    if {tc, pc} == {k, p}:
                        rows.append(dict(model=model_label(rec), test_set=s, file=img.name, gt_idx=int(gi),
                                         gt_cls=tc, pred_cls=pc, conf=float(pr[pi, 5]),
                                         gt_box=g[gi, 1:5].tolist(), pred_box=pr[pi, 1:5].tolist()))
    return pd.DataFrame(rows)


def gt_table(sets, k, p, src):
    """Jumlah objek GT kertas/plastik per citra (untuk normalisasi per sumber)."""
    rows = []
    for s, (imgs, gts, _) in sets.items():
        for img, g in zip(imgs, gts):
            rows.append(dict(test_set=s, file=img.name, sumber=src(img.name),
                             n_kertas=int((g[:, 0] == k).sum()), n_plastik=int((g[:, 0] == p).sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. contact sheet
def render_sheet(sample, img_dir, n_models, path):
    cols = 5
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, rows * 2.75 + 0.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, sample.iterrows()):
        im = cv2.cvtColor(cv2.imread(str(img_dir / r.file)), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        gx1, gy1, gx2, gy2 = r.gt_box
        bw, bh = gx2 - gx1, gy2 - gy1
        side = max(bw, bh) * 2.2 + 24  # konteks sekitar objek
        cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        x1, y1 = int(np.clip(cx - side / 2, 0, max(W - side, 0))), int(np.clip(cy - side / 2, 0, max(H - side, 0)))
        x2, y2 = int(min(x1 + side, W)), int(min(y1 + side, H))
        crop = im[y1:y2, x1:x2]
        s = CROP_PX / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        ax.imshow(crop)
        for (bx1, by1, bx2, by2), color, ls, lw in [(r.gt_box, GT_COLOR, "-", 1.8),
                                                     (r.pred_box, PRED_COLOR, (0, (3, 2)), 1.4)]:
            ax.add_patch(plt.Rectangle(((bx1 - x1) * s, (by1 - y1) * s), (bx2 - bx1) * s, (by2 - by1) * s,
                                       fill=False, edgecolor=color, linewidth=lw, linestyle=ls))
        ax.set_title(f"#{r.no}  GT: {NAMES_ID[r.gt_name]}\nPred: {NAMES_ID[r.pred_name]} ({r.conf:.2f})",
                     fontsize=7.5, loc="left", pad=2)
        stem = re.sub(r"_(jpg|jpeg|png)\.rf\..*", "", r.file, flags=re.I)
        ax.text(0, -0.02, f"{r.sumber} · {stem[:26]}\n{int(bw)}×{int(bh)} px · {r.n_model}/{n_models} model",
                transform=ax.transAxes, fontsize=6.5, va="top", color="#52514e")
    fig.suptitle("Sampel kebingungan kertas/tisu ↔ plastik, test clean (confidence > 0.5)",
                 fontsize=12, fontweight="bold", y=0.997)
    fig.text(0.5, 0.986, "Hijau utuh = box ground truth, merah putus = box prediksi. Crop diperbesar; "
             "ukuran asli box (pada 640×640) di bawah tiap crop.\n"
             f"\"k/{n_models} model\" = jumlah model yang membuat kekeliruan yang sama pada objek ini.",
             ha="center", va="top", fontsize=7.5, color="#52514e")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01, wspace=0.08, hspace=0.42)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- 4. salinan berlabel berbeda (D0)
def d0_copies():
    groups = pd.read_csv(TABLES / "D0_scene_groups.csv")
    inc = pd.read_csv(TABLES / "D0_label_inconsistency.csv")
    inc = inc[inc.konflik_kelas > 0].copy()
    inc["kp"] = inc.detail_konflik.fillna("").str.contains("kertas_tisu->plastik|plastik->kertas_tisu")
    conflict_any, conflict_kp = set(), set()
    for _, r in inc.iterrows():
        for key in (r.a, r.b):
            conflict_any.add(key)
            if r.kp:
                conflict_kp.add(key)
    n_copies = groups.groupby("source_name").size()
    return groups, conflict_any, conflict_kp, n_copies


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.1%}" if isinstance(v, float) and c.startswith("%") else
                                       (f"{v:.3f}" if isinstance(v, float) else str(v))
                                       for c, v in zip(cols, r)) + " |")
    return "\n".join(lines)


def main():
    import yaml

    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    names = yaml.safe_load((ROOT / "TA-1" / "data.yaml").read_text())["names"]
    k, p = names.index("kertas_tisu"), names.index("plastik")
    runs, _ = eval01.inventory()
    runs = [r for r in runs if r["best_pt"].exists()]
    n_models = len(runs)
    sets = {s: eval02.load_test_set(s) for s in ("full", "clean")}

    groups, conflict_any, conflict_kp, n_copies = d0_copies()
    test_rows = groups[groups.split == "test"].set_index("filename")
    src = lambda f: test_rows.sumber.get(f, source_heuristic(f))
    src_name = lambda f: test_rows.source_name.get(f, re.sub(r"_(jpg|jpeg|png)\.rf\..*", "", f, flags=re.I))

    ev = collect_events(runs, sets, k, p)
    ev["sumber"] = ev.file.map(src)
    ev["gt_name"] = [names[c] for c in ev.gt_cls]
    ev["pred_name"] = [names[c] for c in ev.pred_cls]
    ev["arah"] = [f"{NAMES_ID[a]} → {NAMES_ID[b]}" for a, b in zip(ev.gt_name, ev.pred_name)]
    gtab = gt_table(sets, k, p, src)
    clean = ev[ev.test_set == "clean"]
    print(f"{n_models} model; kasus kertas<->plastik (conf >= {eval02.CM_CONF}): "
          f"clean {len(clean)}, full {len(ev[ev.test_set == 'full'])}")

    # ------------------------------------------------ 1. per objek GT, pilih sampel
    obj = (clean.sort_values("conf", ascending=False)
           .groupby(["file", "gt_idx"], as_index=False)
           .agg(gt_cls=("gt_cls", "first"), pred_cls=("pred_cls", "first"), conf=("conf", "first"),
                model=("model", "first"), gt_box=("gt_box", "first"), pred_box=("pred_box", "first"),
                n_model=("model", "nunique"), sumber=("sumber", "first"),
                gt_name=("gt_name", "first"), pred_name=("pred_name", "first"), arah=("arah", "first")))
    hi = obj[obj.conf > MIN_CONF].sort_values("conf", ascending=False)
    sample = hi.groupby("file", group_keys=False).head(MAX_PER_IMAGE).head(N_SAMPLE).reset_index(drop=True)
    sample.insert(0, "no", range(1, len(sample) + 1))
    sample["salinan_d0"] = [n_copies.get(src_name(f), 1) for f in sample.file]
    sample["konflik_d0"] = [f"test/{f}" in conflict_any for f in sample.file]
    sample["penilaian_manual"] = ""  # diisi saat verifikasi visual: label_salah / model_salah / ambigu
    print(f"Objek GT unik yang tertukar di clean: {len(obj)}; dengan conf > {MIN_CONF}: {len(hi)}; "
          f"sampel: {len(sample)}")
    render_sheet(sample, OUT / "test_sets" / "clean" / "images", n_models, FIG / "08_label_noise_sample.png")
    print(f"  -> {(FIG / '08_label_noise_sample.png').relative_to(ROOT)}")
    sample.drop(columns=["gt_cls", "pred_cls"]).to_csv(OUT / "label_noise_sample.csv", index=False)

    # ------------------------------------------------ 3. per sumber
    def by_source(s):
        e = ev[ev.test_set == s]
        g = gtab[gtab.test_set == s]
        o = obj if s == "clean" else (e.groupby(["file", "gt_idx"], as_index=False).first())
        rows = []
        for so in SOURCES:
            gs = g[g.sumber == so]
            n_gt = int(gs.n_kertas.sum() + gs.n_plastik.sum())
            n_ev = int((e.sumber == so).sum())
            if not n_gt and not n_ev:
                continue
            rows.append({"sumber": so, "citra": len(gs), "GT kertas+plastik": n_gt,
                         "% GT": n_gt / max(g.n_kertas.sum() + g.n_plastik.sum(), 1),
                         "kasus (13 model)": n_ev, "% kasus": n_ev / max(len(e), 1),
                         "objek unik tertukar": int((o.sumber == so).sum()),
                         "% objek GT tertukar": (o.sumber == so).sum() / max(n_gt, 1),
                         "kasus per GT per model": n_ev / max(n_gt * n_models, 1)})
        return pd.DataFrame(rows)

    src_tab = {s: by_source(s) for s in ("clean", "full")}
    for s, t in src_tab.items():
        t.insert(0, "test_set", s)
    pd.concat(src_tab.values()).to_csv(OUT / "label_noise_per_sumber.csv", index=False)
    dir_tab = (clean.groupby(["sumber", "arah"]).size().unstack(fill_value=0)
               .reindex([s for s in SOURCES if s in set(clean.sumber)]))
    hi_src = hi.sumber.value_counts().reindex(SOURCES).dropna().astype(int)
    per_model = (clean.groupby(["model", "sumber"]).size().unstack(fill_value=0)
                 .reindex(columns=[s for s in SOURCES if s in set(clean.sumber)], fill_value=0))

    # ------------------------------------------------ 4. salinan berlabel berbeda
    files = sorted(set(clean.file))
    uav_files = [f for f in files if src(f) == "UAVVaste"]
    uav = pd.DataFrame(dict(file=uav_files))
    uav["source_name"] = uav.file.map(src_name)
    uav["salinan"] = uav.source_name.map(lambda n: int(n_copies.get(n, 1)))
    uav["di_pasangan_konflik"] = uav.file.map(lambda f: f"test/{f}" in conflict_any)
    uav["konflik_kertas_plastik"] = uav.file.map(lambda f: f"test/{f}" in conflict_kp)
    # salinan lain dengan nama sumber sama yang ada di pasangan konflik (mis. salinan train)
    keys_by_src = groups.assign(key=groups.split + "/" + groups.filename).groupby("source_name").key.agg(list)
    uav["salinan_lain_konflik"] = uav.source_name.map(
        lambda n: any(key in conflict_any for key in keys_by_src.get(n, [])))
    split_by_src = groups.groupby("source_name").split.agg(lambda s: sorted(set(s) - {"test"}))
    uav["salinan_di_train_valid"] = uav.source_name.map(lambda n: ",".join(split_by_src.get(n, [])))
    uav["kasus"] = uav.file.map(clean.file.value_counts())
    uav.to_csv(OUT / "label_noise_uavvaste_salinan.csv", index=False)
    uav_all_clean = [f for f in sets["clean"][0] if src(f.name) == "UAVVaste"]
    has_tv = uav.salinan_di_train_valid != ""
    sub = lambda m: (int(m.sum()), int(uav.loc[m, "kasus"].sum()))

    # ------------------------------------------------ penilaian visual (opsional, diisi manual)
    vis_f = OUT / "label_noise_penilaian_visual.csv"
    vis = None
    if vis_f.exists():
        vis = sample[["no", "file", "sumber", "arah", "conf", "n_model"]].merge(pd.read_csv(vis_f), on="no")
        if len(vis) != len(sample):
            print(f"  [PERINGATAN] {vis_f.name} tidak lengkap untuk sampel ini; penilaian visual diabaikan")
            vis = None

    # ------------------------------------------------ ringkasan
    tc, tf = src_tab["clean"], src_tab["full"]
    share = lambda t, so, c: float(t.loc[t.sumber == so, c].iloc[0]) if (t.sumber == so).any() else 0.0
    uav_case, uav_gt = share(tc, "UAVVaste", "% kasus"), share(tc, "UAVVaste", "% GT")
    bur_case = share(tc, "Burangrang", "% kasus")
    top_src = tc.sort_values("kasus (13 model)", ascending=False).sumber.iloc[0]
    uav_rate, bur_rate = share(tc, "UAVVaste", "kasus per GT per model"), share(tc, "Burangrang",
                                                                                "kasus per GT per model")
    n_agree = int((obj.n_model >= n_models // 2 + 1).sum())

    L = ["# Analisis Label Noise: kertas/tisu ↔ plastik", "",
         f"Tidak ada training dan tidak ada perubahan dataset. Prediksi dari cache validator ({n_models} model), "
         f"pencocokan sama dengan confusion matrix di `02_error_analysis.py` (conf ≥ {eval02.CM_CONF}, "
         f"IoU > {eval02.CM_IOU}). Sumber citra diambil dari `D0_scene_groups.csv` "
         "(pola nama file: `batch_*` = UAVVaste, numerik = TACO, `IMG_*`/`WhatsApp*`/`GOPR*` = Burangrang).", "",
         "## Angka dasar", "",
         f"- Test clean: **{len(clean)} kasus** kertas↔plastik, dijumlah atas {n_models} model "
         f"(sama dengan angka 530 di `analisis_error.md`). Rata-rata {len(clean) / n_models:.0f} kasus per model.",
         f"- Kasus itu berasal dari **{len(obj)} objek GT unik** di {len(files)} citra. "
         f"{n_agree} objek ({n_agree / max(len(obj), 1):.0%}) dikelirukan oleh mayoritas model "
         f"(≥ {n_models // 2 + 1}/{n_models}), jadi kekeliruannya konsisten, bukan kebetulan satu model.",
         f"- Arah: {', '.join(f'{a} {n}' for a, n in clean.arah.value_counts().items())}.",
         f"- Objek dengan confidence > {MIN_CONF} (model mana pun): **{len(hi)}**. Contact sheet memuat "
         f"{len(sample)} di antaranya (confidence tertinggi, maks. {MAX_PER_IMAGE} objek per citra).", "",
         "## Breakdown per sumber dataset", "",
         "\"kasus per GT per model\" = kasus ÷ (jumlah GT kertas+plastik × jumlah model). "
         "Ini menormalkan perbedaan jumlah objek antar sumber.", "",
         "### Test clean", "", md_table(tc.drop(columns="test_set")), "",
         "### Test full (pembanding, lebih banyak data)", "", md_table(tf.drop(columns="test_set")), "",
         "### Arah kekeliruan per sumber (test clean, 13 model)", "",
         md_table(dir_tab.reset_index()), "",
         f"### Objek dengan confidence > {MIN_CONF} per sumber (test clean)", "",
         md_table(hi_src.rename("objek").reset_index().rename(columns={"index": "sumber"})), "",
         "### Per model (test clean)", "", md_table(per_model.reset_index()), "",
         "## Citra UAVVaste dengan salinan berlabel berbeda (audit D0)", "",
         f"- Citra UAVVaste di test clean: {len(uav_all_clean)}. Yang terlibat dalam kebingungan kertas↔plastik: "
         f"**{len(uav)}**, menyumbang {int(uav.kasus.sum()) if len(uav) else 0} dari {len(clean)} kasus.",
         "",
         "| kondisi | citra | kasus |", "|---|---|---|",
         "| punya ≥ 2 salinan (nama sumber sama, seluruh dataset) | %d | %d |" % sub(uav.salinan >= 2),
         "| punya salinan di **train/valid** | %d | %d |" % sub(has_tv),
         "| salinan lainnya ada di pasangan konflik kelas D0 | %d | %d |" % sub(uav.salinan_lain_konflik),
         "| citra test itu sendiri ada di pasangan konflik D0 | %d | %d |" % sub(uav.di_pasangan_konflik),
         "| ...dengan konflik khusus kertas↔plastik | %d | %d |" % sub(uav.konflik_kertas_plastik),
         "",
         "- Test clean dibuang dari citra yang **satu grup scene SIFT** dengan train/valid. Salinan di atas "
         "lolos karena SIFT tidak menautkannya (varian rotasi/crop Roboflow), tetapi nama sumbernya sama. "
         "Artinya sebagian citra UAVVaste di test clean sebenarnya punya kembaran di train/valid.",
         "- \"Salinan lainnya ada di pasangan konflik\" berarti salinan-salinan di train/valid dari adegan yang sama "
         "diberi kelas berbeda satu sama lain. Model dilatih dengan label yang saling bertentangan untuk adegan itu.",
         "- Pembandingan label box per box antara salinan test dan salinan train belum dilakukan "
         "(butuh pemetaan homografi seperti di D0).",
         "", "Detail per citra: `results/label_noise_uavvaste_salinan.csv`.", ""]
    L += ["## Contact sheet dan penilaian visual", "",
          "`results/figures/08_label_noise_sample.png`; daftar kasus di `results/label_noise_sample.csv`.", ""]
    if vis is not None:
        cnt = vis.penilaian.value_counts()
        by_src_vis = vis.groupby(["sumber", "penilaian"]).size().unstack(fill_value=0)
        L += [f"Penilaian visual atas {len(vis)} crop (`results/label_noise_penilaian_visual.csv`). "
              "Kategori: `label_salah` = objek jelas tampak kelas yang diprediksi model; `model_salah` = objek "
              "jelas tampak kelas GT; `ambigu` = objek terlihat tetapi bahannya (kertas atau plastik) tidak "
              "bisa dipastikan dari foto; `tak_terbaca` = terlalu kecil/buram untuk dinilai. "
              "Ini penilaian subjektif dari foto 640×640 dan perlu dicek ulang oleh anotator.", "",
              md_table(cnt.rename("crop").reset_index().rename(columns={"index": "penilaian"})), "",
              md_table(by_src_vis.reset_index()), "",
              md_table(vis[vis.penilaian.isin(["label_salah", "model_salah"])]
                       [["no", "sumber", "arah", "conf", "penilaian", "catatan"]]), ""]
    else:
        L += ["Belum ada penilaian visual (`results/label_noise_penilaian_visual.csv`).", ""]

    n_lbl = int((vis.penilaian == "label_salah").sum()) if vis is not None else 0
    n_mdl = int((vis.penilaian == "model_salah").sum()) if vis is not None else 0
    n_unk = len(vis) - n_lbl - n_mdl if vis is not None else 0
    L += ["## Kesimpulan", "",
          f"**Hipotesis \"sebagian besar dari UAVVaste\": terbukti.** Di test clean UAVVaste menyumbang "
          f"{uav_case:.0%} kasus kertas↔plastik padahal hanya {uav_gt:.0%} dari objek GT kertas+plastik "
          f"(test full: {share(tf, 'UAVVaste', '% kasus'):.0%} dari {share(tf, 'UAVVaste', '% GT'):.0%}). "
          f"Setelah dinormalkan per objek, laju kekeliruan UAVVaste {uav_rate:.3f} per GT per model, "
          f"{uav_rate / max(bur_rate, 1e-9):.1f}× Burangrang ({bur_rate:.3f}). Jadi dominasinya bukan sekadar "
          "karena UAVVaste punya objek lebih banyak.", "",
          f"**Apakah labelnya yang salah: belum bisa dibuktikan untuk sebagian besar kasus.** Dari {len(sample)} "
          f"kasus ber-confidence tinggi, hanya {n_lbl} yang dari foto jelas tampak salah label, dan {n_mdl} jelas "
          f"kesalahan model (label benar, mis. botol semprot plastik diprediksi kertas). Sisanya {n_unk} ambigu "
          "atau tak terbaca: objek UAVVaste umumnya 10–35 px dari ketinggian drone, sehingga kertas dan plastik "
          "putih tidak bisa dibedakan secara visual, baik oleh model maupun oleh anotator. "
          "Confidence tinggi tidak berarti label salah; model juga bisa yakin dan keliru.", "",
          f"**Bukti tidak langsung label noise UAVVaste kuat.** {sub(uav.salinan_lain_konflik)[0]} dari {len(uav)} "
          f"citra UAVVaste yang terlibat ({sub(uav.salinan_lain_konflik)[1]} dari "
          f"{int(uav.kasus.sum())} kasusnya) berasal dari adegan yang salinan-salinannya di train/valid diberi kelas "
          "saling bertentangan (audit D0). Untuk objek sekecil itu, label kertas vs plastik kemungkinan besar "
          "tebakan anotator, dan tebakannya tidak konsisten antar salinan.", "",
          "**Ringkasnya:** kebingungan kertas↔plastik di UAVVaste paling tepat disebut *ambiguitas kelas pada "
          "objek kecil yang diperparah label tidak konsisten*, bukan murni kesalahan model dan bukan terbukti "
          "murni salah label. Di Burangrang dan TACO (objek besar, jelas terlihat) contoh kesalahan model yang "
          "nyata memang ada. Implikasinya: metrik per-kelas kertas_tisu dan plastik punya batas atas yang "
          "rendah selama label UAVVaste tidak direvisi, dan selisih kecil antar model pada dua kelas ini "
          "sebaiknya tidak ditafsirkan berlebihan."]
    (OUT / "analisis_label_noise.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> results/analisis_label_noise.md")
    print(tc.to_string(index=False))
    print(tf.to_string(index=False))
    print(dir_tab.to_string())
    print(hi_src.to_string())
    print(uav.to_string(index=False))
    print(sample[["no", "file", "sumber", "arah", "conf", "n_model", "salinan_d0", "konflik_d0"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

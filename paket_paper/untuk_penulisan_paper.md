# Bahan Penulisan Paper (dikumpulkan dari file yang sudah ada)

Dokumen ini hanya mengumpulkan angka dari file yang sudah ada. Tidak ada training, evaluasi ulang, atau analisis baru.
Satu-satunya perintah baru adalah `YOLO(path).info()` di CPU (bagian G).
Setiap angka disertai file sumbernya dalam tanda kurung. "TIDAK TERSEDIA" = angka itu tidak ada di file mana pun.
Path relatif terhadap root proyek.

> **Peringatan konsistensi yang perlu diperhatikan saat menulis**
> 1. mAP@50 YOLOv12s aug di test clean tercatat **dua nilai berbeda**: 0.46482521856158154 dari validator Ultralytics
>    (`results/test_metrics_overall.csv`) dan 0.4693354337236079 dari pencocokan ulang di analisis error
>    (`results/bootstrap_head_to_head.csv`). Begitu juga YOLOv8s aug: 0.4707709444438198 vs 0.4803693916528835.
>    Untuk tabel metrik utama pakai angka validator; angka 0.469/0.480 hanya untuk konteks bootstrap.
> 2. GFLOPs YOLOv12m 49.3 berasal dari laporan lama `analysis/C_findings.md`. Hasil final dan `info()` sama-sama 71.0 (lihat G).
> 3. Kalimat "objek UAVVaste umumnya 10–35 px" di `results/analisis_label_noise.md` adalah perkiraan kasar dan
>    **tidak tepat**. Ukuran sebenarnya dari 20 crop UAVVaste di contact sheet berkisar 9×17 px hingga 36×97 px dan 67×31 px
>    (lihat D5). Jangan kutip angka 10–35 px.

---

## A. Test set dan clean subset

**A1. Jumlah citra test full**
- File di `TA-1/test`: **550** (`results/ringkasan.md`; `analysis/tables/E2_leakage_summary.csv`: "Total Citra Test Set,550").
- Dievaluasi: **548**. Dua citra ditolak Ultralytics karena file label mencampur baris poligon dan baris box:
  `000015_jpg.rf.a6dd372912ce2f5a7950ed34a1a6a2d0.jpg` dan `000077_jpg.rf.59307395b28a31c9a0bb1fa813c3db1d.jpg`
  (`results/ringkasan.md`). `n_images` = 548 untuk semua baris full (`results/test_metrics_overall.csv`).
- Angka 530 bukan jumlah citra. 530 adalah jumlah kasus kertas↔plastik tertukar di test clean, dijumlah atas 13 model
  (`results/analisis_label_noise.md`).

**A2. Clean subset**
- Citra: **192** (550 − 358), dan 192 dievaluasi (`results/ringkasan.md`).
- Instance dan citra per kelas (`results/test_metrics_per_class.csv`, kolom `instances` dan `images`, baris test_set=clean):

| kelas | citra (images) | instance |
|---|---|---|
| kertas_tisu | 37 | 75 |
| logam_kaleng_kaca | 26 | 34 |
| plastik | 105 | 241 |
| puntung_rokok | 0 | 0 |

- puntung_rokok: 0 instance di clean (`results/ringkasan.md`: "puntung_rokok tidak punya instance").

**A3. Aturan menandai citra test sebagai "beririsan"**
- Hanya ada **satu** aturan: citra test dianggap beririsan bila *scene group*-nya (kolom `group_id` di
  `analysis/tables/D0_scene_groups.csv`) juga berisi citra dari split train atau valid
  (`scripts/01_evaluate_test.py`, fungsi `build_test_sets`: `cross = {group yang memuat "test" dan split lain}`).
- `group_id` = komponen terhubung dari pasangan citra yang lolos uji **SIFT + RANSAC dengan inlier ≥ 100**
  (`analysis/D0_dataset_audit.md` bagian 3; `analysis/D0_dataset_audit.py` baris 6).
- Pasangan kandidat yang diuji SIFT: nama sumber sama + 3 frame/foto berurutan + 10 tetangga embedding ResNet50
  + 5 tetangga dHash (flip-aware), total **52,370 pasangan** (`analysis/D0_dataset_audit.md`).
  Sesi video, sesi foto, dan nama file **hanya** dipakai untuk memilih kandidat. Sebuah pasangan tetap baru ditautkan
  bila inlier ≥ 100.
- Kalibrasi ambang: 200 pasangan acak menghasilkan inlier maks **8**, p95 **4**, median **0** (`analysis/D0_dataset_audit.md`).
- Zona abu-abu 25–99 inlier (**1,979** pasangan) **tidak** dipakai untuk clean subset. Zona ini hanya dipakai di
  `group_id_konservatif` (`analysis/D0_dataset_audit.md`).
- Kesamaan nama file saja **tidak** membuat citra dianggap beririsan (lihat A5).

**A4. Rincian 358 citra beririsan per penyebab**
- Total: 358 citra test berada di group lintas split (`analysis/D0_dataset_audit.md`: "Citra di dalam group tersebut:
  train 1016, valid 384, test **358** dari 550").
- Rincian per penyebab (sesi video / sesi foto / SIFT / nama file): **TIDAK TERSEDIA**. Hanya ada satu kriteria
  (SIFT ≥ 100 inlier), sehingga tidak ada pemecahan per penyebab. Rincian 358 per sumber dataset juga **TIDAK TERSEDIA**.
- Analisis lama yang berbeda metodenya (bukan dasar clean subset): "Citra Test Asal Sesi Burangrang yang Muncul di Train
  (Video+Burst), 313, 56.91%" (`analysis/tables/E2_leakage_summary.csv`). Analisis berbasis SSIM menyatakan
  1 citra test terpapar (SSIM ≥ 0.70) (`analysis/tables/E1_leakage_impact_summary.csv`). Keduanya **tidak** sama dengan 358.

**A5. "37 citra UAVVaste" dan "24 salinan bernama sama"**
- Definisi 37: citra UAVVaste di test clean yang memuat **minimal satu** kasus kertas↔plastik tertukar oleh **minimal satu**
  dari 13 model. Kasus dihitung pada conf ≥ 0.25 dan IoU > 0.5, dengan pencocokan yang sama dengan confusion matrix
  (`scripts/04_label_noise_check.py`, fungsi `collect_events`; `results/label_noise_uavvaste_salinan.csv`, 37 baris).
  Istilah yang tepat adalah "terlibat kebingungan kertas↔plastik", bukan "bermasalah". Citra UAVVaste di test clean seluruhnya
  **54** (`results/analisis_label_noise.md`).
- Cara 24 ditemukan: nama sumber diambil dengan membuang akhiran Roboflow `_jpg.rf.<hash>` (kolom `source_name`), lalu dicocokkan
  ke semua baris `analysis/tables/D0_scene_groups.csv` di semua split. 24 dari 37 punya ≥ 2 file dengan nama sumber sama.
  Ke-24 file itu semuanya punya salinan di train/valid, dan menyumbang 231 kasus. 15 citra (177 kasus) punya salinan lain yang
  tercatat di pasangan konflik kelas D0 (`results/analisis_label_noise.md`; kolom `salinan`, `salinan_di_train_valid`,
  `salinan_lain_konflik` di `results/label_noise_uavvaste_salinan.csv`).
- Salinan ini tidak tersaring karena aturan clean memakai SIFT ≥ 100, bukan nama file. Laporan D0 mencatat 709 dari 1613 nama
  sumber non-numerik masih terpecah ke > 1 group SIFT (`analysis/D0_dataset_audit.md`).

---

## B. Tabel metrik utama

Catatan cara baca:
- **Pengaturan evaluasi**: imgsz=640, conf=0.001, iou=0.6, batch=16, seed=0, deterministic=True (`results/tabel_rekap.md`).
- **P dan R di test full** dihitung atas 4 kelas. P dan R versi 3 kelas untuk full: **TIDAK TERSEDIA**.
- **Test clean** hanya punya 3 kelas. Karena itu P/R/mAP clean = versi 3 kelas, dan versi 4 kelas untuk clean **TIDAK TERSEDIA**
  (tidak terdefinisi).
- **Sumber**: semua angka akurasi berasal dari `results/test_metrics_overall.csv`. Params dalam M diambil dari `results/tabel_rekap.md`.
  Params mentah, GFLOPs, dan ms/citra diambil dari `results/test_metrics_overall.csv` (ms/citra = inference saja, batch 16, GPU).

### B1. Metrik keseluruhan

| Model | Test | Citra | P/R atas | Precision | Recall | mAP@50 (4 kls) | mAP@50-95 (4 kls) | mAP@50 (3 kls) | mAP@50-95 (3 kls) |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv12n baseline | full | 548 | 4 kelas | 0.3614496329400898 | 0.4084717956808233 | 0.34589420501431606 | 0.1966037487845452 | 0.4524642031769828 | 0.25615148960746376 |
| YOLOv12n baseline | clean | 192 | 3 kelas | 0.4145492819686046 | 0.5228351368209802 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.3873706898016831 | 0.1924012286133242 |
| YOLOv12s baseline | full | 548 | 4 kelas | 0.5788829208160835 | 0.5541078874559482 | 0.5862377464340432 | 0.3438575724220484 | 0.44998366191205763 | 0.2512267632293979 |
| YOLOv12s baseline | clean | 192 | 3 kelas | 0.43236062255813135 | 0.4140048273804681 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.39771928083497193 | 0.20536851040663864 |
| YOLOv12m baseline | full | 548 | 4 kelas | 0.6233103419828891 | 0.528045027896951 | 0.5457788274491964 | 0.3116381695974149 | 0.3960384365989285 | 0.2168508927965532 |
| YOLOv12m baseline | clean | 192 | 3 kelas | 0.34490602758130784 | 0.4121536083312993 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.3510348562496935 | 0.179796639417405 |
| YOLOv12l baseline | full | 548 | 4 kelas | 0.5370210366037766 | 0.5429779516026614 | 0.5551887048315906 | 0.24525871792868154 | 0.408584939775454 | 0.22524971914300396 |
| YOLOv12l baseline | clean | 192 | 3 kelas | 0.39551465171177796 | 0.3879282944159683 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.37011159617697187 | 0.1936762194767411 |
| YOLOv12x baseline | full | 548 | 4 kelas | 0.4259526247187416 | 0.4499082531743641 | 0.40821757159141037 | 0.21067193621660105 | 0.36540120656632497 | 0.20104869273324585 |
| YOLOv12x baseline | clean | 192 | 3 kelas | 0.40264106936139604 | 0.37145282455997614 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.3416734889091582 | 0.18571003757968488 |
| YOLOv12n aug | full | 548 | 4 kelas | 0.653765213022311 | 0.548783381910761 | 0.6186702855118745 | 0.2652396096878621 | 0.493227047349166 | 0.2874861462504828 |
| YOLOv12n aug | clean | 192 | 3 kelas | 0.5187708660550411 | 0.48080221300138315 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.43934072644288724 | 0.2384189871597918 |
| YOLOv12s aug | full | 548 | 4 kelas | 0.48269910185275644 | 0.5177840985151507 | 0.5202765916969821 | 0.2791566227034511 | 0.5248707046381218 | 0.3046762632214598 |
| YOLOv12s aug | clean | 192 | 3 kelas | 0.5138190366267913 | 0.4886540286930816 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.46482521856158154 | 0.25144669021265026 |
| YOLOv12m aug | full | 548 | 4 kelas | 0.6504171354822643 | 0.5758949122748843 | 0.6329757949385129 | 0.40070729057276944 | 0.5123010599180171 | 0.30227638743035934 |
| YOLOv12m aug | clean | 192 | 3 kelas | 0.48992592435489196 | 0.4741436623922513 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.42700448328043433 | 0.22537968611100165 |
| YOLOv12l aug | full | 548 | 4 kelas | 0.6142458108538223 | 0.5867529827883596 | 0.6250028061183419 | 0.268947080030496 | 0.5016704081577892 | 0.2924294400406614 |
| YOLOv12l aug | clean | 192 | 3 kelas | 0.5186584738675535 | 0.43470341892745323 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.40866049884321454 | 0.20900857959620386 |
| YOLOv12x aug | full | 548 | 4 kelas | 0.5994434219146145 | 0.6036916315159799 | 0.6079000775198025 | 0.3275275288754762 | 0.47886677002640327 | 0.271370038500635 |
| YOLOv12x aug | clean | 192 | 3 kelas | 0.40745955050874977 | 0.503736609443224 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.39368566383959286 | 0.20616226433636253 |
| YOLOv8n aug | full | 548 | 4 kelas | 0.619811646959679 | 0.6016712305161042 | 0.6184326879737074 | 0.31008439202594407 | 0.49291025063160987 | 0.28094585603459205 |
| YOLOv8n aug | clean | 192 | 3 kelas | 0.4305724860913222 | 0.5018526103656039 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.42724195139002025 | 0.22839518318549848 |
| YOLOv8s aug | full | 548 | 4 kelas | 0.5555525082809446 | 0.5811325095420676 | 0.5475155458267651 | 0.3003092918121569 | 0.5089096166579091 | 0.30107905574954247 |
| YOLOv8s aug | clean | 192 | 3 kelas | 0.5255514359839265 | 0.5085059528652401 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.4707709444438198 | 0.2513307304702423 |
| YOLOv8m aug | full | 548 | 4 kelas | 0.5332059238997503 | 0.6094405443771432 | 0.5074040258271659 | 0.29602282221527226 | 0.5107053677695546 | 0.303530429620363 |
| YOLOv8m aug | clean | 192 | 3 kelas | 0.5341792689944971 | 0.49961706397635125 | TIDAK TERSEDIA | TIDAK TERSEDIA | 0.4583753385175047 | 0.24269793308113918 |

### B2. Efisiensi dan lama training

| Model | Params (tabel_rekap.md) | params (csv) | GFLOPs (csv) | ms/citra full (csv) | ms/citra clean (csv) | Lama training |
|---|---|---|---|---|---|---|
| YOLOv12n baseline | 2.57 M | 2568828 | 7.4636288 | 1.8199708021286685 | 2.0124427076855986 | results.csv time 2335.38 s (= 0.649 jam, konversi satuan /3600) |
| YOLOv12s baseline | 9.25 M | 9254684 | 23.4896896 | 3.8604764594954784 | 3.8140875039971434 | results.csv time 5352.3 s (= 1.487 jam, konversi satuan /3600) |
| YOLOv12m baseline | 20.14 M | 20140572 | 71.0210048 | 8.87588576675413 | 8.715346875154259 | results.csv time 9993.81 s (= 2.776 jam, konversi satuan /3600) |
| YOLOv12l baseline | 26.39 M | 26392188 | 95.9680768 | 12.683983758367246 | 12.689178124370907 | results.csv time 18933.1 s (= 5.259 jam, konversi satuan /3600) |
| YOLOv12x baseline | 59.12 M | 59123004 | 209.6554496 | 22.00064634909364 | 22.153578125047108 | results.csv time 36033.1 s (= 10.009 jam, konversi satuan /3600) |
| YOLOv12n aug | 2.57 M | 2568828 | 7.4636288 | 2.201430110369612 | 2.31466823121688 | results.csv time 7283.4 s (= 2.023 jam, konversi satuan /3600) |
| YOLOv12s aug | 9.25 M | 9254684 | 23.4896896 | 3.841591238836166 | 4.124677605432225 | results.csv time 15851.8 s (= 4.403 jam, konversi satuan /3600) |
| YOLOv12m aug | 20.14 M | 20140572 | 71.0210048 | 8.681996715727541 | 8.686503646458732 | results.csv time 30170.6 s (= 8.381 jam, konversi satuan /3600) |
| YOLOv12l aug | 26.39 M | 26392188 | 95.9680768 | 12.65993631312311 | 12.723902604193427 | TIDAK TERSEDIA (results.csv time 63445.8 s ter-reset saat resume) |
| YOLOv12x aug | 59.12 M | 59123004 | 209.6554496 | 21.94561733631417 | 21.958091147098457 | results.csv time 104957 s (= 29.155 jam, konversi satuan /3600) |
| YOLOv8n aug | 3.01 M | 3011628 | 8.1949184 | 1.0726346730349763 | 1.878312500290728 | TIDAK TERSEDIA (results.csv time 901.446 s ter-reset saat resume) |
| YOLOv8s aug | 11.14 M | 11137148 | 28.6484992 | 2.7754910585103407 | 3.4850968744043107 | 3.167 jam (resume_log.txt baris 786); results.csv time 11399.9 s |
| YOLOv8m aug | 25.86 M | 25858636 | 79.0685184 | 6.191171717619265 | 6.473719272131954 | 4.586 jam (resume_log.txt baris 1411); results.csv time 16509.2 s |

### B3. AP per kelas (AP@50 dan AP@50-95)

Sumber: `results/test_metrics_per_class.csv` (kolom `mAP50` dan `mAP50_95` per kelas = AP kelas itu).

| Model | Test | Kelas | Instance | AP@50 | AP@50-95 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| YOLOv12n baseline | full | kertas_tisu | 233 | 0.30304087847456684 | 0.17017277723198332 | 0.3787480724248714 | 0.41201716738197425 |
| YOLOv12n baseline | full | logam_kaleng_kaca | 129 | 0.47978237168085697 | 0.3005414172399655 | 0.5121444571674677 | 0.5193798449612403 |
| YOLOv12n baseline | full | plastik | 763 | 0.5745693593755248 | 0.29774027435044254 | 0.5549060021680202 | 0.7024901703800787 |
| YOLOv12n baseline | full | puntung_rokok | 2 | 0.02618421052631579 | 0.017960526315789475 | 0.0 | 0.0 |
| YOLOv12n baseline | clean | kertas_tisu | 75 | 0.2078880185888306 | 0.10099115285501184 | 0.2959926002400173 | 0.3333333333333333 |
| YOLOv12n baseline | clean | logam_kaleng_kaca | 34 | 0.42968479977601187 | 0.23574765777793472 | 0.49459722403014106 | 0.5588235294117647 |
| YOLOv12n baseline | clean | plastik | 241 | 0.5245392510402068 | 0.24046487520702606 | 0.4530580216356555 | 0.6763485477178424 |
| YOLOv12n baseline | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12s baseline | full | kertas_tisu | 233 | 0.30276461559362683 | 0.16383487490348494 | 0.4878410372553097 | 0.2446351931330472 |
| YOLOv12s baseline | full | logam_kaleng_kaca | 129 | 0.4794737571905256 | 0.2910974759211794 | 0.566802968176619 | 0.4108527131782946 |
| YOLOv12s baseline | full | plastik | 763 | 0.5677126129520204 | 0.29874793886352935 | 0.6220545463661402 | 0.5609436435124509 |
| YOLOv12s baseline | full | puntung_rokok | 2 | 0.995 | 0.6217499999999999 | 0.638833131466265 | 1.0 |
| YOLOv12s baseline | clean | kertas_tisu | 75 | 0.2805102306485817 | 0.13957531363783646 | 0.3467015478731058 | 0.26666666666666666 |
| YOLOv12s baseline | clean | logam_kaleng_kaca | 34 | 0.44257349938648916 | 0.24679710191964982 | 0.48335276233913704 | 0.35294117647058826 |
| YOLOv12s baseline | clean | plastik | 241 | 0.4700741124698451 | 0.22973311566242965 | 0.46702755746215124 | 0.6224066390041494 |
| YOLOv12s baseline | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12m baseline | full | kertas_tisu | 233 | 0.2756502377204417 | 0.15462906381478447 | 0.4691692964163839 | 0.2017167381974249 |
| YOLOv12m baseline | full | logam_kaleng_kaca | 129 | 0.38420014273486136 | 0.2167394844557704 | 0.4864585924122638 | 0.34108527131782945 |
| YOLOv12m baseline | full | plastik | 763 | 0.5282649293414825 | 0.27918413011910465 | 0.5812345799577351 | 0.5693781020725496 |
| YOLOv12m baseline | full | puntung_rokok | 2 | 0.995 | 0.596 | 0.956378899145174 | 1.0 |
| YOLOv12m baseline | clean | kertas_tisu | 75 | 0.24450540744227672 | 0.12551422887138955 | 0.3272567250706813 | 0.24 |
| YOLOv12m baseline | clean | logam_kaleng_kaca | 34 | 0.3613443520640024 | 0.19679000555939818 | 0.3026091134094241 | 0.38235294117647056 |
| YOLOv12m baseline | clean | plastik | 241 | 0.44725480924280153 | 0.21708568382142737 | 0.4048522442638181 | 0.6141078838174274 |
| YOLOv12m baseline | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12l baseline | full | kertas_tisu | 233 | 0.28106866642456063 | 0.15999375494973075 | 0.44388486055675047 | 0.22610425360501296 |
| YOLOv12l baseline | full | logam_kaleng_kaca | 129 | 0.3953540430858778 | 0.22448466656006177 | 0.535776332620201 | 0.3953488372093023 |
| YOLOv12l baseline | full | plastik | 763 | 0.5493321098159236 | 0.29127073591921937 | 0.6298058416251164 | 0.5504587155963303 |
| YOLOv12l baseline | full | puntung_rokok | 2 | 0.995 | 0.30528571428571427 | 0.5386171116130386 | 1.0 |
| YOLOv12l baseline | clean | kertas_tisu | 75 | 0.24181750423919737 | 0.12375362073466349 | 0.35927643446729723 | 0.21333333333333335 |
| YOLOv12l baseline | clean | logam_kaleng_kaca | 34 | 0.35958608063059927 | 0.19960828314388807 | 0.3102173951299428 | 0.35294117647058826 |
| YOLOv12l baseline | clean | plastik | 241 | 0.5089312036611189 | 0.25766675455167176 | 0.5170501255380939 | 0.5975103734439834 |
| YOLOv12l baseline | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12x baseline | full | kertas_tisu | 233 | 0.27279398157312135 | 0.15924673765375025 | 0.34392168011706864 | 0.3562231759656652 |
| YOLOv12x baseline | full | logam_kaleng_kaca | 129 | 0.307943758205326 | 0.17181701179061865 | 0.3783715856641467 | 0.3798449612403101 |
| YOLOv12x baseline | full | plastik | 763 | 0.5154658799205276 | 0.2720823287553686 | 0.6092204649210708 | 0.563564875491481 |
| YOLOv12x baseline | full | puntung_rokok | 2 | 0.5366666666666667 | 0.23954166666666668 | 0.3722967681726802 | 0.5 |
| YOLOv12x baseline | clean | kertas_tisu | 75 | 0.21532634460442293 | 0.12184897501063427 | 0.2704237370792537 | 0.21333333333333335 |
| YOLOv12x baseline | clean | logam_kaleng_kaca | 34 | 0.31785584610116424 | 0.19100409616376018 | 0.37394629662167345 | 0.38235294117647056 |
| YOLOv12x baseline | clean | plastik | 241 | 0.49183827602188757 | 0.24427704156466018 | 0.5635531743832611 | 0.5186721991701245 |
| YOLOv12x baseline | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12n aug | full | kertas_tisu | 233 | 0.33751260083751633 | 0.19864975481640068 | 0.5073429138108357 | 0.2317596566523605 |
| YOLOv12n aug | full | logam_kaleng_kaca | 129 | 0.5280273064022228 | 0.32906379016425275 | 0.6953130843339184 | 0.4496124031007752 |
| YOLOv12n aug | full | plastik | 763 | 0.6141412348077587 | 0.33474489377079497 | 0.7267450880972098 | 0.5137614678899083 |
| YOLOv12n aug | full | puntung_rokok | 2 | 0.995 | 0.19849999999999998 | 0.6856597658472804 | 1.0 |
| YOLOv12n aug | clean | kertas_tisu | 75 | 0.32526059344220737 | 0.16750140179673562 | 0.47163101494556703 | 0.32 |
| YOLOv12n aug | clean | logam_kaleng_kaca | 34 | 0.45257578716923696 | 0.27779301060079814 | 0.5193080681290295 | 0.5 |
| YOLOv12n aug | clean | plastik | 241 | 0.5401857987172173 | 0.26996254908184164 | 0.5653735150905269 | 0.6224066390041494 |
| YOLOv12n aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12s aug | full | kertas_tisu | 233 | 0.40016256548485984 | 0.23204561445135902 | 0.5052138550488106 | 0.4034334763948498 |
| YOLOv12s aug | full | logam_kaleng_kaca | 129 | 0.5404773319268102 | 0.3360457130223603 | 0.6331243027562179 | 0.5116279069767442 |
| YOLOv12s aug | full | plastik | 763 | 0.6339722165026953 | 0.34593746219066 | 0.6669264239483861 | 0.6560750106890091 |
| YOLOv12s aug | full | puntung_rokok | 2 | 0.5064942528735631 | 0.20259770114942524 | 0.12553182565761134 | 0.5 |
| YOLOv12s aug | clean | kertas_tisu | 75 | 0.3631418021998705 | 0.18637724164245917 | 0.45265657942205817 | 0.37333333333333335 |
| YOLOv12s aug | clean | logam_kaleng_kaca | 34 | 0.47634539202996945 | 0.29753425214089557 | 0.5487069283550792 | 0.4411764705882353 |
| YOLOv12s aug | clean | plastik | 241 | 0.5549884614549045 | 0.27042857685459615 | 0.5400936021032365 | 0.6514522821576764 |
| YOLOv12s aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12m aug | full | kertas_tisu | 233 | 0.3451181500692434 | 0.20148270836674725 | 0.5104189270198833 | 0.28190185455087624 |
| YOLOv12m aug | full | logam_kaleng_kaca | 129 | 0.5680530265618218 | 0.35663376342057607 | 0.664292291665131 | 0.49087831879505694 |
| YOLOv12m aug | full | plastik | 763 | 0.623732003122986 | 0.34871269050375475 | 0.7407438387565868 | 0.5307994757536042 |
| YOLOv12m aug | full | puntung_rokok | 2 | 0.995 | 0.696 | 0.6862134844874559 | 1.0 |
| YOLOv12m aug | clean | kertas_tisu | 75 | 0.2924839200414604 | 0.15287624325767485 | 0.4249038770308518 | 0.37435962955453567 |
| YOLOv12m aug | clean | logam_kaleng_kaca | 34 | 0.4604517224227058 | 0.2519172149668897 | 0.4629815799929032 | 0.47058823529411764 |
| YOLOv12m aug | clean | plastik | 241 | 0.5280778073771367 | 0.27134560010844033 | 0.581892316040921 | 0.5774831223281006 |
| YOLOv12m aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12l aug | full | kertas_tisu | 233 | 0.3285730908840966 | 0.17646979464534782 | 0.46992376618335074 | 0.30042918454935624 |
| YOLOv12l aug | full | logam_kaleng_kaca | 129 | 0.5414481266807729 | 0.3504141130108164 | 0.693854629904894 | 0.49612403100775193 |
| YOLOv12l aug | full | plastik | 763 | 0.6349900069084982 | 0.35040441246582 | 0.7216473413980437 | 0.5504587155963303 |
| YOLOv12l aug | full | puntung_rokok | 2 | 0.995 | 0.19849999999999998 | 0.5715575059290012 | 1.0 |
| YOLOv12l aug | clean | kertas_tisu | 75 | 0.29255886496860994 | 0.12921830356390263 | 0.44663648539431144 | 0.3333333333333333 |
| YOLOv12l aug | clean | logam_kaleng_kaca | 34 | 0.38853765885019786 | 0.2203175402652175 | 0.5098354640055485 | 0.4117647058823529 |
| YOLOv12l aug | clean | plastik | 241 | 0.5448849727108358 | 0.2774898949594915 | 0.5995034722028005 | 0.5590122175666734 |
| YOLOv12l aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv12x aug | full | kertas_tisu | 233 | 0.3110113591456911 | 0.1731639729846859 | 0.4112184635249105 | 0.3836876554288296 |
| YOLOv12x aug | full | logam_kaleng_kaca | 129 | 0.5208300065772412 | 0.3122459059833048 | 0.6066222864439248 | 0.4806201550387597 |
| YOLOv12x aug | full | plastik | 763 | 0.6047589443562775 | 0.32870023653391434 | 0.680034211990934 | 0.5504587155963303 |
| YOLOv12x aug | full | puntung_rokok | 2 | 0.995 | 0.496 | 0.6998987256986888 | 1.0 |
| YOLOv12x aug | clean | kertas_tisu | 75 | 0.2356437735862414 | 0.11332046991753456 | 0.3071477809370238 | 0.41333333333333333 |
| YOLOv12x aug | clean | logam_kaleng_kaca | 34 | 0.44659814431099365 | 0.2613155363089664 | 0.40152605370488575 | 0.5294117647058824 |
| YOLOv12x aug | clean | plastik | 241 | 0.49881507362154354 | 0.2438507867825867 | 0.5137048168843399 | 0.5684647302904564 |
| YOLOv12x aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv8n aug | full | kertas_tisu | 233 | 0.3363182644567486 | 0.19862938978857853 | 0.44497294649048347 | 0.30472103004291845 |
| YOLOv8n aug | full | logam_kaleng_kaca | 129 | 0.5548527641604635 | 0.3274194123901247 | 0.6427652968352598 | 0.4728682170542636 |
| YOLOv8n aug | full | plastik | 763 | 0.5875597232776176 | 0.3167887659250729 | 0.635130739233071 | 0.6290956749672346 |
| YOLOv8n aug | full | puntung_rokok | 2 | 0.995 | 0.39749999999999996 | 0.7563776052799018 | 1.0 |
| YOLOv8n aug | clean | kertas_tisu | 75 | 0.2857115256767898 | 0.15937771566507802 | 0.3763678695169309 | 0.3541055489391353 |
| YOLOv8n aug | clean | logam_kaleng_kaca | 34 | 0.5013516319108561 | 0.27534217190580307 | 0.45014184456009065 | 0.5 |
| YOLOv8n aug | clean | plastik | 241 | 0.4946626965824148 | 0.2504656619856144 | 0.46520774419694505 | 0.6514522821576764 |
| YOLOv8n aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv8s aug | full | kertas_tisu | 233 | 0.3597819484681893 | 0.21190836648801356 | 0.5215102841101462 | 0.30873152759698436 |
| YOLOv8s aug | full | logam_kaleng_kaca | 129 | 0.5298093144648728 | 0.34532518936332623 | 0.6352435182892253 | 0.4496124031007752 |
| YOLOv8s aug | full | plastik | 763 | 0.6371375870406653 | 0.3460036113972876 | 0.7288777538682869 | 0.5661861074705111 |
| YOLOv8s aug | full | puntung_rokok | 2 | 0.6633333333333333 | 0.29799999999999993 | 0.3365784768561198 | 1.0 |
| YOLOv8s aug | clean | kertas_tisu | 75 | 0.3608495326594323 | 0.19643106181422648 | 0.5275480176329023 | 0.37333333333333335 |
| YOLOv8s aug | clean | logam_kaleng_kaca | 34 | 0.49226184037735493 | 0.27543703805359954 | 0.4724048491812696 | 0.5588235294117647 |
| YOLOv8s aug | clean | plastik | 241 | 0.5592014602946722 | 0.2821240915429008 | 0.5767014411376076 | 0.5933609958506224 |
| YOLOv8s aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |
| YOLOv8m aug | full | kertas_tisu | 233 | 0.3591108206037629 | 0.2093417055764657 | 0.5355672336404869 | 0.30042918454935624 |
| YOLOv8m aug | full | logam_kaleng_kaca | 129 | 0.5529216598966684 | 0.35532040692121963 | 0.6100768247262175 | 0.5426356589147286 |
| YOLOv8m aug | full | plastik | 763 | 0.6200836228082321 | 0.34592917636340365 | 0.7193833664378592 | 0.5946973340444881 |
| YOLOv8m aug | full | puntung_rokok | 2 | 0.4975 | 0.27349999999999997 | 0.26779627079443763 | 1.0 |
| YOLOv8m aug | clean | kertas_tisu | 75 | 0.3520969784843523 | 0.18068032045286817 | 0.5123848300238056 | 0.3466666666666667 |
| YOLOv8m aug | clean | logam_kaleng_kaca | 34 | 0.4803801645487801 | 0.26897584415635245 | 0.5081572321382299 | 0.5588235294117647 |
| YOLOv8m aug | clean | plastik | 241 | 0.5426488725193819 | 0.2784376346341969 | 0.5819957448214557 | 0.5933609958506224 |
| YOLOv8m aug | clean | puntung_rokok | 0 | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA | TIDAK TERSEDIA |

---

## C. Perbandingan YOLOv12 vs YOLOv8

**C1. Pasangan yang dibandingkan.** Temuan "FN 172 vs 172" membandingkan **YOLOv12s augmented** dengan **YOLOv8s augmented**
pada **test clean (192 citra)** (`results/analisis_error.md`; `results/bootstrap_head_to_head.csv`: clean, TOTAL, FN, 172.0, 172.0).
FN dihitung pada conf ≥ 0.25 dan IoU > 0.5 (`results/analisis_error.md`).

**C2. Metode CI [−19, +17].**
- Jenis: **bootstrap persentil**, CI 95% dari persentil 2.5 dan 97.5 (`scripts/02_error_analysis.py` baris 448).
- **Berpasangan**: indeks resample yang sama dipakai untuk kedua model.
- Unit resampling: **citra**.
- Jumlah resample: **1000** (`N_BOOT = 1000`, baris 40). Seed RNG 0 (baris 423).
- Pernyataan di laporan: "bootstrap 1000× berpasangan pada level citra" (`results/analisis_error.md` baris 35).
- Nilai persis: ci_lo −19.0, ci_hi 17.0 (`results/bootstrap_head_to_head.csv`).

**C3. FN dan FP per kelas di test clean, beserta CI selisih (v12 − v8)** (`results/bootstrap_head_to_head.csv`)

| kelas | metrik | YOLOv12s aug | YOLOv8s aug | delta (v12−v8) | CI 2.5% | CI 97.5% | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_tisu | FN | 51.0 | 53.0 | -2.0 | -10.0 | 5.0 | setara |
| kertas_tisu | FP | 35.0 | 33.0 | 2.0 | -13.0 | 16.0 | setara |
| logam_kaleng_kaca | FN | 23.0 | 18.0 | 5.0 | -1.0 | 11.0 | setara |
| logam_kaleng_kaca | FP | 15.0 | 25.0 | -10.0 | -20.025 | 0.0 | setara |
| plastik | FN | 98.0 | 101.0 | -3.0 | -18.0 | 11.0 | setara |
| plastik | FP | 137.0 | 117.0 | 20.0 | -5.0 | 46.0 | setara |
| TOTAL | FN | 172.0 | 172.0 | 0.0 | -19.0 | 17.0 | setara |
| TOTAL | FP | 194.0 | 181.0 | 13.0 | -22.025 | 47.0 | setara |
| mAP50 | mAP50 | 0.4693354337236079 | 0.4803693916528835 | -0.011033957929275584 | -0.06833105175081142 | 0.04523110188229537 | setara |

puntung_rokok tidak muncul di baris clean karena tidak ada instance di test clean.

**C4. Satu-satunya selisih signifikan di test full**
- Hanya satu baris di `results/bootstrap_head_to_head.csv` yang kesimpulannya bukan "setara":
  full, logam_kaleng_kaca, **FP**: YOLOv12s aug 57.0, YOLOv8s aug 79.0, delta **−22.0**, CI [**−41.0**, **−1.9750000000000227**],
  kesimpulan "YOLOv12 lebih baik".
- Artinya YOLOv12s aug menghasilkan **FP lebih sedikit** pada kelas logam/kaleng/kaca di test full.
- Di test clean, metrik yang sama: 15.0 vs 25.0, delta −10.0, CI [−20.025, 0.0], "setara".

---

## D. Analisis error

**D1. Objek kertas+plastik dan kasus tertukar per sumber** (`results/label_noise_per_sumber.csv`; kasus dijumlah atas 13 model,
conf ≥ 0.25, IoU > 0.5)

| test_set | sumber | citra | GT kertas+plastik | % GT | kasus (13 model) | % kasus | objek unik tertukar | % objek GT tertukar | kasus per GT per model |
|---|---|---|---|---|---|---|---|---|---|
| clean | UAVVaste | 54 | 152 | 0.4810126582278481 | 388 | 0.7320754716981132 | 78 | 0.5131578947368421 | 0.19635627530364372 |
| clean | Burangrang | 125 | 139 | 0.439873417721519 | 134 | 0.2528301886792453 | 34 | 0.2446043165467626 | 0.07415605976757056 |
| clean | TACO | 11 | 22 | 0.06962025316455696 | 8 | 0.01509433962264151 | 4 | 0.18181818181818182 | 0.027972027972027972 |
| clean | Unknown | 2 | 3 | 0.00949367088607595 | 0 | 0.0 | 0 | 0.0 | 0.0 |
| full | UAVVaste | 195 | 697 | 0.6997991967871486 | 1731 | 0.850196463654224 | 335 | 0.4806312769010043 | 0.19103851672000882 |
| full | Burangrang | 319 | 236 | 0.23694779116465864 | 257 | 0.12622789783889982 | 60 | 0.2542372881355932 | 0.08376792698826598 |
| full | TACO | 31 | 60 | 0.060240963855421686 | 48 | 0.023575638506876228 | 9 | 0.15 | 0.06153846153846154 |
| full | Unknown | 3 | 3 | 0.0030120481927710845 | 0 | 0.0 | 0 | 0.0 | 0.0 |

- Pemisahan **foto Burangrang vs video Burangrang** untuk angka-angka ini: **TIDAK TERSEDIA**. Semua digabung sebagai
  "Burangrang". "Lainnya" = baris `Unknown`.
- Sumber angka "48% objek vs 73% kasus": clean UAVVaste % GT 0.4810126582278481 dan % kasus 0.7320754716981132.

**D2. Cara menghitung "2,6 kali Burangrang"**
- Rumus: laju = kasus ÷ (jumlah GT kertas+plastik × 13 model) (`results/analisis_label_noise.md`: "kasus per GT per model").
- UAVVaste (clean): 388 ÷ (152 × 13) = 0.19635627530364372.
- Burangrang (clean): 134 ÷ (139 × 13) = 0.07415605976757056 (`results/label_noise_per_sumber.csv`).
- Rasio dicetak script dengan satu desimal sebagai "2.6×" (`scripts/04_label_noise_check.py` baris 342; `results/analisis_label_noise.md`).

**D3. Kelompok ukuran objek** (test clean; `results/stratifikasi_clean.csv`; batas di `scripts/02_error_analysis.py`
`SIZE_BINS`, luas bbox pada 640×640)
- Batas: small = luas < 32² (1024 px²); medium = 32² ≤ luas < 96² (9216 px²); large = luas ≥ 96².
- Recall dihitung pada conf ≥ 0.25 (`results/analisis_error.md` bagian 3).
- Recall model selain YOLOv12s aug dan YOLOv8s aug: **TIDAK TERSEDIA**. Stratifikasi hanya dijalankan untuk dua model ini.

| kelompok | model | citra | objek | recall | mAP50 |
|---|---|---|---|---|---|
| small (<32²) | YOLOv12s aug | 76 | 235 | 0.48936170212765956 | 0.5022142180027002 |
| small (<32²) | YOLOv8s aug | 76 | 235 | 0.4808510638297872 | 0.48834474897591984 |
| medium | YOLOv12s aug | 61 | 96 | 0.5833333333333334 | 0.50554326858467 |
| medium | YOLOv8s aug | 61 | 96 | 0.5625 | 0.5066272271130029 |
| large (>96²) | YOLOv12s aug | 15 | 19 | 0.3684210526315789 | 0.3956578643578643 |
| large (>96²) | YOLOv8s aug | 15 | 19 | 0.5789473684210527 | 0.5869721594830289 |

**D4. Kelompok kepadatan** (jumlah objek GT per citra, semua kelas; test clean; `results/stratifikasi_clean.csv`;
batas di `DENSITY_BINS`: 1–2, 3–5, ≥ 6 ditulis ">5")

| kelompok | model | citra | objek | recall | mAP50 |
|---|---|---|---|---|---|
| 1-2 | YOLOv12s aug | 77 | 102 | 0.6470588235294118 | 0.47026833851814304 |
| 1-2 | YOLOv8s aug | 77 | 102 | 0.6666666666666666 | 0.559027080124781 |
| 3-5 | YOLOv12s aug | 36 | 126 | 0.5396825396825397 | 0.5370741588405143 |
| 3-5 | YOLOv8s aug | 36 | 126 | 0.5238095238095238 | 0.5590766946614812 |
| >5 | YOLOv12s aug | 10 | 122 | 0.36065573770491804 | 0.3554507043500266 |
| >5 | YOLOv8s aug | 10 | 122 | 0.36065573770491804 | 0.3518496719705348 |

- **Konfirmasi 0,361**: recall kelompok >5 = 0.36065573770491804 untuk **kedua** model, YOLOv12s aug dan YOLOv8s aug
  (`results/stratifikasi_clean.csv`). Nilai ini dicetak 0.361 di `results/analisis_error.md` baris 99.
- Recall model lain: **TIDAK TERSEDIA**.

**D5. Pemeriksaan 40 crop** (`results/analisis_label_noise.md`, `results/label_noise_sample.csv`,
`results/label_noise_penilaian_visual.csv`, `results/figures/08_label_noise_sample.png`)
- **Kriteria**:
  1. Kasus kertas↔plastik di test clean (match conf ≥ 0.25, IoU > 0.5) dari **ke-13 model**. Satu baris per objek GT, dengan
     confidence tertinggi di antara model.
  2. Diambil yang **confidence > 0.5**: ada **64** objek.
  3. Diurutkan menurun berdasarkan confidence, dibatasi **maks. 3 objek per citra**, lalu diambil **40** teratas
     (`scripts/04_label_noise_check.py`: `N_SAMPLE=40`, `MIN_CONF=0.5`, `MAX_PER_IMAGE=3`).
- **Model**: bukan satu model. Tiap crop memakai prediksi model dengan confidence tertinggi untuk objek itu. Kolom `model` dan
  `n_model` (jumlah model yang keliru pada objek yang sama) ada di `results/label_noise_sample.csv`.
- **Rentang confidence sampel**: 0.912719… (#1) sampai 0.594898… (#40) (`results/label_noise_sample.csv`, kolom `conf`).
- **Hasil 3 / 5 / 32**: label_salah 3, model_salah 5, ambigu 19, tak_terbaca 13, sehingga 19 + 13 = 32
  (`results/analisis_label_noise.md`).
  Ini penilaian visual subjektif dari Claude atas crop 640×640 dan belum diverifikasi anotator.
- **Per sumber** (`results/analisis_label_noise.md`):
  - Burangrang: ambigu 12, label_salah 2, model_salah 3
  - TACO: ambigu 1, label_salah 1, model_salah 1
  - UAVVaste: ambigu 6, label_salah 0, model_salah 1, tak_terbaca 13

  Jumlah crop per sumber: Burangrang 17, TACO 3, UAVVaste 20 (penjumlahan baris tabel tersebut).
- **Rentang ukuran objek**, dibaca dari keterangan tiap crop di `results/figures/08_label_noise_sample.png`
  (ukuran box GT pada 640×640):
  - Semua 40 crop: terkecil 9×17 px (#16, UAVVaste), terbesar 205×266 px (#6, TACO).
  - 20 crop UAVVaste: 9×17 px sampai 36×97 px (#12) dan 67×31 px (#36). Kalimat "10–35 px" di
    `results/analisis_label_noise.md` tidak akurat.
  - Daftar lengkap koordinat box ada di kolom `gt_box` di `results/label_noise_sample.csv`.

---

## E. Dampak leakage

**E1. Selisih mAP full vs clean per model** (Δ = clean − full, mAP 3 kelas; `results/tabel_rekap.md`)

| Model | Δ mAP@50 | Δ mAP@50-95 |
|---|---|---|
| YOLOv12n baseline | -0.065 | -0.064 |
| YOLOv12s baseline | -0.052 | -0.046 |
| YOLOv12m baseline | -0.045 | -0.037 |
| YOLOv12l baseline | -0.038 | -0.032 |
| YOLOv12x baseline | -0.024 | -0.015 |
| YOLOv12n aug | -0.054 | -0.049 |
| YOLOv12s aug | -0.060 | -0.053 |
| YOLOv12m aug | -0.085 | -0.077 |
| YOLOv12l aug | -0.093 | -0.083 |
| YOLOv12x aug | -0.085 | -0.065 |
| YOLOv8n aug | -0.066 | -0.053 |
| YOLOv8s aug | -0.038 | -0.050 |
| YOLOv8m aug | -0.052 | -0.061 |

Δ hanya tersedia dalam 3 desimal (`results/tabel_rekap.md`). Nilai Δ tanpa pembulatan tidak tertulis di file mana pun:
**TIDAK TERSEDIA**. Komponennya (full 3 kelas dan clean) ada utuh di `results/test_metrics_overall.csv`.

**E2. Angka −0,053**
- Utamanya: **rata-rata Δ mAP@50-95 (3 kelas) dari 13 model** (`results/ringkasan.md`: "Δ mAP@50-95: **-0.053**
  (min -0.083, maks -0.015)").
- Kebetulan sama dengan Δ mAP@50-95 per model untuk **YOLOv12s aug** (−0.053) dan **YOLOv8n aug** (−0.053) (`results/tabel_rekap.md`).
- Pasangannya untuk mAP@50 adalah −0.058 (min −0.093, maks −0.024) (`results/ringkasan.md`).

**E3. Angka 0,251**
- mAP@50-95 test clean (3 kelas) untuk **YOLOv12s aug** = 0.2514466902126503 dan **YOLOv8s aug** = 0.2513307304702423
  (`results/test_metrics_overall.csv`).
- Catatan: mAP@50-95 full 3 kelas **YOLOv12s non-aug** = 0.2512267632293979 juga tercetak 0.251.

---

## F. Konfigurasi training

Sumber: `args.yaml` tiap run. Kolom waktu = nilai `time` di baris terakhir `results.csv` (detik).

| run | model | pretrained | data | epochs | patience | imgsz | batch | optimizer | lr0 | momentum | weight_decay | seed | mosaic | hsv_h | hsv_s | hsv_v | fliplr | scale | translate | time terakhir results.csv (detik) | time monoton naik |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| runs/detect/runs/augmented/aug_waste_yolov12l | runs\detect\runs\augmented\aug_waste_yolov12l\weights\last.pt | True | augmented\data.yaml | 100 | 100 | 640 | 2 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 63445.8 (epoch 100) | TIDAK (ada reset, kemungkinan resume) |
| runs/detect/runs/augmented/aug_waste_yolov12m | yolov12m.pt | True | augmented\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 30170.6 (epoch 100) | ya |
| runs/detect/runs/augmented/aug_waste_yolov12n | yolov12n.pt | True | augmented\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 7283.4 (epoch 100) | ya |
| runs/detect/runs/augmented/aug_waste_yolov12s | yolov12s.pt | True | augmented\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 15851.8 (epoch 100) | ya |
| runs/detect/runs/augmented/aug_waste_yolov12x | yolov12x.pt | True | augmented\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 104957 (epoch 100) | ya |
| runs/detect/runs/detect/waste_yolov12l | yolov12l.pt | True | TA-1\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 18933.1 (epoch 100) | ya |
| runs/detect/runs/detect/waste_yolov12m | yolov12m.pt | True | TA-1\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 9993.81 (epoch 100) | ya |
| runs/detect/runs/detect/waste_yolov12n | yolov12n.pt | True | TA-1\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 2335.38 (epoch 100) | ya |
| runs/detect/runs/detect/waste_yolov12s | yolov12s.pt | True | TA-1\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 5352.3 (epoch 100) | ya |
| runs/detect/runs/detect/waste_yolov12x | yolov12x.pt | True | TA-1\data.yaml | 100 | 100 | 640 | -1 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 36033.1 (epoch 100) | ya |
| runs/detect/runs_baseline/v8m | yolov8m.pt | True | augmented\data.yaml | 100 | 100 | 640 | 16 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 16509.2 (epoch 100) | ya |
| runs/detect/runs_baseline/v8n | runs\detect\runs_baseline\v8n\weights\last.pt | True | augmented\data.yaml | 100 | 100 | 640 | 16 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 901.446 (epoch 100) | TIDAK (ada reset, kemungkinan resume) |
| runs/detect/runs_baseline/v8n_test_3ep | yolov8n.pt | True | augmented/data.yaml | 3 | 100 | 640 | 16 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 120.669 (epoch 3) | ya |
| runs/detect/runs_baseline/v8s | yolov8s.pt | True | augmented\data.yaml | 100 | 100 | 640 | 16 | auto | 0.01 | 0.937 | 0.0005 | 0 | 1.0 | 0.015 | 0.7 | 0.4 | 0.5 | 0.5 | 0.1 | 11399.9 (epoch 100) | ya |

Catatan:
- **batch = -1** berarti AutoBatch Ultralytics. Ukuran batch yang benar-benar dipakai tidak tercatat (tidak ada log training
  YOLOv12): **TIDAK TERSEDIA**. `aug_waste_yolov12l` tercatat **batch = 2** di args.yaml-nya.
- `aug_waste_yolov12l` dan `v8n` di-resume, sehingga args.yaml mencatat `model` = `last.pt`.
  - Bobot pretrained awal `aug_waste_yolov12l`: **TIDAK TERSEDIA**.
  - Untuk `v8n`, args.yaml maupun `resume_log.txt` (baris 4: `model=...v8n\weights\last.pt`) tidak memuat bobot awal: **TIDAK TERSEDIA**.
  - Kolom `time` keduanya tidak monoton (ter-reset saat resume), sehingga total lama training dari `results.csv` tidak bisa dipakai.
- **Optimizer yang benar-benar dipakai**: hanya tercatat untuk YOLOv8 di `runs/detect/runs_baseline/resume_log.txt`.
  - v8n (resume, baris 100–101):
    `optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...`
    `optimizer: MuSGD(lr=0.01, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)`
  - v8s (baris 277): `optimizer: MuSGD(lr=0.01, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)`
  - v8m (baris 902): `optimizer: MuSGD(lr=0.01, momentum=0.9) with parameter groups 77 weight(decay=0.0), 84 weight(decay=0.0005), 83 bias(decay=0.0)`
  - Optimizer aktual untuk 10 run YOLOv12: **TIDAK TERSEDIA**. Tidak ada log training YOLOv12 yang tersimpan; args.yaml hanya
    mencatat `optimizer: auto`.
- **Lama training dari log**:
  - v8s: "100 epochs completed in 3.167 hours." (baris 786)
  - v8m: "100 epochs completed in 4.586 hours." (baris 1411)
  - v8n: hanya "10 epochs completed in 0.250 hours." (baris 161, bagian resume epoch 91–100). Total v8n: **TIDAK TERSEDIA**.

---

## G. Pengecekan `YOLO(path).info()` di CPU

Perintah: `YOLO(path).info()` untuk tiap `runs/**/weights/best.pt`, dengan `CUDA_VISIBLE_DEVICES=""`. Tidak ada inferensi lain.
Output persis:

```
### runs/detect/runs/augmented/aug_waste_yolov12l/weights/best.pt
YOLOv12l summary: 488 layers, 26,392,188 parameters, 0 gradients, 96.0 GFLOPs
### runs/detect/runs/augmented/aug_waste_yolov12m/weights/best.pt
YOLOv12m summary: 292 layers, 20,140,572 parameters, 0 gradients, 71.0 GFLOPs
### runs/detect/runs/augmented/aug_waste_yolov12n/weights/best.pt
YOLOv12n summary: 272 layers, 2,568,828 parameters, 0 gradients, 7.5 GFLOPs
### runs/detect/runs/augmented/aug_waste_yolov12s/weights/best.pt
YOLOv12s summary: 272 layers, 9,254,684 parameters, 0 gradients, 23.5 GFLOPs
### runs/detect/runs/augmented/aug_waste_yolov12x/weights/best.pt
YOLOv12x summary: 488 layers, 59,123,004 parameters, 0 gradients, 209.7 GFLOPs
### runs/detect/runs/detect/waste_yolov12l/weights/best.pt
YOLOv12l summary: 488 layers, 26,392,188 parameters, 0 gradients, 96.0 GFLOPs
### runs/detect/runs/detect/waste_yolov12m/weights/best.pt
YOLOv12m summary: 292 layers, 20,140,572 parameters, 0 gradients, 71.0 GFLOPs
### runs/detect/runs/detect/waste_yolov12n/weights/best.pt
YOLOv12n summary: 272 layers, 2,568,828 parameters, 0 gradients, 7.5 GFLOPs
### runs/detect/runs/detect/waste_yolov12s/weights/best.pt
YOLOv12s summary: 272 layers, 9,254,684 parameters, 0 gradients, 23.5 GFLOPs
### runs/detect/runs/detect/waste_yolov12x/weights/best.pt
YOLOv12x summary: 488 layers, 59,123,004 parameters, 0 gradients, 209.7 GFLOPs
### runs/detect/runs_baseline/v8m/weights/best.pt
Model summary: 170 layers, 25,858,636 parameters, 0 gradients, 79.1 GFLOPs
### runs/detect/runs_baseline/v8n/weights/best.pt
Model summary: 130 layers, 3,011,628 parameters, 0 gradients, 8.2 GFLOPs
### runs/detect/runs_baseline/v8n_test_3ep/weights/best.pt
Model summary: 130 layers, 3,011,628 parameters, 0 gradients, 8.2 GFLOPs
### runs/detect/runs_baseline/v8s/weights/best.pt
Model summary: 130 layers, 11,137,148 parameters, 0 gradients, 28.6 GFLOPs
```

- **YOLOv12m** = 20,140,572 parameter dan **71.0 GFLOPs**, sama untuk baseline dan augmented. Nilai ini sama dengan
  `results/test_metrics_overall.csv` (71.0210048) dan `results/tabel_rekap.md` (71.0).
- Angka 49.3 hanya muncul di laporan lama `analysis/C_findings.md` dan `analysis/tables/C_*.csv`.
- Selisih terhadap nilai resmi 67.5: **TIDAK TERSEDIA** di file mana pun. Tidak ada file yang mencatat nilai resmi, dan
  penyebab selisihnya tidak diselidiki.

---

## H. Dataset

**H1. Komposisi citra per sumber** (berdasarkan pola nama file)

Seluruh TA-1, 3948 citra (`analysis/D0_dataset_audit.md` bagian 2):

| sumber | citra | nama sumber unik | kertas_tisu | logam_kaleng_kaca | plastik | puntung_rokok |
|---|---|---|---|---|---|---|
| UAVVaste | 1726 | 722 | 1712 | 1148 | 4441 | 0 |
| TACO | 438 | 117 | 137 | 253 | 705 | 2 |
| Burangrang | 1762 | 879 | 301 | 92 | 1443 | 86 |
| Unknown | 22 | 12 | 22 | 23 | 161 | 1 |

Rincian prefiks (`analysis/tables/E3_source_counts.csv`):

| Source | Count |
|---|---|
| Batch_BATCH_dXX_img | 438 |
| Batch_BATCH_sXX_img | 341 |
| Batch_batch_01_frame | 90 |
| Batch_batch_XX_img | 386 |
| Batch_batch_sXX_img | 471 |
| Mandiri_Foto | 723 |
| Mandiri_GoPro | 87 |
| Mandiri_Video | 952 |
| TACO | 438 |
| Unknown | 22 |

- **BATCH_ dan batch_**: keduanya dimasukkan ke UAVVaste oleh aturan heuristik D0 ("`batch_*`/`BATCH_*` = UAVVaste")
  (`analysis/D0_dataset_audit.md`), dan juga oleh `analysis/tables/E2_uavvaste_patterns.csv`.
- Ini **asumsi berbasis pola nama**, bukan hasil verifikasi. D0 menulis: "tagging manual belum pernah diselesaikan".
  Bukti independen bahwa prefiks BATCH_ berasal dari UAVVaste: **TIDAK TERSEDIA**.
- Unknown = pola `DJI_#`, `camera_img_#`, `photo_#-#-#_#-#-#` (`analysis/D0_dataset_audit.md`).

**H2. Apakah UAVVaste berasal dari versi Roboflow Universe yang sudah berisi salinan rotasi/crop?**
- Asal UAVVaste sebelum diimpor ke proyek Roboflow: **TIDAK TERSEDIA**. Tidak ada file yang mencatat dari mana UAVVaste diimpor.
- Yang tercatat:
  - `TA-1/data.yaml`: workspace `weapon-mpr3p`, project `ta-rhzr5`, version 1,
    url `https://universe.roboflow.com/weapon-mpr3p/ta-rhzr5/dataset/1`, sehingga TA-1 adalah ekspor dari proyek Roboflow milik sendiri.
  - `TA-1/README.roboflow.txt`: "The dataset includes 3948 images." … "Resize to 640x640 (Stretch)" …
    "**No image augmentation techniques were applied.**" Jadi salinan rotasi/crop **tidak** dibuat saat ekspor TA-1.
  - Bukti salinan sudah ada di data sumber:
    - UAVVaste 1726 file dari 722 nama sumber unik (`analysis/D0_dataset_audit.md`).
    - 1724 citra unik secara piksel dari 1726 file, sehingga salinan bukan duplikat piksel melainkan varian (`analysis/tables/E2_unique_counts_comparison.csv`).
    - Proposal TA menyebut 772 citra mentah UAVVaste (`analysis/C0_verify.md`).
    - Catatan manual D0: "pada adegan yang sama (varian rotasi Roboflow), objek kecil yang sama diberi kelas berbeda antar
      salinan … tiap salinan dianotasi terpisah, bukan diwariskan" (`analysis/D0_dataset_audit.md`).
  - Kesimpulan yang bisa ditulis: varian rotasi/crop sudah ada di data sebelum ekspor TA-1. Apakah varian itu berasal dari
    versi Universe tertentu tidak bisa dibuktikan dari file yang ada.

**H3. 89 instance puntung rokok**
- Total **89**: train 86, valid 1, test 2 (`analysis/D0_dataset_audit.md` bagian 1).
- Per sumber: Burangrang 86, TACO 2, Unknown 1 (`analysis/D0_dataset_audit.md` bagian 2).
- Sebaran per group: 18 group (≥ 100 inlier). Kelompok terbesar berasal dari foto Burangrang `IMG_5456_JPG`–`IMG_5508_JPG`
  (group 1321–1336, antara lain 16, 11, 10, 9, 8, 5, 4, 4, 3, 3, 2, 2 instance; ditambah group 1725 `IMG_5463_JPG` 1 instance), plus `IMG_5332_JPG`–`IMG_5334_JPG` (5),
  `IMG_4948` (3), TACO `000036` (2), dan `photo_2020-10-23_14-02-04` (1) (`analysis/D0_dataset_audit.md`, tabel
  "Sebaran puntung_rokok per group").
- Tanggal atau nama **sesi** foto: **TIDAK TERSEDIA**. Yang tercatat hanya nomor file `IMG_*`.
- Masuk clean subset: **0** (`results/ringkasan.md`; `results/test_metrics_per_class.csv`, clean puntung_rokok instances 0).

---

## I. Daftar gambar

Resolusi dibaca dari header file PNG. Semua disimpan 300 dpi.

| file | isi | model | test set | resolusi (px) | tanda |
|---|---|---|---|---|---|
| 01_map_per_varian.png | Bar chart mAP@50 dan mAP@50-95 (3 kelas) per varian, 2×2 panel | YOLOv12 n–x baseline & aug, YOLOv8 n/s/m aug | full + clean | 2969×2396 |  |
| 02_selisih_full_vs_clean.png | Grouped bar Δ mAP (clean − full) per model + garis rata-rata | 13 model | full vs clean | 2973×1416 |  |
| 03_tradeoff_akurasi_vs_waktu.png | Scatter mAP@50-95 vs ms/citra + batas Pareto | 13 model | full + clean (waktu dari full) | 3251×1712 |  |
| 04_kurva_training_yolov12.png | Kurva training: box loss, cls loss (train), mAP50, mAP50-95 (valid) per epoch | YOLOv12 n–x baseline & aug | split valid (bukan test) | 2973×2558 |  |
| 05_kurva_pr_v12s_vs_v8s_clean.png | Kurva presisi-recall per kelas, berdampingan | YOLOv12s aug vs YOLOv8s aug | clean | 2973×1506 |  |
| 05_kurva_pr_v12s_vs_v8s_full.png | Kurva presisi-recall per kelas, berdampingan | YOLOv12s aug vs YOLOv8s aug | full | 2973×1506 |  |
| 06_contoh_deteksi_benar.png | Grid 3×3 deteksi tanpa FN/FP, GT hijau vs prediksi merah | YOLOv12s aug | clean | 2824×3279 |  |
| 07_contoh_kegagalan.png | Grid 2×3 kegagalan terparah + dugaan penyebab (heuristik) | YOLOv12s aug | clean | 3085×2713 |  |
| 08_label_noise_sample.png | Contact sheet 40 crop kertas↔plastik tertukar (conf > 0.5) | 13 model (per objek: confidence tertinggi) | clean | 3300×6780 | **grid crop ambigu** |

- **Confusion matrix**: tidak ada di `results/figures/`. Ke-26 confusion matrix ada di `results/confusion_matrix/`, dan ikut disalin
  ke `paket_paper/confusion_matrix/`. Tiap file 2250×1950 px, untuk 13 model × {full, clean}, conf ≥ 0.25, IoU > 0.5,
  dengan label Indonesia (`results/analisis_error.md`).
- **Grid crop ambigu**: `08_label_noise_sample.png`.
- **Grafik per sumber data**: **TIDAK ADA** di `results/figures/`. Data untuk grafik "48% objek vs 73% kasus" ada di
  `results/label_noise_per_sumber.csv` (lihat D1).

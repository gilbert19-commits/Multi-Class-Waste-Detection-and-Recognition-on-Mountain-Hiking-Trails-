# Evaluasi dataset 2 kelas — test split penuh

Dataset berdiri sendiri; tidak dibandingkan dengan run atau hasil lama.

## Pengaturan evaluasi

- Bobot: `best.pt` tiap run di `runs_2kelas/`. Data: `dataset-2kelas/test` (550 citra), di-mirror dengan hardlink ke `eval_2kelas/_testset/` agar dataset asli tidak tersentuh.
- `model.val(split='test', imgsz=512, conf=0.001, iou=0.6, batch=16, device=0, half=False, seed=0, deterministic=True)`. GPU: NVIDIA GeForce RTX 5070.
- Citra yang benar-benar dinilai: **548**. Dibuang Ultralytics (label campuran poligon+box, ditandai corrupt): 2 citra: `000015_jpg.rf.b8f56c460b5d8716b3bf1caa8e37e747.jpg`, `000077_jpg.rf.6085ccd41b6b72d0f29b45ebe9a71ede.jpg`.
- Pesan labels.cache terkait: 2 (lihat `_testset/labels.cache`).
- Waktu per citra dari `validator.speed` (ms/citra, batch 16, FP32), median dari 3 kali val. FPS_inference = 1000 / ms_inference (forward saja); FPS_total = 1000 / (pre + inference + post).
- Puncak memori GPU = `torch.cuda.max_memory_allocated` / `max_memory_reserved` selama satu kali val, diukur di proses Python baru per model (`scripts/eval_01b_memori.py`, termasuk bobot model). Angka dari proses bersama (10 model berurutan) ikut menghitung sisa alokasi model sebelumnya, disimpan sebagai `*_proses_bersama` di `tables/eval_ultralytics_raw.json` dan tidak dipakai.
- Params dan GFLOPs pada 512×512 dari `ultralytics.utils.torch_utils.get_num_params/get_flops`, model sebelum di-fuse.

## 1. Run training

Sumber: `tables/01_inventaris_run.csv`, `salinan/runs/*/args.yaml`, `salinan/runs/*/results.csv`, `best.pt` (train_args), `train_2kelas.log`.

| model | epoch_dijalankan | epoch_terbaik | early_stopping_terpicu | batch_dipakai | lama_training_jam | jumlah_sesi | epochs | patience | imgsz | optimizer | lr0 | momentum | weight_decay | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv12n | 200 | 195 | tidak | 23 | 2.1665722222222223 | 1 | 200 | 100 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv12s | 200 | 190 | tidak | 12 | 3.7007222222222222 | 1 | 200 | 100 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv12m | 200 | 199 | tidak | 6 | 7.397777777777778 | 1 | 200 | 100 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv12l | 200 | 196 | tidak | 3 | 19.59386111111111 | 2 | 200 | 50 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv12x | 200 | 193 | tidak | 2 | 29.44638888888889 | 1 | 200 | 50 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv8n | 200 | 200 | tidak | 16 | 2.7900125 | 3 | 200 | 50 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv8s | 200 | 166 | tidak | 16 | 4.613611111111111 | 1 | 200 | 50 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv8m | 171 | 121 | ya | 11 | 3.7957222222222224 | 1 | 200 | 50 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv8l | 200 | 174 | tidak | 5 | 7.975444444444444 | 1 | 200 | 50 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |
| YOLOv8x | 200 | 183 | tidak | 3 | 12.969472222222223 | 1 | 200 | 50 | 512 | auto | 0.01 | 0.937 | 0.0005 | 0 |

- Optimizer yang benar-benar dipakai (log Ultralytics, `optimizer=auto`): `MuSGD(lr=0.01, momentum=0.9)` untuk semua run; lr0=0.01 dan momentum=0.937 di args.yaml diabaikan oleh mode auto.
- weight_decay efektif = weight_decay × batch × akumulasi / nbs(64): YOLOv12n 0.0005390625, YOLOv12s 0.00046875, YOLOv12m 0.000515625, YOLOv12l 0.0004921875, YOLOv12x 0.0005, YOLOv8n 0.0005, YOLOv8s 0.0005, YOLOv8m 0.000515625, YOLOv8l 0.0005078125, YOLOv8x 0.0004921875.
- Epoch terbaik = epoch dengan val mAP@50-95 tertinggi (fitness Ultralytics, = best.pt).
- Lama training = jumlah kolom `time` results.csv per sesi (train + val tiap epoch); tidak termasuk epoch parsial yang hilang saat jeda dan validasi akhir best.pt.
- Catatan run (dari `salinan/waktu_training_2kelas.txt`): patience 100 untuk YOLOv12n/s/m, 50 untuk lainnya (YOLOv12l: 100 pada epoch 1–46, lalu 50 setelah resume, state early stop dipulihkan). Batch YOLOv12 dan YOLOv8m/l/x dari AutoBatch; YOLOv8n/s batch tetap 16. workers 4, kecuali YOLOv8s (0) dan YOLOv8n epoch 1–67 (0). Jeda/resume: YOLOv12l setelah epoch 46; YOLOv8n setelah epoch 67 dan 74.

## 2. Metrik test penuh

Sumber: `tables/02_metrik_test_penuh.csv`, `tables/eval_ultralytics_raw.json`. ★ = mAP@50 tertinggi (YOLOv12l).

| model | mAP50 | mAP50_95 | mAP75 | precision | recall | n_citra |
|---|---|---|---|---|---|---|
| YOLOv12n | 0.6157665902323093 | 0.3583621655214536 | 0.3567817798790943 | 0.7131215732536291 | 0.5969963182178311 | 548 |
| YOLOv12s | 0.6530196913979882 | 0.39541818844957033 | 0.3953295654919169 | 0.7210064508857872 | 0.6260272482950398 | 548 |
| YOLOv12m | 0.648566626153444 | 0.3932439948783493 | 0.4200812915197154 | 0.6988108588216588 | 0.6394222553634401 | 548 |
| YOLOv12l ★ | 0.6673706586147301 | 0.40609178574579463 | 0.44575508143830966 | 0.7424799111709737 | 0.6334387926102298 | 548 |
| YOLOv12x | 0.6631210638342127 | 0.4004873410189428 | 0.40636466081946654 | 0.7635847656859618 | 0.6231260971555515 | 548 |
| YOLOv8n | 0.6229786550618368 | 0.3467982012418869 | 0.3452094533072582 | 0.686205162395854 | 0.6051172111665191 | 548 |
| YOLOv8s | 0.6507079239020179 | 0.3884243326085295 | 0.38891112338015127 | 0.7390689399809116 | 0.6067582648559615 | 548 |
| YOLOv8m | 0.6465209960640073 | 0.3936187687737399 | 0.3986275554677858 | 0.6687341053455119 | 0.6652879402215284 | 548 |
| YOLOv8l | 0.6610413892352578 | 0.3987241325187082 | 0.42189183411512277 | 0.7970334000042352 | 0.5871273315700771 | 548 |
| YOLOv8x | 0.6473513407827314 | 0.38691855564068234 | 0.41100415897048 | 0.7418059435318847 | 0.6273691578046381 | 548 |

### Per kelas

Sumber: `tables/02_metrik_per_kelas_test_penuh.csv`.

| model | kelas | citra | instance | precision | recall | AP50 | AP50_95 | AP75 |
|---|---|---|---|---|---|---|---|---|
| YOLOv12n | kertas_plastik_puntungRokok | 293 | 998 | 0.8158774989357145 | 0.6823647294589178 | 0.7156793206101763 | 0.39904576865572106 | 0.39668288825035936 |
| YOLOv12n | logam_kaleng_kaca | 81 | 129 | 0.6103656475715439 | 0.5116279069767442 | 0.5158538598544422 | 0.31767856238718617 | 0.3168806715078293 |
| YOLOv12s | kertas_plastik_puntungRokok | 293 | 998 | 0.8370759856450422 | 0.7094188376753507 | 0.7462154771887738 | 0.4252551362757715 | 0.4238952826941332 |
| YOLOv12s | logam_kaleng_kaca | 81 | 129 | 0.6049369161265323 | 0.5426356589147286 | 0.5598239056072025 | 0.36558124062336905 | 0.36676384828970066 |
| YOLOv12m | kertas_plastik_puntungRokok | 293 | 998 | 0.8199178340652715 | 0.7284569138276553 | 0.7431393638733108 | 0.42780689019309426 | 0.4465170751143449 |
| YOLOv12m | logam_kaleng_kaca | 81 | 129 | 0.577703883578046 | 0.5503875968992248 | 0.5539938884335772 | 0.3586810995636044 | 0.39364550792508585 |
| YOLOv12l | kertas_plastik_puntungRokok | 293 | 998 | 0.8221592629691246 | 0.7087380503367388 | 0.7366026895470265 | 0.42218665416113677 | 0.4317001341248236 |
| YOLOv12l | logam_kaleng_kaca | 81 | 129 | 0.6628005593728229 | 0.5581395348837209 | 0.5981386276824338 | 0.3899969173304524 | 0.4598100287517958 |
| YOLOv12x | kertas_plastik_puntungRokok | 293 | 998 | 0.8404024051193031 | 0.6803607214428857 | 0.7351891154723003 | 0.42159055457493827 | 0.423320038861706 |
| YOLOv12x | logam_kaleng_kaca | 81 | 129 | 0.6867671262526205 | 0.5658914728682171 | 0.591053012196125 | 0.37938412746294725 | 0.3894092827772271 |
| YOLOv8n | kertas_plastik_puntungRokok | 293 | 998 | 0.7632799050610125 | 0.6753507014028056 | 0.7074749875015589 | 0.383513204330989 | 0.3658766154373194 |
| YOLOv8n | logam_kaleng_kaca | 81 | 129 | 0.6091304197306957 | 0.5348837209302325 | 0.5384823226221146 | 0.31008319815278496 | 0.324542291177197 |
| YOLOv8s | kertas_plastik_puntungRokok | 293 | 998 | 0.8338658153540823 | 0.6940422063269384 | 0.7476807980453513 | 0.4231340486556065 | 0.42106162506628864 |
| YOLOv8s | logam_kaleng_kaca | 81 | 129 | 0.644272064607741 | 0.5194743233849846 | 0.5537350497586847 | 0.35371461656145253 | 0.3567606216940139 |
| YOLOv8m | kertas_plastik_puntungRokok | 293 | 998 | 0.7923506271657675 | 0.7104208416833667 | 0.7339564046061187 | 0.4172553152172117 | 0.41711987811066764 |
| YOLOv8m | logam_kaleng_kaca | 81 | 129 | 0.5451175835252563 | 0.6201550387596899 | 0.5590855875218957 | 0.369982222330268 | 0.3801352328249039 |
| YOLOv8l | kertas_plastik_puntungRokok | 293 | 998 | 0.8686459491156965 | 0.6626267561634102 | 0.7347593707022048 | 0.4153675236906819 | 0.4053507961066751 |
| YOLOv8l | logam_kaleng_kaca | 81 | 129 | 0.7254208508927739 | 0.5116279069767442 | 0.5873234077683109 | 0.38208074134673453 | 0.43843287212357046 |
| YOLOv8x | kertas_plastik_puntungRokok | 293 | 998 | 0.8265674713222809 | 0.6733429667720667 | 0.7153205269245106 | 0.4036914591735334 | 0.4112763022845922 |
| YOLOv8x | logam_kaleng_kaca | 81 | 129 | 0.6570444157414884 | 0.5813953488372093 | 0.579382154640952 | 0.37014565210783135 | 0.41073201565636774 |

### Kompleksitas dan kecepatan

Sumber: `tables/02_metrik_test_penuh.csv`.

| model | params | GFLOPs_512 | ms_preprocess | ms_inference | ms_postprocess | ms_total | FPS_inference | FPS_total | ms_inference_min | ms_inference_max | peak_mem_alloc_MiB | peak_mem_reserved_MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv12n | 2568438 | 4.54885376 | 0.4728691606530852 | 1.211908760643734 | 0.615422992481014 | 2.300200913777833 | 825.1446251356632 | 434.74463209285807 | 1.1915598526923326 | 1.2189700734720015 | 312.8173828125 | 478.0 |
| YOLOv12s | 9253910 | 14.577664 | 0.4575888695956691 | 2.4198428838153503 | 0.5892166068240402 | 3.4666483602350597 | 413.2499703548135 | 288.4630617488397 | 2.413793978532844 | 2.466398540690514 | 564.16259765625 | 856.0 |
| YOLOv12m | 20139030 | 44.692963328 | 0.5101689777377802 | 5.462785402268688 | 0.5791618609393057 | 6.552116240945774 | 183.05679728599648 | 152.62244490577805 | 5.4048514605438625 | 5.494365876499754 | 1032.8359375 | 1678.0 |
| YOLOv12l | 26390646 | 59.904114688 | 0.4510465345726124 | 7.646895255077712 | 0.565898540960013 | 8.663840330610338 | 130.77202794637174 | 115.42225639441735 | 7.644460765546337 | 7.66800073048535 | 1491.7822265625 | 1994.0 |
| YOLOv12x | 59120694 | 131.906306048 | 0.46208558347497647 | 13.491183759416897 | 0.5456631379473927 | 14.498932480839267 | 74.12248011980388 | 68.9705949952886 | 13.445789050939664 | 13.502807299059265 | 2267.716796875 | 2924.0 |
| YOLOv8n | 3011238 | 5.24337152 | 0.47847372309787434 | 0.8918547443705646 | 0.6346684310374523 | 2.0049968985058912 | 1121.258821923698 | 498.75388871932546 | 0.811202738801167 | 0.9370074827516329 | 280.7607421875 | 392.0 |
| YOLOv8s | 11136374 | 18.332286976 | 0.4627343074309837 | 1.9921087596164404 | 0.5683826633565889 | 3.023225730404013 | 501.98062488944606 | 330.77252219150824 | 1.8470660570064001 | 2.011590328383647 | 530.30029296875 | 806.0 |
| YOLOv8m | 25857478 | 50.599723008 | 0.48366478129364826 | 4.224770255879683 | 0.5669572992236728 | 5.2753923363970046 | 236.69926160086064 | 189.55936094091183 | 4.076362591335401 | 4.2492704386109095 | 868.65771484375 | 1128.0 |
| YOLOv8l | 43631382 | 105.855483904 | 0.4648271889466602 | 6.907101279128017 | 0.5487642330831312 | 7.920692701157807 | 144.77853437907666 | 126.25158401282567 | 6.865877008922126 | 7.017352190854395 | 1741.0068359375 | 2028.0 |
| YOLOv8x | 68154534 | 165.195317248 | 0.48349142315913074 | 11.144143612086637 | 0.5494089407105353 | 12.177043975956302 | 89.73322982982981 | 82.12173676752012 | 11.032029380936883 | 11.460752373843379 | 2213.83642578125 | 2512.0 |

Waktu bervariasi antar-pengulangan (lihat kolom min/max); selisih kecil antar model tidak bermakna.

### Cek ulang mAP

mAP dihitung ulang dari matriks TP validator (`tables/02_cek_ulang_mAP.csv`), selisih absolut maksimum terhadap Ultralytics: 5.551115123125783e-17. Ini memastikan prediksi yang dipakai untuk bootstrap sama dengan yang dinilai Ultralytics.

## 3. Confusion matrix test penuh

conf ≥ 0.25, IoU > 0.5 (pencocokan seperti Ultralytics ConfusionMatrix). Angka mentah: `confusion_matrix/cm_<model>_test_penuh.csv` dan `tables/03_confusion_matrix_semua_model.csv`; PNG: `confusion_matrix/cm_<model>_test_penuh.png`. ★ = mAP@50 tertinggi (YOLOv12l). Format kolom `a->b` = GT a diprediksi b.

| model | kertas_plastik_puntungRokok->kertas_plastik_puntungRokok | kertas_plastik_puntungRokok->logam_kaleng_kaca | kertas_plastik_puntungRokok->latar | logam_kaleng_kaca->kertas_plastik_puntungRokok | logam_kaleng_kaca->logam_kaleng_kaca | logam_kaleng_kaca->latar | latar->kertas_plastik_puntungRokok | latar->logam_kaleng_kaca | latar->latar |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv12n | 693 | 30 | 275 | 37 | 65 | 27 | 176 | 33 | 0 |
| YOLOv12s | 716 | 33 | 249 | 39 | 68 | 22 | 146 | 27 | 0 |
| YOLOv12m | 721 | 28 | 249 | 45 | 63 | 21 | 125 | 32 | 0 |
| YOLOv12l ★ | 705 | 22 | 271 | 38 | 68 | 23 | 128 | 23 | 0 |
| YOLOv12x | 683 | 24 | 291 | 39 | 66 | 24 | 121 | 22 | 0 |
| YOLOv8n | 674 | 34 | 290 | 36 | 65 | 28 | 206 | 26 | 0 |
| YOLOv8s | 731 | 35 | 232 | 38 | 71 | 20 | 161 | 30 | 0 |
| YOLOv8m | 700 | 45 | 253 | 38 | 71 | 20 | 171 | 37 | 0 |
| YOLOv8l | 705 | 33 | 260 | 36 | 71 | 22 | 138 | 26 | 0 |
| YOLOv8x | 674 | 37 | 287 | 40 | 68 | 21 | 137 | 26 | 0 |

TP/FP/FN per kelas (`tables/03_tp_fp_fn_per_kelas_conf025.csv`). FN = terlewat + salah kelas; FP = latar + salah kelas.

| model | kelas | GT | TP | FN | FN_terlewat | FN_salah_kelas | FP | FP_latar | FP_salah_kelas | precision | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv12n | kertas_plastik_puntungRokok | 998 | 693 | 305 | 275 | 30 | 213 | 176 | 37 | 0.7649006622516556 | 0.6943887775551102 |
| YOLOv12n | logam_kaleng_kaca | 129 | 65 | 64 | 27 | 37 | 63 | 33 | 30 | 0.5078125 | 0.5038759689922481 |
| YOLOv12s | kertas_plastik_puntungRokok | 998 | 716 | 282 | 249 | 33 | 185 | 146 | 39 | 0.7946725860155383 | 0.717434869739479 |
| YOLOv12s | logam_kaleng_kaca | 129 | 68 | 61 | 22 | 39 | 60 | 27 | 33 | 0.53125 | 0.5271317829457365 |
| YOLOv12m | kertas_plastik_puntungRokok | 998 | 721 | 277 | 249 | 28 | 170 | 125 | 45 | 0.8092031425364759 | 0.7224448897795591 |
| YOLOv12m | logam_kaleng_kaca | 129 | 63 | 66 | 21 | 45 | 60 | 32 | 28 | 0.5121951219512195 | 0.4883720930232558 |
| YOLOv12l | kertas_plastik_puntungRokok | 998 | 705 | 293 | 271 | 22 | 166 | 128 | 38 | 0.8094144661308841 | 0.7064128256513026 |
| YOLOv12l | logam_kaleng_kaca | 129 | 68 | 61 | 23 | 38 | 45 | 23 | 22 | 0.6017699115044248 | 0.5271317829457365 |
| YOLOv12x | kertas_plastik_puntungRokok | 998 | 683 | 315 | 291 | 24 | 160 | 121 | 39 | 0.8102016607354685 | 0.6843687374749499 |
| YOLOv12x | logam_kaleng_kaca | 129 | 66 | 63 | 24 | 39 | 46 | 22 | 24 | 0.5892857142857143 | 0.5116279069767442 |
| YOLOv8n | kertas_plastik_puntungRokok | 998 | 674 | 324 | 290 | 34 | 242 | 206 | 36 | 0.7358078602620087 | 0.6753507014028056 |
| YOLOv8n | logam_kaleng_kaca | 129 | 65 | 64 | 28 | 36 | 60 | 26 | 34 | 0.52 | 0.5038759689922481 |
| YOLOv8s | kertas_plastik_puntungRokok | 998 | 731 | 267 | 232 | 35 | 199 | 161 | 38 | 0.7860215053763441 | 0.7324649298597194 |
| YOLOv8s | logam_kaleng_kaca | 129 | 71 | 58 | 20 | 38 | 65 | 30 | 35 | 0.5220588235294118 | 0.5503875968992248 |
| YOLOv8m | kertas_plastik_puntungRokok | 998 | 700 | 298 | 253 | 45 | 209 | 171 | 38 | 0.77007700770077 | 0.7014028056112225 |
| YOLOv8m | logam_kaleng_kaca | 129 | 71 | 58 | 20 | 38 | 82 | 37 | 45 | 0.46405228758169936 | 0.5503875968992248 |
| YOLOv8l | kertas_plastik_puntungRokok | 998 | 705 | 293 | 260 | 33 | 174 | 138 | 36 | 0.8020477815699659 | 0.7064128256513026 |
| YOLOv8l | logam_kaleng_kaca | 129 | 71 | 58 | 22 | 36 | 59 | 26 | 33 | 0.5461538461538461 | 0.5503875968992248 |
| YOLOv8x | kertas_plastik_puntungRokok | 998 | 674 | 324 | 287 | 37 | 177 | 137 | 40 | 0.7920094007050529 | 0.6753507014028056 |
| YOLOv8x | logam_kaleng_kaca | 129 | 68 | 61 | 21 | 40 | 63 | 26 | 37 | 0.5190839694656488 | 0.5271317829457365 |

## 4. Bootstrap berpasangan YOLOv12 vs YOLOv8

1000× resampling citra dengan pengembalian (seed 0), citra yang sama untuk kedua model. Δ = YOLOv12 − YOLOv8; CI 95% = persentil 2,5–97,5. mAP dari matriks TP validator (conf ≥ 0.001); FN/FP pada conf ≥ 0.25, IoU > 0.5. Untuk mAP, Δ positif = YOLOv12 lebih baik; untuk FN/FP, Δ negatif = YOLOv12 lebih baik. Kalau CI memuat 0, dicatat setara. Sumber: `tables/04_bootstrap_v12_vs_v8.csv`.

### YOLOv12n vs YOLOv8n (548 citra)

| metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50 | 0.6157665902323093 | 0.6229786550618368 | -0.00721206482952752 | -0.035948178527295684 | 0.02026627457908669 | setara (CI memuat 0) |
| mAP50_95 | 0.3583621655214536 | 0.346798201241887 | 0.01156396427956663 | -0.006631357769157384 | 0.028730418851202245 | setara (CI memuat 0) |

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN | 305.0 | 324.0 | -19.0 | -41.0 | 3.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP | 213.0 | 242.0 | -29.0 | -62.025 | 7.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN | 64.0 | 64.0 | 0.0 | -8.0 | 8.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP | 63.0 | 60.0 | 3.0 | -14.0 | 18.0 | setara (CI memuat 0) |
| TOTAL | FN | 369.0 | 388.0 | -19.0 | -43.0 | 3.0249999999999773 | setara (CI memuat 0) |
| TOTAL | FP | 276.0 | 302.0 | -26.0 | -62.0 | 13.024999999999977 | setara (CI memuat 0) |

Rincian FN terlewat vs salah kelas dan FP latar vs salah kelas:

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN_terlewat | 275.0 | 290.0 | -15.0 | -38.0 | 6.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FN_salah_kelas | 30.0 | 34.0 | -4.0 | -16.0 | 8.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP_latar | 176.0 | 206.0 | -30.0 | -64.0 | 6.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP_salah_kelas | 37.0 | 36.0 | 1.0 | -7.0 | 9.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_terlewat | 27.0 | 28.0 | -1.0 | -5.0 | 2.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_salah_kelas | 37.0 | 36.0 | 1.0 | -7.0 | 9.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_latar | 33.0 | 26.0 | 7.0 | -7.0 | 20.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_salah_kelas | 30.0 | 34.0 | -4.0 | -16.0 | 8.0 | setara (CI memuat 0) |
| TOTAL | FN_terlewat | 302.0 | 318.0 | -16.0 | -40.0 | 6.0 | setara (CI memuat 0) |
| TOTAL | FN_salah_kelas | 67.0 | 70.0 | -3.0 | -17.0 | 12.0 | setara (CI memuat 0) |
| TOTAL | FP_latar | 209.0 | 232.0 | -23.0 | -57.0 | 15.0 | setara (CI memuat 0) |
| TOTAL | FP_salah_kelas | 67.0 | 70.0 | -3.0 | -17.0 | 12.0 | setara (CI memuat 0) |

### YOLOv12s vs YOLOv8s (548 citra)

| metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50 | 0.6530196913979882 | 0.6507079239020179 | 0.002311767495970285 | -0.02343838955759875 | 0.026937745950977598 | setara (CI memuat 0) |
| mAP50_95 | 0.3954181884495703 | 0.3884243326085295 | 0.006993855841040786 | -0.008166097899438715 | 0.022119558048484516 | setara (CI memuat 0) |

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN | 282.0 | 267.0 | 15.0 | -4.0 | 33.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP | 185.0 | 199.0 | -14.0 | -41.025 | 15.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN | 61.0 | 58.0 | 3.0 | -6.0 | 12.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP | 60.0 | 65.0 | -5.0 | -21.0 | 10.0 | setara (CI memuat 0) |
| TOTAL | FN | 343.0 | 325.0 | 18.0 | -4.0 | 39.0 | setara (CI memuat 0) |
| TOTAL | FP | 245.0 | 264.0 | -19.0 | -55.0 | 17.024999999999977 | setara (CI memuat 0) |

Rincian FN terlewat vs salah kelas dan FP latar vs salah kelas:

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN_terlewat | 249.0 | 232.0 | 17.0 | 0.0 | 34.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FN_salah_kelas | 33.0 | 35.0 | -2.0 | -11.024999999999999 | 9.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP_latar | 146.0 | 161.0 | -15.0 | -41.0 | 13.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP_salah_kelas | 39.0 | 38.0 | 1.0 | -6.0 | 8.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_terlewat | 22.0 | 20.0 | 2.0 | -3.0 | 8.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_salah_kelas | 39.0 | 38.0 | 1.0 | -6.0 | 8.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_latar | 27.0 | 30.0 | -3.0 | -16.0 | 11.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_salah_kelas | 33.0 | 35.0 | -2.0 | -11.024999999999999 | 9.0 | setara (CI memuat 0) |
| TOTAL | FN_terlewat | 271.0 | 252.0 | 19.0 | 0.0 | 38.0 | setara (CI memuat 0) |
| TOTAL | FN_salah_kelas | 72.0 | 73.0 | -1.0 | -13.0 | 11.0 | setara (CI memuat 0) |
| TOTAL | FP_latar | 173.0 | 191.0 | -18.0 | -49.0 | 13.0 | setara (CI memuat 0) |
| TOTAL | FP_salah_kelas | 72.0 | 73.0 | -1.0 | -13.0 | 11.0 | setara (CI memuat 0) |

### YOLOv12m vs YOLOv8m (548 citra)

| metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50 | 0.648566626153444 | 0.6465209960640073 | 0.0020456300894367008 | -0.03044695361709506 | 0.03170961168819934 | setara (CI memuat 0) |
| mAP50_95 | 0.3932439948783493 | 0.39361876877373986 | -0.0003747738953905344 | -0.021800425819929647 | 0.017402048377887294 | setara (CI memuat 0) |

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN | 277.0 | 298.0 | -21.0 | -41.025 | -0.9750000000000227 | YOLOv12 lebih baik |
| kertas_plastik_puntungRokok | FP | 170.0 | 209.0 | -39.0 | -65.0 | -13.0 | YOLOv12 lebih baik |
| logam_kaleng_kaca | FN | 66.0 | 58.0 | 8.0 | -1.0 | 17.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP | 60.0 | 82.0 | -22.0 | -38.0 | -5.0 | YOLOv12 lebih baik |
| TOTAL | FN | 343.0 | 356.0 | -13.0 | -35.025 | 9.0 | setara (CI memuat 0) |
| TOTAL | FP | 230.0 | 291.0 | -61.0 | -92.0 | -30.0 | YOLOv12 lebih baik |

Rincian FN terlewat vs salah kelas dan FP latar vs salah kelas:

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN_terlewat | 249.0 | 253.0 | -4.0 | -22.0 | 14.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FN_salah_kelas | 28.0 | 45.0 | -17.0 | -29.0 | -7.0 | YOLOv12 lebih baik |
| kertas_plastik_puntungRokok | FP_latar | 125.0 | 171.0 | -46.0 | -73.0 | -19.0 | YOLOv12 lebih baik |
| kertas_plastik_puntungRokok | FP_salah_kelas | 45.0 | 38.0 | 7.0 | 0.0 | 15.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_terlewat | 21.0 | 20.0 | 1.0 | -4.0 | 7.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_salah_kelas | 45.0 | 38.0 | 7.0 | 0.0 | 15.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_latar | 32.0 | 37.0 | -5.0 | -19.0 | 9.024999999999977 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_salah_kelas | 28.0 | 45.0 | -17.0 | -29.0 | -7.0 | YOLOv12 lebih baik |
| TOTAL | FN_terlewat | 270.0 | 273.0 | -3.0 | -22.025 | 15.024999999999977 | setara (CI memuat 0) |
| TOTAL | FN_salah_kelas | 73.0 | 83.0 | -10.0 | -23.0 | 3.0 | setara (CI memuat 0) |
| TOTAL | FP_latar | 157.0 | 208.0 | -51.0 | -80.0 | -22.975000000000023 | YOLOv12 lebih baik |
| TOTAL | FP_salah_kelas | 73.0 | 83.0 | -10.0 | -23.0 | 3.0 | setara (CI memuat 0) |

### YOLOv12l vs YOLOv8l (548 citra)

| metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50 | 0.6673706586147301 | 0.6610413892352578 | 0.006329269379472313 | -0.01582586124266625 | 0.029190192224094837 | setara (CI memuat 0) |
| mAP50_95 | 0.4060917857457946 | 0.3987241325187082 | 0.007367653227086379 | -0.006199321725327644 | 0.021011783451148167 | setara (CI memuat 0) |

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN | 293.0 | 293.0 | 0.0 | -24.0 | 21.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP | 166.0 | 174.0 | -8.0 | -30.025 | 15.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN | 61.0 | 58.0 | 3.0 | -4.0 | 11.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP | 45.0 | 59.0 | -14.0 | -29.0 | -1.0 | YOLOv12 lebih baik |
| TOTAL | FN | 354.0 | 351.0 | 3.0 | -23.0 | 25.0 | setara (CI memuat 0) |
| TOTAL | FP | 211.0 | 233.0 | -22.0 | -48.025 | 6.0 | setara (CI memuat 0) |

Rincian FN terlewat vs salah kelas dan FP latar vs salah kelas:

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN_terlewat | 271.0 | 260.0 | 11.0 | -12.0 | 31.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FN_salah_kelas | 22.0 | 33.0 | -11.0 | -21.0 | -2.0 | YOLOv12 lebih baik |
| kertas_plastik_puntungRokok | FP_latar | 128.0 | 138.0 | -10.0 | -33.0 | 13.024999999999977 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP_salah_kelas | 38.0 | 36.0 | 2.0 | -5.0 | 9.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_terlewat | 23.0 | 22.0 | 1.0 | -5.0 | 8.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_salah_kelas | 38.0 | 36.0 | 2.0 | -5.0 | 9.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_latar | 23.0 | 26.0 | -3.0 | -16.0 | 10.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_salah_kelas | 22.0 | 33.0 | -11.0 | -21.0 | -2.0 | YOLOv12 lebih baik |
| TOTAL | FN_terlewat | 294.0 | 282.0 | 12.0 | -12.0 | 34.0 | setara (CI memuat 0) |
| TOTAL | FN_salah_kelas | 60.0 | 69.0 | -9.0 | -20.025 | 1.0 | setara (CI memuat 0) |
| TOTAL | FP_latar | 151.0 | 164.0 | -13.0 | -39.0 | 13.024999999999977 | setara (CI memuat 0) |
| TOTAL | FP_salah_kelas | 60.0 | 69.0 | -9.0 | -20.025 | 1.0 | setara (CI memuat 0) |

### YOLOv12x vs YOLOv8x (548 citra)

| metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|
| mAP50 | 0.6631210638342127 | 0.6473513407827314 | 0.015769723051481277 | -0.015664604368876456 | 0.04528182436315063 | setara (CI memuat 0) |
| mAP50_95 | 0.4004873410189428 | 0.38691855564068234 | 0.01356878537826045 | -0.007330391063533072 | 0.0322523871803574 | setara (CI memuat 0) |

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN | 315.0 | 324.0 | -9.0 | -28.025 | 11.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP | 160.0 | 177.0 | -17.0 | -43.025 | 10.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN | 63.0 | 61.0 | 2.0 | -6.0 | 11.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP | 46.0 | 63.0 | -17.0 | -33.025 | -1.0 | YOLOv12 lebih baik |
| TOTAL | FN | 378.0 | 385.0 | -7.0 | -29.025 | 16.0 | setara (CI memuat 0) |
| TOTAL | FP | 206.0 | 240.0 | -34.0 | -65.025 | -2.0 | YOLOv12 lebih baik |

Rincian FN terlewat vs salah kelas dan FP latar vs salah kelas:

| kelas | metrik | YOLOv12 | YOLOv8 | delta | ci_lo | ci_hi | kesimpulan |
|---|---|---|---|---|---|---|---|
| kertas_plastik_puntungRokok | FN_terlewat | 291.0 | 287.0 | 4.0 | -14.0 | 23.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FN_salah_kelas | 24.0 | 37.0 | -13.0 | -24.0 | -3.0 | YOLOv12 lebih baik |
| kertas_plastik_puntungRokok | FP_latar | 121.0 | 137.0 | -16.0 | -44.0 | 9.0 | setara (CI memuat 0) |
| kertas_plastik_puntungRokok | FP_salah_kelas | 39.0 | 40.0 | -1.0 | -9.0 | 7.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_terlewat | 24.0 | 21.0 | 3.0 | -2.0 | 8.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FN_salah_kelas | 39.0 | 40.0 | -1.0 | -9.0 | 7.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_latar | 22.0 | 26.0 | -4.0 | -16.0 | 9.0 | setara (CI memuat 0) |
| logam_kaleng_kaca | FP_salah_kelas | 24.0 | 37.0 | -13.0 | -24.0 | -3.0 | YOLOv12 lebih baik |
| TOTAL | FN_terlewat | 315.0 | 308.0 | 7.0 | -13.0 | 26.0 | setara (CI memuat 0) |
| TOTAL | FN_salah_kelas | 63.0 | 77.0 | -14.0 | -28.0 | 0.0 | setara (CI memuat 0) |
| TOTAL | FP_latar | 143.0 | 163.0 | -20.0 | -49.0 | 7.0 | setara (CI memuat 0) |
| TOTAL | FP_salah_kelas | 63.0 | 77.0 | -14.0 | -28.0 | 0.0 | setara (CI memuat 0) |

## 5. Analisis error model terbaik per keluarga

Model terbaik = mAP@50 test tertinggi dalam keluarganya: YOLOv12 → YOLOv12l, YOLOv8 → YOLOv8l. Recall = TP / GT pada conf ≥ 0.25, IoU > 0.5, kelas benar. Sumber dari pola nama file (sama dengan analisis sebelumnya): `batch_*`/`BATCH_*` = UAVVaste, nama numerik = TACO, `WhatsApp-Video*`/`IMG_*`/`GOPR*` = Burangrang, lainnya Unknown. Ukuran = luas bbox GT dalam piksel pada input 512 (letterbox), bin COCO.

### Per sumber (`tables/05_recall_per_sumber.csv`)

| model | sumber | citra | GT | TP | FN_terlewat | FN_salah_kelas | recall |
|---|---|---|---|---|---|---|---|
| YOLOv12l | Burangrang | 319 | 252.0 | 122.0 | 127.0 | 3.0 | 0.48412698412698413 |
| YOLOv12l | TACO | 31 | 69.0 | 50.0 | 17.0 | 2.0 | 0.7246376811594203 |
| YOLOv12l | UAVVaste | 195 | 803.0 | 600.0 | 148.0 | 55.0 | 0.7471980074719801 |
| YOLOv12l | Unknown | 3 | 3.0 | 1.0 | 2.0 | 0.0 | 0.3333333333333333 |
| YOLOv8l | Burangrang | 319 | 252.0 | 108.0 | 141.0 | 3.0 | 0.42857142857142855 |
| YOLOv8l | TACO | 31 | 69.0 | 47.0 | 18.0 | 4.0 | 0.6811594202898551 |
| YOLOv8l | UAVVaste | 195 | 803.0 | 620.0 | 121.0 | 62.0 | 0.772104607721046 |
| YOLOv8l | Unknown | 3 | 3.0 | 1.0 | 2.0 | 0.0 | 0.3333333333333333 |

### Per ukuran objek (`tables/05_recall_per_ukuran.csv`)

| model | ukuran | GT | TP | FN_terlewat | FN_salah_kelas | recall |
|---|---|---|---|---|---|---|
| YOLOv12l | kecil (<32²) | 908.0 | 610.0 | 245.0 | 53.0 | 0.6718061674008811 |
| YOLOv12l | sedang (32²–96²) | 193.0 | 141.0 | 48.0 | 4.0 | 0.7305699481865285 |
| YOLOv12l | besar (>96²) | 26.0 | 22.0 | 1.0 | 3.0 | 0.8461538461538461 |
| YOLOv8l | kecil (<32²) | 908.0 | 623.0 | 226.0 | 59.0 | 0.6861233480176211 |
| YOLOv8l | sedang (32²–96²) | 193.0 | 130.0 | 56.0 | 7.0 | 0.6735751295336787 |
| YOLOv8l | besar (>96²) | 26.0 | 23.0 | 0.0 | 3.0 | 0.8846153846153846 |

### Per ukuran dan kelas (`tables/05_recall_per_ukuran_kelas.csv`)

| model | ukuran | kelas | GT | TP | FN_terlewat | FN_salah_kelas | recall |
|---|---|---|---|---|---|---|---|
| YOLOv12l | kecil (<32²) | kertas_plastik_puntungRokok | 803.0 | 559.0 | 223.0 | 21.0 | 0.6961394769613948 |
| YOLOv12l | kecil (<32²) | logam_kaleng_kaca | 105.0 | 51.0 | 22.0 | 32.0 | 0.4857142857142857 |
| YOLOv12l | sedang (32²–96²) | kertas_plastik_puntungRokok | 175.0 | 128.0 | 47.0 | 0.0 | 0.7314285714285714 |
| YOLOv12l | sedang (32²–96²) | logam_kaleng_kaca | 18.0 | 13.0 | 1.0 | 4.0 | 0.7222222222222222 |
| YOLOv12l | besar (>96²) | kertas_plastik_puntungRokok | 20.0 | 18.0 | 1.0 | 1.0 | 0.9 |
| YOLOv12l | besar (>96²) | logam_kaleng_kaca | 6.0 | 4.0 | 0.0 | 2.0 | 0.6666666666666666 |
| YOLOv8l | kecil (<32²) | kertas_plastik_puntungRokok | 803.0 | 568.0 | 206.0 | 29.0 | 0.7073474470734745 |
| YOLOv8l | kecil (<32²) | logam_kaleng_kaca | 105.0 | 55.0 | 20.0 | 30.0 | 0.5238095238095238 |
| YOLOv8l | sedang (32²–96²) | kertas_plastik_puntungRokok | 175.0 | 118.0 | 54.0 | 3.0 | 0.6742857142857143 |
| YOLOv8l | sedang (32²–96²) | logam_kaleng_kaca | 18.0 | 12.0 | 2.0 | 4.0 | 0.6666666666666666 |
| YOLOv8l | besar (>96²) | kertas_plastik_puntungRokok | 20.0 | 19.0 | 0.0 | 1.0 | 0.95 |
| YOLOv8l | besar (>96²) | logam_kaleng_kaca | 6.0 | 4.0 | 0.0 | 2.0 | 0.6666666666666666 |

## 6. Gambar

- `figures/fig_map50_vs_inference_ms.png`: mAP@50 test vs ms inference (forward, batch 16), 600 dpi, lebar 3,45 inci, font serif.
- `figures/fig_kurva_training_terbaik.png`: kurva training YOLOv12l dan YOLOv8l (box loss, cls loss train/val; val mAP@50; val mAP@50-95), 600 dpi.
- `figures/fig_contoh_prediksi.png`: 5 contoh YOLOv12l (GT kiri, prediksi kanan), daftar di `tables/06_contoh_prediksi.csv`:

| contoh | jenis | file | sumber | GT | TP | FN_terlewat | FN_salah_kelas | FP_latar |
|---|---|---|---|---|---|---|---|---|
| 1 | benar semua | BATCH_d06_img_920_jpg.rf.7c509fa2cc59ee11fa1c25352aa60fc3.jpg | UAVVaste | 12 | 12 | 0 | 0 | 0 |
| 2 | benar semua | 000026_jpg.rf.6eea6a959012f9027fd1f3064a5afc2d.jpg | TACO | 3 | 3 | 0 | 0 | 0 |
| 3 | objek terlewat (FN) | IMG_5468_JPG_jpg.rf.105644c6e0432068ccad8f6bc8968743.jpg | Burangrang | 29 | 3 | 26 | 0 | 4 |
| 4 | salah kelas | BATCH_d08_img_1030_jpg.rf.0d74fd02c5e425f2f69e605e61560ae7.jpg | UAVVaste | 10 | 6 | 1 | 3 | 1 |
| 5 | deteksi palsu di latar (FP) | batch_s02_img_93_jpg.rf.797765599a111fbe43d4616c556dc418.jpg | UAVVaste | 3 | 3 | 0 | 0 | 5 |

## Isi folder

- `tables/` semua CSV; `confusion_matrix/` CSV + PNG per model; `figures/` PNG; `val/` keluaran Ultralytics; `salinan/` args.yaml dan results.csv tiap run, README.roboflow.txt, data.yaml, waktu_training_2kelas.txt; `scripts/` script evaluasi; `eval_01.log` log inferensi.
- `_cache/` (prediksi per citra, pickle) dan `_testset/` (hardlink test split) tidak dimasukkan ke zip.

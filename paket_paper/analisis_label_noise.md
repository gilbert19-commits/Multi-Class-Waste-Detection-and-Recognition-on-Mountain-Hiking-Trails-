# Analisis Label Noise: kertas/tisu ↔ plastik

Tidak ada training dan tidak ada perubahan dataset. Prediksi dari cache validator (13 model), pencocokan sama dengan confusion matrix di `02_error_analysis.py` (conf ≥ 0.25, IoU > 0.5). Sumber citra diambil dari `D0_scene_groups.csv` (pola nama file: `batch_*` = UAVVaste, numerik = TACO, `IMG_*`/`WhatsApp*`/`GOPR*` = Burangrang).

## Angka dasar

- Test clean: **530 kasus** kertas↔plastik, dijumlah atas 13 model (sama dengan angka 530 di `analisis_error.md`). Rata-rata 41 kasus per model.
- Kasus itu berasal dari **116 objek GT unik** di 63 citra. 35 objek (30%) dikelirukan oleh mayoritas model (≥ 7/13), jadi kekeliruannya konsisten, bukan kebetulan satu model.
- Arah: Kertas/Tisu → Plastik 352, Plastik → Kertas/Tisu 178.
- Objek dengan confidence > 0.5 (model mana pun): **64**. Contact sheet memuat 40 di antaranya (confidence tertinggi, maks. 3 objek per citra).

## Breakdown per sumber dataset

"kasus per GT per model" = kasus ÷ (jumlah GT kertas+plastik × jumlah model). Ini menormalkan perbedaan jumlah objek antar sumber.

### Test clean

| sumber | citra | GT kertas+plastik | % GT | kasus (13 model) | % kasus | objek unik tertukar | % objek GT tertukar | kasus per GT per model |
|---|---|---|---|---|---|---|---|---|
| UAVVaste | 54 | 152 | 48.1% | 388 | 73.2% | 78 | 51.3% | 0.196 |
| Burangrang | 125 | 139 | 44.0% | 134 | 25.3% | 34 | 24.5% | 0.074 |
| TACO | 11 | 22 | 7.0% | 8 | 1.5% | 4 | 18.2% | 0.028 |
| Unknown | 2 | 3 | 0.9% | 0 | 0.0% | 0 | 0.0% | 0.000 |

### Test full (pembanding, lebih banyak data)

| sumber | citra | GT kertas+plastik | % GT | kasus (13 model) | % kasus | objek unik tertukar | % objek GT tertukar | kasus per GT per model |
|---|---|---|---|---|---|---|---|---|
| UAVVaste | 195 | 697 | 70.0% | 1731 | 85.0% | 335 | 48.1% | 0.191 |
| Burangrang | 319 | 236 | 23.7% | 257 | 12.6% | 60 | 25.4% | 0.084 |
| TACO | 31 | 60 | 6.0% | 48 | 2.4% | 9 | 15.0% | 0.062 |
| Unknown | 3 | 3 | 0.3% | 0 | 0.0% | 0 | 0.0% | 0.000 |

### Arah kekeliruan per sumber (test clean, 13 model)

| sumber | Kertas/Tisu → Plastik | Plastik → Kertas/Tisu |
|---|---|---|
| UAVVaste | 248 | 140 |
| Burangrang | 100 | 34 |
| TACO | 4 | 4 |

### Objek dengan confidence > 0.5 per sumber (test clean)

| sumber | objek |
|---|---|
| UAVVaste | 37 |
| Burangrang | 23 |
| TACO | 4 |

### Per model (test clean)

| model | UAVVaste | Burangrang | TACO |
|---|---|---|---|
| YOLOv12l | 28 | 6 | 1 |
| YOLOv12l aug | 25 | 11 | 1 |
| YOLOv12m | 36 | 16 | 0 |
| YOLOv12m aug | 35 | 8 | 0 |
| YOLOv12n | 32 | 18 | 0 |
| YOLOv12n aug | 28 | 7 | 1 |
| YOLOv12s | 27 | 16 | 2 |
| YOLOv12s aug | 29 | 12 | 1 |
| YOLOv12x | 22 | 6 | 0 |
| YOLOv12x aug | 36 | 6 | 1 |
| YOLOv8m aug | 26 | 11 | 0 |
| YOLOv8n aug | 31 | 9 | 0 |
| YOLOv8s aug | 33 | 8 | 1 |

## Citra UAVVaste dengan salinan berlabel berbeda (audit D0)

- Citra UAVVaste di test clean: 54. Yang terlibat dalam kebingungan kertas↔plastik: **37**, menyumbang 388 dari 530 kasus.

| kondisi | citra | kasus |
|---|---|---|
| punya ≥ 2 salinan (nama sumber sama, seluruh dataset) | 24 | 231 |
| punya salinan di **train/valid** | 24 | 231 |
| salinan lainnya ada di pasangan konflik kelas D0 | 15 | 177 |
| citra test itu sendiri ada di pasangan konflik D0 | 1 | 1 |
| ...dengan konflik khusus kertas↔plastik | 0 | 0 |

- Test clean dibuang dari citra yang **satu grup scene SIFT** dengan train/valid. Salinan di atas lolos karena SIFT tidak menautkannya (varian rotasi/crop Roboflow), tetapi nama sumbernya sama. Artinya sebagian citra UAVVaste di test clean sebenarnya punya kembaran di train/valid.
- "Salinan lainnya ada di pasangan konflik" berarti salinan-salinan di train/valid dari adegan yang sama diberi kelas berbeda satu sama lain. Model dilatih dengan label yang saling bertentangan untuk adegan itu.
- Pembandingan label box per box antara salinan test dan salinan train belum dilakukan (butuh pemetaan homografi seperti di D0).

Detail per citra: `results/label_noise_uavvaste_salinan.csv`.

## Contact sheet dan penilaian visual

`results/figures/08_label_noise_sample.png`; daftar kasus di `results/label_noise_sample.csv`.

Penilaian visual atas 40 crop (`results/label_noise_penilaian_visual.csv`). Kategori: `label_salah` = objek jelas tampak kelas yang diprediksi model; `model_salah` = objek jelas tampak kelas GT; `ambigu` = objek terlihat tetapi bahannya (kertas atau plastik) tidak bisa dipastikan dari foto; `tak_terbaca` = terlalu kecil/buram untuk dinilai. Ini penilaian subjektif dari foto 640×640 dan perlu dicek ulang oleh anotator.

| penilaian | crop |
|---|---|
| ambigu | 19 |
| tak_terbaca | 13 |
| model_salah | 5 |
| label_salah | 3 |

| sumber | ambigu | label_salah | model_salah | tak_terbaca |
|---|---|---|---|---|
| Burangrang | 12 | 2 | 3 | 0 |
| TACO | 1 | 1 | 1 | 0 |
| UAVVaste | 6 | 0 | 1 | 13 |

| no | sumber | arah | conf | penilaian | catatan |
|---|---|---|---|---|---|
| 6 | TACO | Plastik → Kertas/Tisu | 0.869 | model_salah | botol semprot plastik ungu; label plastik benar |
| 7 | Burangrang | Plastik → Kertas/Tisu | 0.868 | model_salah | kantong plastik bening kusut; label plastik benar |
| 8 | Burangrang | Kertas/Tisu → Plastik | 0.843 | label_salah | bungkus snack mengkilap (film plastik/metalized) dilabeli kertas |
| 12 | UAVVaste | Plastik → Kertas/Tisu | 0.799 | model_salah | bentuk kantong plastik putih dengan pegangan; label plastik masuk akal |
| 14 | Burangrang | Plastik → Kertas/Tisu | 0.773 | model_salah | bungkus kusut biru-hijau tampak plastik; label plastik benar |
| 18 | TACO | Plastik → Kertas/Tisu | 0.750 | label_salah | karton/kotak kertas penyok dilabeli plastik |
| 31 | Burangrang | Plastik → Kertas/Tisu | 0.677 | model_salah | wadah/nampan putih (foil atau plastik); bukan kertas |
| 40 | Burangrang | Kertas/Tisu → Plastik | 0.595 | label_salah | lembaran putih kusut transparan tampak kantong plastik; dilabeli kertas |

## Kesimpulan

**Hipotesis "sebagian besar dari UAVVaste": terbukti.** Di test clean UAVVaste menyumbang 73% kasus kertas↔plastik padahal hanya 48% dari objek GT kertas+plastik (test full: 85% dari 70%). Setelah dinormalkan per objek, laju kekeliruan UAVVaste 0.196 per GT per model, 2.6× Burangrang (0.074). Jadi dominasinya bukan sekadar karena UAVVaste punya objek lebih banyak.

**Apakah labelnya yang salah: belum bisa dibuktikan untuk sebagian besar kasus.** Dari 40 kasus ber-confidence tinggi, hanya 3 yang dari foto jelas tampak salah label, dan 5 jelas kesalahan model (label benar, mis. botol semprot plastik diprediksi kertas). Sisanya 32 ambigu atau tak terbaca: objek UAVVaste umumnya 10–35 px dari ketinggian drone, sehingga kertas dan plastik putih tidak bisa dibedakan secara visual, baik oleh model maupun oleh anotator. Confidence tinggi tidak berarti label salah; model juga bisa yakin dan keliru.

**Bukti tidak langsung label noise UAVVaste kuat.** 15 dari 37 citra UAVVaste yang terlibat (177 dari 388 kasusnya) berasal dari adegan yang salinan-salinannya di train/valid diberi kelas saling bertentangan (audit D0). Untuk objek sekecil itu, label kertas vs plastik kemungkinan besar tebakan anotator, dan tebakannya tidak konsisten antar salinan.

**Ringkasnya:** kebingungan kertas↔plastik di UAVVaste paling tepat disebut *ambiguitas kelas pada objek kecil yang diperparah label tidak konsisten*, bukan murni kesalahan model dan bukan terbukti murni salah label. Di Burangrang dan TACO (objek besar, jelas terlihat) contoh kesalahan model yang nyata memang ada. Implikasinya: metrik per-kelas kertas_tisu dan plastik punya batas atas yang rendah selama label UAVVaste tidak direvisi, dan selisih kecil antar model pada dua kelas ini sebaiknya tidak ditafsirkan berlebihan.

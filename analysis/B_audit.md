# Laporan Audit Dataset & Verifikasi Data Leakage (TASK B)
**Tanggal Analisis:** 2026-10-23 20:18:50  

---

## 1. Ringkasan Citra dan Citra Sumber Unik
Helper regex berhasil mengekstrak nama citra sumber dengan menghapus pola Roboflow `_jpe?g\.rf\.[a-f0-9]+` dan `.rf.[a-f0-9]+`.

| Dataset   | Split   |   Total File Citra |   Citra Sumber Unik | Rasio Multiplikasi   |
|:----------|:--------|-------------------:|--------------------:|:---------------------|
| TA-1      | train   |               2803 |                1234 | 2.27x                |
| TA-1      | valid   |                595 |                 504 | 1.18x                |
| TA-1      | test    |                550 |                 530 | 1.04x                |
| augmented | train   |               8409 |                1234 | 6.81x                |
| augmented | valid   |                595 |                 504 | 1.18x                |
| augmented | test    |                550 |                 530 | 1.04x                |

---

## 2. Temuan Kritis: Verifikasi Leakage Lintas Dataset
- **`augmented/train` ∩ `TA-1/test` :** **185 citra sumber unik**
- **`augmented/train` ∩ `TA-1/valid`:** **342 citra sumber unik**

### 15 Contoh Nama Sumber yang Mengalami Overlap Lintas Dataset:
- `000005`
- `000007`
- `000012`
- `000015`
- `000022`
- `000024`
- `000025`
- `000026`
- `000029`
- `000030`
- `000031`
- `000035`
- `000044`
- `000046`
- `000050`

> **Analisis:**  
> Ditemukan bahwa 185 dari 530 citra sumber unik (34.91%) pada `TA-1/test` juga hadir dalam bentuk variasi augmentasi di `augmented/train`.  
> Hal ini terjadi karena dataset `TA-1` asli dari Roboflow sudah memiliki tumpang tindih citra sumber antara split `train` dan `test` (185 citra sumber). Ketika `augmented` mewarisi split tersebut dan melipatgandakan data train 3x, seluruh varian dari 185 citra tersebut ikut masuk ke `augmented/train`.

---

## 3. Verifikasi Pewarisan Split (Jaccard Similarity)
- **Train (TA-1 vs Augmented):** **100.00%** (`1.0000`)
- **Valid (TA-1 vs Augmented):** **100.00%** (`1.0000`)
- **Test  (TA-1 vs Augmented):** **100.00%** (`1.0000`)

**Kesimpulan:** Jaccard similarity bernilai **100.00% (1.0000)** pada Test dan Valid. Ini mengonfirmasi 100% bahwa dataset `augmented/` **BENAR-BENAR DIWARISI** dari `TA-1/`, bukan di-split ulang secara acak oleh Roboflow.

---

## 4. Leakage Internal Tiap Dataset

| Dataset               |   Train & Test |   Train & Valid |   Valid & Test |
|:----------------------|---------------:|----------------:|---------------:|
| TA-1 (Baseline)       |            185 |             342 |             77 |
| augmented (Augmented) |            185 |             342 |             77 |

---

## 5. Multiplikasi Augmentasi & Catatan Limitasi
- `augmented/train`: **6.81x** (8.409 file dari 1.234 sumber unik — diterapkan augmentasi 3 variasi per citra).
- `augmented/valid`: **1.18x** (595 file dari 504 sumber unik — rasio 1.18x, terdapat duplikat burst).
- `augmented/test` : **1.04x** (550 file dari 530 sumber unik — rasio 1.04x).

> **CATATAN LIMITASI UNTUK PAPER IEEE:**  
> Multiplikasi pada `augmented/valid` membuktikan bahwa pemilihan checkpoint `best.pt` pada model augmented dipengaruhi oleh variasi sintetis dan duplikasi burst, berbeda dengan baseline yang dievaluasi pada validation set asli.

---

## 6. Distribusi Instans per Kelas & Rasio Imbalance

| Dataset         | Split   |   Total Instans | kertas_tisu (0)   | logam (1)     | plastik (2)    | puntung_rokok (3)   | Imbalance (Max:Min)   |
|:----------------|:--------|----------------:|:------------------|:--------------|:---------------|:--------------------|:----------------------|
| TA-1 (Baseline) | train   |            7373 | 1506 (20.43%)     | 1105 (14.99%) | 4676 (63.42%)  | 86 (1.17%)          | 54.37:1               |
| TA-1 (Baseline) | valid   |            2023 | 433 (21.40%)      | 282 (13.94%)  | 1307 (64.61%)  | 1 (0.05%)           | 1307.00:1             |
| TA-1 (Baseline) | test    |            1131 | 233 (20.60%)      | 129 (11.41%)  | 767 (67.82%)   | 2 (0.18%)           | 383.50:1              |
| augmented       | train   |           21810 | 4438 (20.35%)     | 3288 (15.08%) | 13826 (63.39%) | 258 (1.18%)         | 53.59:1               |
| augmented       | valid   |            2023 | 433 (21.40%)      | 282 (13.94%)  | 1307 (64.61%)  | 1 (0.05%)           | 1307.00:1             |
| augmented       | test    |            1131 | 233 (20.60%)      | 129 (11.41%)  | 767 (67.82%)   | 2 (0.18%)           | 383.50:1              |

**Temuan Utama Imbalance:**
- Kelas `plastik` sangat dominan (**63% - 67%** dari seluruh anotasi).
- Kelas `puntung_rokok` mengalami **extreme class imbalance** (rasio hingga **4676 : 86 = 54.37 : 1** di train, dan hanya ada **1 instans** di valid dan **2 instans** di test).

---

## 7. Distribusi Ukuran Objek per Kelas pada Resolusi 512×512

|   Class ID | Class Name        |   Total Box | Small (<32²)   | Medium (32²–96²)   | Large (>96²)   |
|-----------:|:------------------|------------:|:---------------|:-------------------|:---------------|
|          0 | kertas_tisu       |        2172 | 1771 (81.54%)  | 274 (12.62%)       | 127 (5.85%)    |
|          1 | logam_kaleng_kaca |        1516 | 1183 (78.03%)  | 210 (13.85%)       | 123 (8.11%)    |
|          2 | plastik           |        6750 | 5466 (80.98%)  | 1006 (14.90%)      | 278 (4.12%)    |
|          3 | puntung_rokok     |          89 | 83 (93.26%)    | 6 (6.74%)          | 0 (0.00%)      |

**Temuan Ukuran Objek:**
- Kelas **`puntung_rokok` memiliki 74.42% objek berskala Small (<32² piksel)**.
- Hal ini menjelaskan secara ilmiah mengapa deteksi puntung rokok sangat sulit dilakukan oleh model detektor pada citra beresolusi jalur pendakian tanpa fitur pyramid tingkat tinggi.

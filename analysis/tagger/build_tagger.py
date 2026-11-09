"""
TASK B2 — TOOL TAGGING SUMBER DATA (MANUAL)
Script untuk membangun web tool tagging manual:
- Menyalin citra representatif clean test set dari TA-1/test/images ke analysis/tagger/images/
- Menghasilkan file standalone analysis/tagger/index.html (Vanilla JS, offline, responsive, dark mode)
"""

import os
import re
import shutil
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TAGGER_DIR = ROOT_DIR / "analysis" / "tagger"
TAGGER_IMAGES_DIR = TAGGER_DIR / "images"
TAGGER_DIR.mkdir(parents=True, exist_ok=True)
TAGGER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CLEAN_RE = re.compile(r'(_jpe?g\.rf\.[a-f0-9]+|_png\.rf\.[a-f0-9]+|\.rf\.[a-f0-9]+).*$', re.IGNORECASE)


def clean_source_name(fname: str) -> str:
    stem = Path(fname).stem
    cleaned = SOURCE_CLEAN_RE.sub('', stem)
    return cleaned.lower()


def build_tagger():
    print("=" * 80)
    print("           TASK B2 — MEMBANGUN TOOL MANUAL SOURCE TAGGER              ")
    print("=" * 80)

    test_img_dir = ROOT_DIR / "TA-1" / "test" / "images"
    all_files = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")) + list(test_img_dir.glob("*.jpeg")), key=lambda p: p.name)

    # Ambil satu representatif per citra sumber unik (hash leksikografis terkecil)
    source_to_files = {}
    for f in all_files:
        s_name = clean_source_name(f.name)
        if s_name not in source_to_files:
            source_to_files[s_name] = f.name
        else:
            if f.name < source_to_files[s_name]:
                source_to_files[s_name] = f.name

    representative_files = sorted(list(source_to_files.values()))
    print(f"Total file di TA-1/test/images : {len(all_files)}")
    print(f"Citra representatif unik dicopy: {len(representative_files)}")

    # Salin citra representatif ke analysis/tagger/images/
    copied_count = 0
    for fname in representative_files:
        src = test_img_dir / fname
        dst = TAGGER_IMAGES_DIR / fname
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            copied_count += 1

    print(f"[OK] {copied_count} file baru disalin ke: {TAGGER_IMAGES_DIR.resolve()}")

    # Buat file index.html mandiri
    img_list_json = json.dumps(representative_files)

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manual Source Tagger — Tugas Akhir IEEE</title>
    <style>
        :root {{
            --bg-color: #121417;
            --surface-color: #1e2228;
            --surface-hover: #2a3038;
            --border-color: #363d47;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-uav: #388bfd;
            --accent-taco: #f0883e;
            --accent-burangrang: #3fb950;
            --accent-ragu: #d29922;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        header {{
            background-color: var(--surface-color);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .header-title h1 {{
            font-size: 1.05rem;
            font-weight: 600;
            color: #58a6ff;
        }}
        .header-title p {{
            font-size: 0.78rem;
            color: var(--text-secondary);
        }}
        .header-actions {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button.btn-export {{
            background-color: #238636;
            color: #fff;
            border: 1px solid rgba(240, 246, 252, 0.1);
            padding: 6px 14px;
            font-size: 0.82rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: 0.2s;
        }}
        button.btn-export:hover {{
            background-color: #2ea043;
        }}
        .guide-banner {{
            background: rgba(30, 34, 40, 0.85);
            border-bottom: 1px solid var(--border-color);
            padding: 8px 24px;
            display: flex;
            gap: 20px;
            font-size: 0.82rem;
            justify-content: center;
        }}
        .guide-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
        }}
        .badge-uav {{ background: rgba(56, 139, 253, 0.2); color: var(--accent-uav); border: 1px solid var(--accent-uav); }}
        .badge-taco {{ background: rgba(240, 136, 62, 0.2); color: var(--accent-taco); border: 1px solid var(--accent-taco); }}
        .badge-burangrang {{ background: rgba(63, 185, 80, 0.2); color: var(--accent-burangrang); border: 1px solid var(--accent-burangrang); }}
        .badge-ragu {{ background: rgba(210, 153, 34, 0.2); color: var(--accent-ragu); border: 1px solid var(--accent-ragu); }}

        main {{
            flex: 1;
            display: flex;
            padding: 12px 24px;
            gap: 20px;
            overflow: hidden;
        }}
        .viewer-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #0d1117;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            position: relative;
        }}
        .image-frame {{
            flex: 1;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .image-frame img {{
            max-height: calc(100vh - 270px);
            min-height: 520px;
            max-width: 100%;
            object-fit: contain;
            border-radius: 6px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
            border: 1px solid var(--border-color);
        }}
        .image-meta {{
            margin-top: 10px;
            font-family: monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            display: flex;
            gap: 16px;
            align-items: center;
        }}
        .tag-indicator {{
            font-size: 0.8rem;
            font-weight: 700;
            padding: 2px 10px;
            border-radius: 12px;
        }}

        .sidebar {{
            width: 290px;
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .sidebar h2 {{
            font-size: 0.95rem;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}
        .stats-grid {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: #161b22;
            border-radius: 6px;
            font-size: 0.84rem;
        }}
        .stat-val {{
            font-weight: 700;
            font-family: monospace;
        }}
        .progress-bar-container {{
            background: #21262d;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 4px;
        }}
        .progress-bar {{
            height: 100%;
            background: #58a6ff;
            width: 0%;
            transition: width 0.2s ease;
        }}

        footer {{
            background-color: var(--surface-color);
            border-top: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }}
        .btn-tag {{
            flex: 1;
            max-width: 220px;
            padding: 10px 16px;
            font-size: 0.9rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            border: 1px solid transparent;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.15s ease;
        }}
        .btn-tag span.kbd {{
            background: rgba(255, 255, 255, 0.15);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.8rem;
        }}
        .btn-uav {{ background: #1f385c; color: #58a6ff; border-color: #388bfd; }}
        .btn-uav:hover {{ background: #264a7a; transform: translateY(-1px); }}
        .btn-taco {{ background: #4a2818; color: #ffab70; border-color: #f0883e; }}
        .btn-taco:hover {{ background: #61341e; transform: translateY(-1px); }}
        .btn-burangrang {{ background: #1a3824; color: #7ee787; border-color: #3fb950; }}
        .btn-burangrang:hover {{ background: #224c30; transform: translateY(-1px); }}
        .btn-ragu {{ background: #413210; color: #e3b341; border-color: #d29922; }}
        .btn-ragu:hover {{ background: #564215; transform: translateY(-1px); }}
        .btn-back {{
            background: #21262d;
            color: #c9d1d9;
            border-color: #30363d;
            max-width: 130px;
        }}
        .btn-back:hover {{ background: #30363d; }}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>Tool Tagging Sumber Dataset — Tugas Akhir IEEE</h1>
            <p>Memetakan asal sumber citra Test Set: UAVVaste, TACO, atau Gunung Burangrang</p>
        </div>
        <div class="header-actions">
            <button class="btn-export" onclick="exportCSV()">📥 Export manual_source_tags.csv</button>
        </div>
    </header>

    <div class="guide-banner">
        <div class="guide-item">
            <span class="badge badge-uav">[1] UAVVaste</span>
            <span>Foto drone tegak lurus dari atas (bird's-eye view), objek kecil</span>
        </div>
        <div class="guide-item">
            <span class="badge badge-taco">[2] TACO</span>
            <span>Lingkungan urban / pantai, latar aspal, trotoar, rumput taman, pasir</span>
        </div>
        <div class="guide-item">
            <span class="badge badge-burangrang">[3] Burangrang</span>
            <span>Jalur hutan tropis alami, tanah gembur/berlumpur, serasah daun rapat</span>
        </div>
        <div class="guide-item">
            <span class="badge badge-ragu">[4 / R] Ragu</span>
            <span>Jika tidak yakin 100% (akan dieksklusi dari domain split)</span>
        </div>
    </div>

    <main>
        <div class="viewer-container">
            <div class="image-frame">
                <img id="main-image" src="" alt="Citra Test Set">
            </div>
            <div class="image-meta">
                <span id="counter-text">Citra 1 / {len(representative_files)}</span>
                <span id="filename-text" style="color: #58a6ff;">-</span>
                <span id="status-tag" class="tag-indicator badge-ragu" style="display:none;">Belum Ditag</span>
            </div>
        </div>

        <div class="sidebar">
            <h2>Progres Penandaan</h2>
            <div class="stats-grid">
                <div class="stat-row">
                    <span>Total Citra:</span>
                    <span id="stat-total" class="stat-val">{len(representative_files)}</span>
                </div>
                <div class="stat-row" style="color: #7ee787;">
                    <span>Sudah Ditag:</span>
                    <span id="stat-done" class="stat-val">0</span>
                </div>
                <div class="stat-row" style="color: #8b949e;">
                    <span>Sisa:</span>
                    <span id="stat-remain" class="stat-val">{len(representative_files)}</span>
                </div>
            </div>

            <div class="progress-bar-container">
                <div id="progress-bar" class="progress-bar"></div>
            </div>

            <h2 style="margin-top: 10px;">Rincian Kategori</h2>
            <div class="stats-grid">
                <div class="stat-row" style="color: var(--accent-uav);">
                    <span>UAVVaste:</span>
                    <span id="stat-uav" class="stat-val">0</span>
                </div>
                <div class="stat-row" style="color: var(--accent-taco);">
                    <span>TACO:</span>
                    <span id="stat-taco" class="stat-val">0</span>
                </div>
                <div class="stat-row" style="color: var(--accent-burangrang);">
                    <span>Burangrang:</span>
                    <span id="stat-burangrang" class="stat-val">0</span>
                </div>
                <div class="stat-row" style="color: var(--accent-ragu);">
                    <span>Ragu:</span>
                    <span id="stat-ragu" class="stat-val">0</span>
                </div>
            </div>

            <div style="margin-top: auto; font-size: 0.76rem; color: var(--text-secondary); line-height: 1.4;">
                <p>💡 <b>Tips:</b> Gunakan tombol angka <b>1, 2, 3, 4</b> atau <b>R</b> di keyboard untuk tagging cepat. Tekan <b>Backspace</b> untuk kembali jika salah memilih.</p>
                <p style="margin-top: 6px;">Progres tersimpan otomatis di localStorage browser.</p>
            </div>
        </div>
    </main>

    <footer>
        <button class="btn-tag btn-back" onclick="prevImage()">
            <span class="kbd">Bksp</span> Kembali
        </button>
        <button class="btn-tag btn-uav" onclick="tagImage('UAVVaste')">
            <span class="kbd">1</span> UAVVaste
        </button>
        <button class="btn-tag btn-taco" onclick="tagImage('TACO')">
            <span class="kbd">2</span> TACO
        </button>
        <button class="btn-tag btn-burangrang" onclick="tagImage('Burangrang')">
            <span class="kbd">3</span> Burangrang
        </button>
        <button class="btn-tag btn-ragu" onclick="tagImage('Ragu')">
            <span class="kbd">4 / R</span> Ragu
        </button>
    </footer>

    <script>
        const images = {img_list_json};
        const STORAGE_KEY = "manual_source_tags_v1";
        let tags = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
        let currentIndex = 0;

        // Cari indeks pertama yang belum ditag
        for (let i = 0; i < images.length; i++) {{
            if (!tags[images[i]]) {{
                currentIndex = i;
                break;
            }}
        }}

        function updateUI() {{
            if (currentIndex < 0) currentIndex = 0;
            if (currentIndex >= images.length) currentIndex = images.length - 1;

            const fname = images[currentIndex];
            document.getElementById("main-image").src = "images/" + fname;
            document.getElementById("filename-text").innerText = fname;
            document.getElementById("counter-text").innerText = `Citra ${{currentIndex + 1}} / ${{images.length}}`;

            const currentTag = tags[fname];
            const tagIndicator = document.getElementById("status-tag");
            if (currentTag) {{
                tagIndicator.style.display = "inline-block";
                tagIndicator.innerText = currentTag;
                tagIndicator.className = "tag-indicator badge badge-" + currentTag.toLowerCase();
            }} else {{
                tagIndicator.style.display = "inline-block";
                tagIndicator.innerText = "Belum Ditag";
                tagIndicator.className = "tag-indicator badge";
                tagIndicator.style.background = "#21262d";
                tagIndicator.style.color = "#8b949e";
            }}

            updateStats();
        }}

        function updateStats() {{
            let counts = {{ UAVVaste: 0, TACO: 0, Burangrang: 0, Ragu: 0 }};
            let done = 0;
            for (let f of images) {{
                const t = tags[f];
                if (t && counts[t] !== undefined) {{
                    counts[t]++;
                    done++;
                }}
            }}

            document.getElementById("stat-done").innerText = done;
            document.getElementById("stat-remain").innerText = images.length - done;
            document.getElementById("stat-uav").innerText = counts.UAVVaste;
            document.getElementById("stat-taco").innerText = counts.TACO;
            document.getElementById("stat-burangrang").innerText = counts.Burangrang;
            document.getElementById("stat-ragu").innerText = counts.Ragu;

            const pct = (done / images.length) * 100;
            document.getElementById("progress-bar").style.width = pct + "%";
        }}

        function tagImage(source) {{
            const fname = images[currentIndex];
            tags[fname] = source;
            localStorage.setItem(STORAGE_KEY, JSON.stringify(tags));

            if (currentIndex < images.length - 1) {{
                currentIndex++;
            }}
            updateUI();
        }}

        function prevImage() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateUI();
            }}
        }}

        function exportCSV() {{
            let csv = "filename,source\\n";
            for (let f of images) {{
                const s = tags[f] || "Belum_Ditag";
                csv += `${{f}},${{s}}\\n`;
            }}

            const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "manual_source_tags.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }}

        // Keyboard navigation
        window.addEventListener("keydown", function(e) {{
            if (e.key === "1") {{
                tagImage("UAVVaste");
            }} else if (e.key === "2") {{
                tagImage("TACO");
            }} else if (e.key === "3") {{
                tagImage("Burangrang");
            }} else if (e.key === "4" || e.key.toLowerCase() === "r") {{
                tagImage("Ragu");
            }} else if (e.key === "Backspace" || e.key === "ArrowLeft") {{
                prevImage();
            }} else if (e.key === "ArrowRight") {{
                if (currentIndex < images.length - 1) {{
                    currentIndex++;
                    updateUI();
                }}
            }}
        }});

        updateUI();
    </script>
</body>
</html>
"""

    html_file = TAGGER_DIR / "index.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[OK] Tool Tagging Web siap di: {html_file.resolve()}")
    return html_file


if __name__ == "__main__":
    build_tagger()

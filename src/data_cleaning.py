"""
==============================================================
 data_cleaning.py — Capstone Project: CV Classification (NLP)
==============================================================
 Pipeline lengkap dari PDF mentah → Dataset siap model:
   TAHAP 1 : Konfigurasi path & tools (OCR)
   TAHAP 2 : Ekstraksi teks dari PDF (per kategori)
   TAHAP 3 : Pembersihan & normalisasi teks
   TAHAP 4 : Feature engineering
   TAHAP 5 : Restrukturisasi & export ke Excel

 Cara pakai:
   1. Sesuaikan KONFIGURASI di bagian bawah jika perlu
   2. Jalankan: python src/data_cleaning.py
   3. Output  : data/processed/Dataset_NLP_Siap_Model.xlsx
==============================================================
"""

import os
import re
import pandas as pd
from pdf2image import convert_from_path
import pytesseract

# ==============================================================
# KONFIGURASI — Sesuaikan dengan setup lokal kamu
# ==============================================================

# Path ke executable Tesseract-OCR
# Windows : r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux   : '/usr/bin/tesseract'  (atau cek: which tesseract)
# Mac     : '/usr/local/bin/tesseract'
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path ke folder bin Poppler (Windows saja, Linux/Mac biarkan None)
# Contoh  : r'C:\poppler-25.12.0\Library\bin'
# Linux/Mac: None
POPPLER_PATH = r'C:\poppler-25.12.0\Library\bin'

# Root folder proyek (relatif dari lokasi script ini dijalankan)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder input (PDF mentah) — key = nama folder, value = nama kategori di dataset
KATEGORI_FOLDER = {
    'BusinessAnalyst'           : 'Business Analyst',
    'Data_Science'              : 'Data Science',
    'JavaDeveloper'             : 'Java Developer',
    'Network_Security_Engineer' : 'Network Security Engineer',
    'React_Developer'           : 'React Developer',
    'SAP_Developer'             : 'SAP Developer',
    'SQL'                       : 'SQL Developer',
    'web_designing'             : 'Web Designing',
}

# Folder output
RAW_DIR       = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, 'Dataset_NLP_Siap_Model.xlsx')

# Konfigurasi OCR Tesseract
# --psm 4 : asumsi satu kolom teks berukuran variasi
# --psm 6 : asumsi blok teks seragam (coba jika hasil kurang bagus)
TESSERACT_CONFIG = r'--psm 4'

# ==============================================================
# TAHAP 1 — INISIALISASI
# ==============================================================

def inisialisasi():
    """Siapkan environment: cek path, buat folder output jika belum ada."""
    print("=" * 60)
    print(" CAPSTONE — CV CLASSIFICATION DATA CLEANING PIPELINE")
    print("=" * 60)

    # Set Tesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    # Buat folder processed jika belum ada
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Validasi folder raw
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(
            f"Folder data/raw tidak ditemukan: {RAW_DIR}\n"
            "Pastikan kamu menjalankan script dari root folder proyek."
        )

    print(f"✅ BASE_DIR      : {BASE_DIR}")
    print(f"✅ RAW_DIR       : {RAW_DIR}")
    print(f"✅ PROCESSED_DIR : {PROCESSED_DIR}")
    print()


# ==============================================================
# TAHAP 2 — EKSTRAKSI TEKS DARI PDF (OCR)
# ==============================================================

def ekstrak_teks_dari_pdf(lokasi_pdf: str) -> str:
    """
    Konversi satu file PDF ke string teks menggunakan OCR.
    Hanya memproses halaman pertama untuk efisiensi.
    
    Returns:
        str: teks mentah hasil OCR, atau string kosong jika gagal.
    """
    try:
        halaman_gambar = convert_from_path(
            lokasi_pdf,
            poppler_path=POPPLER_PATH,
            dpi=200,          # resolusi lebih tinggi = OCR lebih akurat
            first_page=1,
            last_page=1       # hanya halaman pertama
        )
        hasil_teks = pytesseract.image_to_string(
            halaman_gambar[0],
            config=TESSERACT_CONFIG
        )
        return hasil_teks
    except Exception as e:
        print(f"      ⚠️  ERROR OCR: {e}")
        return ""


def ekstrak_semua_kategori() -> list[dict]:
    """
    Loop semua folder kategori, ekstrak teks dari setiap PDF,
    kembalikan list of dict: [{'Category': ..., 'Text': ...}, ...]
    """
    semua_data = []

    for nama_folder, nama_kategori in KATEGORI_FOLDER.items():
        folder_pdf = os.path.join(RAW_DIR, nama_folder)

        if not os.path.exists(folder_pdf):
            print(f"⚠️  Folder tidak ditemukan, skip: {folder_pdf}")
            continue

        file_pdf = sorted([f for f in os.listdir(folder_pdf) if f.endswith('.pdf')])
        print(f"\n📂 [{nama_kategori}] — {len(file_pdf)} file PDF ditemukan")

        for idx, nama_file in enumerate(file_pdf, 1):
            lokasi_pdf = os.path.join(folder_pdf, nama_file)
            print(f"   [{idx:>3}/{len(file_pdf)}] Ekstrak: {nama_file} ... ", end='', flush=True)

            teks = ekstrak_teks_dari_pdf(lokasi_pdf)

            if teks.strip():
                semua_data.append({
                    'Category' : nama_kategori,
                    'Text'     : teks
                })
                print("✅")
            else:
                print("⛔ (teks kosong, skip)")

    print(f"\n✅ Total CV berhasil diekstrak: {len(semua_data)}")
    return semua_data


# ==============================================================
# TAHAP 3 — PEMBERSIHAN & NORMALISASI TEKS
# ==============================================================

def bersihkan_teks(teks_mentah: str) -> str:
    """
    Bersihkan teks hasil OCR:
    - Hapus karakter unicode tersembunyi (BOM, ZWNJ, dll)
    - Hapus simbol OCR noise, sisakan huruf/angka/tanda baca penting
    - Rapikan spasi berlebihan
    
    Returns:
        str: teks yang sudah bersih
    """
    if not isinstance(teks_mentah, str):
        teks_mentah = str(teks_mentah)

    # Hapus karakter unicode tersembunyi
    teks = re.sub(r'[\ufeff\u200b\u200c\u200d\u00a0]', ' ', teks_mentah)

    # Hapus karakter simbol OCR noise
    # Sisakan: huruf, angka, spasi, titik, koma, titik dua, strip, garis miring, @, +
    teks = re.sub(r'[^a-zA-Z0-9\s.,\-\/|@+:;()]', ' ', teks)

    # Rapikan spasi berlebihan & strip
    teks = re.sub(r'\s+', ' ', teks).strip()

    return teks


def terapkan_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan fungsi bersihkan_teks ke kolom Text → kolom Clean_Text."""
    print("\n🔧 Membersihkan teks...")
    df = df.copy()
    df['Clean_Text'] = df['Text'].apply(bersihkan_teks)
    print(f"   ✅ Kolom Clean_Text berhasil dibuat")
    return df


# ==============================================================
# TAHAP 4 — FEATURE ENGINEERING
# ==============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan fitur-fitur turunan:
    - Word_Count : jumlah kata di Clean_Text
    - ID_CV      : ID unik format CV_XXXX
    """
    print("\n🔧 Feature engineering...")
    df = df.copy()

    # Hitung jumlah kata
    df['Word_Count'] = df['Clean_Text'].apply(lambda x: len(str(x).split()))
    print(f"   ✅ Kolom Word_Count berhasil dibuat")
    print(f"      Rata-rata : {df['Word_Count'].mean():.0f} kata")
    print(f"      Min       : {df['Word_Count'].min()} kata")
    print(f"      Maks      : {df['Word_Count'].max()} kata")

    # Buat ID unik
    df.insert(0, 'ID_CV', ['CV_' + str(i).zfill(4) for i in range(1, len(df) + 1)])
    print(f"   ✅ Kolom ID_CV berhasil dibuat (CV_0001 s/d CV_{len(df):04d})")

    return df


# ==============================================================
# TAHAP 5 — DEDUPLICATION & RESTRUKTURISASI
# ==============================================================

def deduplikasi(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris dengan teks identik (CV yang sama di-upload dua kali)."""
    sebelum = len(df)
    df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)
    sesudah = len(df)

    if sebelum != sesudah:
        print(f"   ⚠️  {sebelum - sesudah} baris duplikat dihapus")
    else:
        print(f"   ✅ Tidak ada duplikat ditemukan")

    return df


def restrukturisasi_dan_export(df: pd.DataFrame) -> pd.DataFrame:
    """Susun urutan kolom final dan export ke Excel."""
    print("\n🔧 Restrukturisasi kolom...")

    kolom_final = ['ID_CV', 'Category', 'Word_Count', 'Clean_Text', 'Text']
    df_final = df[kolom_final].copy()

    # Reset ID setelah deduplikasi (agar ID tetap urut)
    df_final['ID_CV'] = ['CV_' + str(i).zfill(4) for i in range(1, len(df_final) + 1)]

    print(f"   ✅ Urutan kolom: {kolom_final}")
    print(f"   ✅ Total baris final: {len(df_final):,}")

    # Export
    print(f"\n💾 Menyimpan ke: {OUTPUT_FILE}")
    df_final.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"   ✅ File berhasil disimpan!")

    return df_final


# ==============================================================
# MAIN — Jalankan pipeline dari awal sampai akhir
# ==============================================================

def main():
    # Tahap 1: Inisialisasi
    inisialisasi()

    # Tahap 2: Ekstraksi teks dari semua PDF
    print("\n📄 TAHAP 2 — EKSTRAKSI TEKS (OCR)")
    print("-" * 40)
    semua_data = ekstrak_semua_kategori()

    if not semua_data:
        print("❌ Tidak ada data yang berhasil diekstrak. Periksa folder data/raw/")
        return

    df = pd.DataFrame(semua_data)

    # Tahap 3: Pembersihan teks
    print("\n📄 TAHAP 3 — PEMBERSIHAN TEKS")
    print("-" * 40)
    df = terapkan_cleaning(df)

    # Tahap 4: Deduplikasi (sebelum feature engineering)
    print("\n📄 TAHAP 4 — DEDUPLIKASI")
    print("-" * 40)
    df = deduplikasi(df)

    # Tahap 5: Feature engineering
    print("\n📄 TAHAP 5 — FEATURE ENGINEERING")
    print("-" * 40)
    df = feature_engineering(df)

    # Tahap 6: Export
    print("\n📄 TAHAP 6 — EXPORT")
    print("-" * 40)
    df_final = restrukturisasi_dan_export(df)

    # Ringkasan akhir
    print("\n" + "=" * 60)
    print(" PIPELINE SELESAI — RINGKASAN HASIL")
    print("=" * 60)
    print(f"  Total CV diproses : {len(df_final):,}")
    print(f"  Jumlah kategori   : {df_final['Category'].nunique()}")
    print()
    print(df_final['Category'].value_counts().to_string())
    print()
    print(f"  Output disimpan di: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()
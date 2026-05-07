"""
==============================================================
 data_cleaning.py — Capstone Project: CV Classification (NLP)
==============================================================
 Pipeline cleaning dataset CV dari Excel yang sudah ada:
   TAHAP 1 : Load dataset dari Excel
   TAHAP 2 : Pembersihan & normalisasi teks
   TAHAP 3 : Deduplikasi
   TAHAP 4 : Feature engineering
   TAHAP 5 : Restrukturisasi & export

 Cara pakai:
   python src/data_cleaning.py
   Output: data/processed/Dataset_NLP_Siap_Model.xlsx
==============================================================
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pastikan resource NLTK tersedia
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


# ==============================================================
# KONFIGURASI PATH
# ==============================================================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'Dataset_NLP_Siap_Model.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'Dataset_NLP_Siap_Model_V2.xlsx')

# ==============================================================
# TAHAP 1 — LOAD DATASET
# ==============================================================

def load_data() -> pd.DataFrame:
    print("=" * 60)
    print(" CAPSTONE — CV CLASSIFICATION DATA CLEANING PIPELINE")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_FILE}\n"
            "Pastikan file Dataset_NLP_Siap_Model.xlsx ada di folder data/processed/"
        )

    print(f"\nTAHAP 1 — LOAD DATA")
    print("-" * 40)
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    print(f" FILE BERHASIL DIMUAT: {INPUT_FILE}")
    print(f"   Jumlah baris  : {df.shape[0]:,}")
    print(f"   Jumlah kolom  : {df.shape[1]}")
    print(f"   Kolom         : {list(df.columns)}")
    return df


# ==============================================================
# TAHAP 2 — PEMBERSIHAN & NORMALISASI TEKS
# ==============================================================

def bersihkan_teks(teks_mentah: str) -> str:
    """
    Bersihkan teks untuk keperluan pemodelan NLP:
    - Lowercase semua teks
    - Hapus URL, Email, dan Angka (PII/Noise)
    - Hapus simbol/tanda baca (hanya sisakan alfabet)
    - Hapus Stopwords (kata umum bahasa Inggris)
    - Terapkan Lemmatization (kembalikan kata ke bentuk dasar)
    """
    if not isinstance(teks_mentah, str):
        teks_mentah = str(teks_mentah)

    # 1. Lowercase
    teks = teks_mentah.lower()

    # 2. Hapus Email
    teks = re.sub(r'\S+@\S+', ' ', teks)

    # 3. Hapus URL/Links
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)

    # 4. Hapus karakter selain alfabet (angka, tanda baca, simbol dihapus)
    teks = re.sub(r'[^a-z\s]', ' ', teks)

    # 5. Tokenisasi kata, hapus stopwords & kata pendek (< 3 huruf), lalu Lemmatize
    words = teks.split()
    cleaned_words = [
        LEMMATIZER.lemmatize(w) for w in words 
        if w not in STOPWORDS and len(w) > 2
    ]

    return ' '.join(cleaned_words)


def terapkan_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nTAHAP 2 — PEMBERSIHAN TEKS")
    print("-" * 40)

    df = df.copy()

    if 'Text' not in df.columns:
        raise KeyError("Kolom 'Text' tidak ditemukan di dataset.")

    df['Clean_Text'] = df['Text'].apply(bersihkan_teks)

    rata_sebelum = df['Text'].apply(len).mean()
    rata_sesudah = df['Clean_Text'].apply(len).mean()
    print(f" KOLOM Clean_Text berhasil dibuat")
    print(f"   Rata-rata panjang sebelum : {rata_sebelum:.0f} karakter")
    print(f"   Rata-rata panjang sesudah : {rata_sesudah:.0f} karakter")

    return df


# ==============================================================
# TAHAP 3 — DEDUPLIKASI
# ==============================================================

def deduplikasi(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nTAHAP 3 — DEDUPLIKASI")
    print("-" * 40)

    sebelum = len(df)
    df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)
    sesudah = len(df)

    if sebelum != sesudah:
        print(f"{sebelum - sesudah} baris duplikat dihapus")
    else:
        print(f"TIDAK ADA DUPLIKAT DITEMUKAN")

    print(f"   Sisa data: {sesudah:,} baris")
    return df


# ==============================================================
# TAHAP 4 — FEATURE ENGINEERING
# ==============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nTAHAP 4 — FEATURE ENGINEERING")
    print("-" * 40)

    df = df.copy()

    # Word Count dari Clean_Text
    df['Word_Count'] = df['Clean_Text'].apply(lambda x: len(str(x).split()))
    print(f" KOLOM Word_Count berhasil dibuat ")
    print(f"   Rata-rata : {df['Word_Count'].mean():.0f} kata")
    print(f"   Min       : {df['Word_Count'].min()} kata")
    print(f"   Maks      : {df['Word_Count'].max()} kata")

    # ID unik per CV (jika belum ada)
    if 'ID_CV' not in df.columns:
        df.insert(0, 'ID_CV', ['CV_' + str(i).zfill(4) for i in range(1, len(df) + 1)])
        print(f"KOLOM ID_CV BERHASIL DIBUAT (CV_0001 s/d CV_{len(df):04d})")
    else:
        print(f" KOLOM ID_CV SUDAH ADA.")

    return df


# ==============================================================
# TAHAP 5 — RESTRUKTURISASI & EXPORT
# ==============================================================

def restrukturisasi_dan_export(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nTAHAP 5 — RESTRUKTURISASI & EXPORT")
    print("-" * 40)

    kolom_final = ['ID_CV', 'Category', 'Word_Count', 'Clean_Text', 'Text']

    kolom_kurang = [k for k in kolom_final if k not in df.columns]
    if kolom_kurang:
        raise KeyError(f"Kolom berikut tidak ditemukan di dataset: {kolom_kurang}")

    df_final = df[kolom_final].copy()

    # Reset ID setelah deduplikasi agar tetap urut
    df_final['ID_CV'] = ['CV_' + str(i).zfill(4) for i in range(1, len(df_final) + 1)]

    print(f" URUTAN KOLOM FINAL : {kolom_final}")
    print(f" TOTAL BARIS FINAL  : {len(df_final):,}")

    print(f"\nMENYIMPAN FILE KE: {OUTPUT_FILE}")
    df_final.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"✅ FILE BERHASI DISIMPAN!!")

    return df_final


# ==============================================================
# MAIN
# ==============================================================

def main():
    df = load_data()
    df = terapkan_cleaning(df)
    df = deduplikasi(df)
    df = feature_engineering(df)
    df_final = restrukturisasi_dan_export(df)

    print("\n" + "=" * 60)
    print(" PIPELINE SELESAI — RINGKASAN HASIL")
    print("=" * 60)
    print(f"  Total CV : {len(df_final):,}")
    print(f"  Kategori : {df_final['Category'].nunique()}")
    print()
    print(df_final['Category'].value_counts().to_string())
    print()
    print(f"  Output   : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()

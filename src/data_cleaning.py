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

# ==============================================================
# KONFIGURASI PATH
# ==============================================================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'raw', 'Dataset_CV.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'Dataset_NLP_Siap_Model.xlsx')

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
            "Pastikan file Dataset_CV.xlsx ada di folder data/raw/"
        )

    print(f"\n📄 TAHAP 1 — LOAD DATA")
    print("-" * 40)
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    print(f"✅ File berhasil dimuat: {INPUT_FILE}")
    print(f"   Jumlah baris  : {df.shape[0]:,}")
    print(f"   Jumlah kolom  : {df.shape[1]}")
    print(f"   Kolom         : {list(df.columns)}")
    return df


# ==============================================================
# TAHAP 2 — PEMBERSIHAN & NORMALISASI TEKS
# ==============================================================

def bersihkan_teks(teks_mentah: str) -> str:
    """
    Bersihkan teks hasil OCR:
    - Hapus karakter unicode tersembunyi (BOM, ZWNJ, dll)
    - Hapus simbol noise, sisakan huruf/angka/tanda baca penting
    - Rapikan spasi berlebihan
    """
    if not isinstance(teks_mentah, str):
        teks_mentah = str(teks_mentah)

    # Hapus karakter unicode tersembunyi
    teks = re.sub(r'[\ufeff\u200b\u200c\u200d\u00a0]', ' ', teks_mentah)

    # Hapus simbol noise OCR, sisakan: huruf, angka, spasi, dan tanda baca penting
    teks = re.sub(r'[^a-zA-Z0-9\s.,\-\/|@+:;()]', ' ', teks)

    # Rapikan spasi berlebihan
    teks = re.sub(r'\s+', ' ', teks).strip()

    return teks


def terapkan_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n📄 TAHAP 2 — PEMBERSIHAN TEKS")
    print("-" * 40)

    df = df.copy()

    if 'Text' not in df.columns:
        raise KeyError("Kolom 'Text' tidak ditemukan di dataset.")

    df['Clean_Text'] = df['Text'].apply(bersihkan_teks)

    rata_sebelum = df['Text'].apply(len).mean()
    rata_sesudah = df['Clean_Text'].apply(len).mean()
    print(f"✅ Kolom Clean_Text berhasil dibuat")
    print(f"   Rata-rata panjang sebelum : {rata_sebelum:.0f} karakter")
    print(f"   Rata-rata panjang sesudah : {rata_sesudah:.0f} karakter")

    return df


# ==============================================================
# TAHAP 3 — DEDUPLIKASI
# ==============================================================

def deduplikasi(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n📄 TAHAP 3 — DEDUPLIKASI")
    print("-" * 40)

    sebelum = len(df)
    df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)
    sesudah = len(df)

    if sebelum != sesudah:
        print(f"⚠️  {sebelum - sesudah} baris duplikat dihapus")
    else:
        print(f"✅ Tidak ada duplikat ditemukan")

    print(f"   Sisa data: {sesudah:,} baris")
    return df


# ==============================================================
# TAHAP 4 — FEATURE ENGINEERING
# ==============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n📄 TAHAP 4 — FEATURE ENGINEERING")
    print("-" * 40)

    df = df.copy()

    # Word Count dari Clean_Text
    df['Word_Count'] = df['Clean_Text'].apply(lambda x: len(str(x).split()))
    print(f"✅ Kolom Word_Count berhasil dibuat")
    print(f"   Rata-rata : {df['Word_Count'].mean():.0f} kata")
    print(f"   Min       : {df['Word_Count'].min()} kata")
    print(f"   Maks      : {df['Word_Count'].max()} kata")

    # ID unik per CV
    df.insert(0, 'ID_CV', ['CV_' + str(i).zfill(4) for i in range(1, len(df) + 1)])
    print(f"✅ Kolom ID_CV berhasil dibuat (CV_0001 s/d CV_{len(df):04d})")

    return df


# ==============================================================
# TAHAP 5 — RESTRUKTURISASI & EXPORT
# ==============================================================

def restrukturisasi_dan_export(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n📄 TAHAP 5 — RESTRUKTURISASI & EXPORT")
    print("-" * 40)

    kolom_final = ['ID_CV', 'Category', 'Word_Count', 'Clean_Text', 'Text']

    kolom_kurang = [k for k in kolom_final if k not in df.columns]
    if kolom_kurang:
        raise KeyError(f"Kolom berikut tidak ditemukan di dataset: {kolom_kurang}")

    df_final = df[kolom_final].copy()

    # Reset ID setelah deduplikasi agar tetap urut
    df_final['ID_CV'] = ['CV_' + str(i).zfill(4) for i in range(1, len(df_final) + 1)]

    print(f"✅ Urutan kolom final : {kolom_final}")
    print(f"✅ Total baris final  : {len(df_final):,}")

    print(f"\n💾 Menyimpan ke: {OUTPUT_FILE}")
    df_final.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"✅ File berhasil disimpan!")

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

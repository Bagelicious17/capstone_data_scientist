import pandas as pd
import re

print("1. Memuat dataset...")
lokasi_file = 'Dataset_CV_Final_Siap_Training.xlsx'
df = pd.read_excel(lokasi_file)
print(f"Bentuk awal: {df.shape[0]} Baris")

# --- TAHAP 1: DATA DEDUPLICATION ---
print("\n2. Menghapus data duplikat...")
df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)
print(f"Sisa data setelah hapus duplikat: {df.shape[0]} Baris")

# --- TAHAP 2: TEXT NORMALIZATION (Pembersihan Teks Lanjutan) ---
print("3. Membersihkan karakter aneh dan merapikan teks...")
# Copy data ke kolom baru agar raw data tetap aman jika dibutuhkan
df['Clean_Text'] = df['Text'].astype(str)

# Hapus karakter unicode tersembunyi (\ufeff dll)
df['Clean_Text'] = df['Clean_Text'].str.replace(r'\ufeff', '', regex=False)

# Hapus simbol-simbol aneh bawaan OCR tapi sisakan tanda baca penting
# (Hanya menyisakan huruf, angka, spasi, titik, koma, strip, dan garis miring)
df['Clean_Text'] = df['Clean_Text'].str.replace(r'[^a-zA-Z0-9\s.,\-\/|]', ' ', regex=True)

# Rapikan spasi yang berlebihan
df['Clean_Text'] = df['Clean_Text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# --- TAHAP 3: FEATURE ENGINEERING (Penambahan Fitur) ---
print("4. Menghitung jumlah kata (Word Count)...")
df['Word_Count'] = df['Clean_Text'].apply(lambda x: len(str(x).split()))

# --- TAHAP 4: RESTRUKTURISASI KOLOM ---
print("5. Menyusun ulang urutan kolom sesuai standar industri...")
# Bikin kolom ID unik untuk setiap CV
df.insert(0, 'ID_CV', ['CV_' + str(i).zfill(4) for i in range(1, len(df) + 1)])

# Susun urutan kolomnya biar cantik pas dibuka di Excel
kolom_final = ['ID_CV', 'Category', 'Word_Count', 'Clean_Text', 'Text']
df_final = df[kolom_final]

# Export ke Excel siap pakai
nama_output = 'Dataset_NLP_Siap_Model.xlsx'
df_final.to_excel(nama_output, index=False, engine='openpyxl')

print(f"\nSUKSES BESAR! Dataset profesional lu udah tersimpan di '{nama_output}'")
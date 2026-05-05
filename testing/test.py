import os
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import re

# ==========================================
# 1. KONFIGURASI AWAL
# ==========================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
path_ke_poppler = r'C:\poppler-25.12.0\Library\bin'
folder_pdf = r'capstone_data_scientist/data/raw/Network_Security_Engineer'
konfigurasi_custom = r'--psm 4'

# ==========================================
# 2. FUNGSI PARSING (Sudah disesuaikan)
# ==========================================
def parse_resume_text(raw_text):
    """Fungsi merapikan teks mentah CV menjadi Dictionary"""
    raw_text = re.sub(r'[©«¢\(\)]', '', raw_text)
    
    data_bersih = {
        "Nama_dan_Kontak": None,
        "Objective": None,
        "Executive_Summary": None,
        "Areas_of_Expertise": None,
        "Product_Portfolio": None,
        "Experience": None,
        "Education_and_Training": None
    }
    
    try:
        nama_kontak = re.split(r'\bOBJECTIVE\b', raw_text)[0]
        data_bersih["Nama_dan_Kontak"] = nama_kontak.strip()
        data_bersih["Objective"] = re.search(r'OBJECTIVE(.*?)EXECUTIVE SUMMARY', raw_text, re.DOTALL).group(1).strip()
        data_bersih["Executive_Summary"] = re.search(r'EXECUTIVE SUMMARY(.*?)AREAS OF EXPERTISE', raw_text, re.DOTALL).group(1).strip()
        data_bersih["Areas_of_Expertise"] = re.search(r'AREAS OF EXPERTISE(.*?)PRODUCT\s*PORTFOLIO', raw_text, re.DOTALL).group(1).strip()
        data_bersih["Product_Portfolio"] = re.search(r'PRODUCT\s*PORTFOLIO(.*?)EXPERIENCE', raw_text, re.DOTALL).group(1).strip()
        data_bersih["Experience"] = re.search(r'EXPERIENCE(.*?)EDUCATION AND', raw_text, re.DOTALL).group(1).strip()
        data_bersih["Education_and_Training"] = re.split(r'EDUCATION AND', raw_text)[-1].strip()
    except Exception as e:
        # Pass aja, biar kalau ada 1 kolom gagal, kolom lain tetep keisi
        pass 
        
    return data_bersih

# ==========================================
# 3. PROSES LOOPING UTAMA 
# ==========================================
semua_data_cv = [] # List kosong untuk menampung 200 data CV
total_file = 100

print(f"Mulai memproses {total_file} PDF sekaligus...")

for i in range(total_file):
    nama_file = f"image_{i}.pdf"
    lokasi_pdf = os.path.join(folder_pdf, nama_file)
    
    # Keamanan: Cek apakah filenya beneran ada di folder
    if not os.path.exists(lokasi_pdf):
        print(f"Warning: File {nama_file} tidak ditemukan, skip...")
        continue
        
    print(f"[{i+1}/{total_file}] Sedang mengekstrak: {nama_file}...")
    
    try:
        # A. Proses Ekstrak Gambar ke Teks (OCR)
        halaman_gambar = convert_from_path(lokasi_pdf, poppler_path=path_ke_poppler)
        halaman_pertama = halaman_gambar[0]
        hasil_teks = pytesseract.image_to_string(halaman_pertama, config=konfigurasi_custom)
        
        # B. Proses Rapikan Teks pakai Regex
        data_terstruktur = parse_resume_text(hasil_teks)
        
        # C. Tambahkan kolom "Nama_File" biar lu gampang nge-track ini CV siapa
        data_terstruktur["Nama_File"] = nama_file
        
        # D. Masukkan hasil CV ini ke dalam koper besar kita
        semua_data_cv.append(data_terstruktur)
        
    except Exception as e:
        print(f"  -> ERROR saat memproses {nama_file}: {e}")

# ==========================================
# 4. TAHAP LOAD (EXPORT KE EXCEL)
# ==========================================
print("\nSemua file selesai diproses! Menyiapkan file Excel...")

# Ubah List jadi DataFrame
df_final = pd.DataFrame(semua_data_cv)

# Pindahkan kolom "Nama_File" ke paling depan (opsional biar rapi aja)
kolom_urutan = ['Nama_File'] + [kolom for kolom in df_final.columns if kolom != 'Nama_File']
df_final = df_final[kolom_urutan]

# Export!
nama_output = "hasil_ekstrak_200_cv_network_security_engineer.xlsx"
df_final.to_excel(nama_output, index=False, engine='openpyxl')

print(f"SUKSES BESAR! {len(semua_data_cv)} CV berhasil diekstrak dan disimpan ke '{nama_output}'")